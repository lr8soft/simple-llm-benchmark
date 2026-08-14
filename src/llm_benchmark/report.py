from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .process import utf8_subprocess_env
from .runner import inspect_executable


class ReportError(RuntimeError):
    """Raised when logs cannot be converted into a report."""


@dataclass(frozen=True)
class BenchmarkResult:
    id: str
    label: str
    dimension: str
    status: str
    score: float | None
    stderr: float | None
    samples: int | None
    completed_samples: int | None
    metric: str | None
    weight: float
    log_file: str | None
    error: str | None


def _load_log(path: Path, *, header_only: bool = True) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    inspect = inspect_executable()
    if not inspect:
        raise ReportError(f"读取 {path.name} 需要 Inspect CLI")
    command = [inspect, "log", "dump"]
    if header_only:
        command.append("--header-only")
    command.append(str(path))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=utf8_subprocess_env(),
    )
    if completed.returncode != 0:
        raise ReportError(f"Inspect 无法读取 {path}: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def _error_message(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        message = value.get("message")
        return message if isinstance(message, str) else None
    return None


def _compact_error(value: str | None, limit: int = 600) -> str | None:
    if not value:
        return None
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _extract_error(header: dict[str, Any], log_path: Path) -> str | None:
    if str(header.get("status", "unknown")) == "success":
        return None
    try:
        detail = _load_log(log_path, header_only=False)
    except (ReportError, json.JSONDecodeError):
        detail = header

    messages: list[str] = []
    for sample in detail.get("samples") or []:
        for event in sample.get("events") or []:
            message = _error_message(event.get("error"))
            if message and "CancelledError" not in message:
                messages.append(message)
    # Model/API failures contain the useful HTTP response while Inspect's
    # top-level RetryError often hides it. Prefer these detailed messages.
    if messages:
        preferred = next((message for message in messages if "Error code:" in message), messages[0])
        return _compact_error(preferred)
    return _compact_error(_error_message(detail.get("error")) or _error_message(header.get("error")))


def _metric_value(metric: Any) -> float | None:
    if isinstance(metric, (int, float)) and not isinstance(metric, bool):
        return float(metric)
    if isinstance(metric, dict):
        value = metric.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _named_metric(metrics: dict[str, Any], path: str) -> float | None:
    """Read either a normal metric or a member of a multi-value metric.

    Inspect's IFEval metric, for example, is serialized as
    ``if_metric -> value -> prompt_strict_acc``.
    """
    parts = path.split(".")
    root_name = next((name for name in metrics if str(name).lower() == parts[0].lower()), None)
    if root_name is None:
        return None
    current: Any = metrics[root_name]
    if len(parts) == 1:
        return _metric_value(current)
    if isinstance(current, dict) and "value" in current:
        current = current["value"]
    for part in parts[1:]:
        if not isinstance(current, dict):
            return None
        key = next((name for name in current if str(name).lower() == part.lower()), None)
        if key is None:
            return None
        current = current[key]
    return _metric_value(current)


def _extract_score(log: dict[str, Any], benchmark: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    results = log.get("results") or {}
    scores = results.get("scores") or []
    scorer_filter = benchmark.get("scorer_contains")
    candidates = benchmark.get("metric_names") or ["accuracy", "mean"]
    normalized = [str(name).lower() for name in candidates]

    for wanted in normalized:
        for score in scores:
            haystack = f"{score.get('name', '')} {score.get('scorer', '')}".lower()
            if scorer_filter and scorer_filter.lower() not in haystack:
                continue
            metrics = score.get("metrics") or {}
            value = _named_metric(metrics, wanted)
            if value is None:
                continue
            stderr = None
            if "." in wanted and wanted.endswith("_acc"):
                stderr = _named_metric(metrics, wanted[:-4] + "_stderr")
            if stderr is None:
                for stderr_name in ("stderr", "std_error", "standard_error"):
                    stderr = _named_metric(metrics, stderr_name)
                    if stderr is not None:
                        break
            return value, stderr, wanted
    return None, None, None


def _latest_log(directory: Path) -> Path | None:
    files = [*directory.rglob("*.eval"), *directory.rglob("*.json")]
    files = [path for path in files if path.name not in {"manifest.json", "run-status.json", "results.json"}]
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def collect_results(run_dir: str | Path) -> tuple[dict[str, Any], list[BenchmarkResult]]:
    root = Path(run_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ReportError(f"缺少 {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collected: list[BenchmarkResult] = []

    for benchmark in manifest.get("benchmarks", []):
        benchmark_id = benchmark["id"]
        log_path = _latest_log(root / "logs" / benchmark_id)
        if log_path is None:
            collected.append(BenchmarkResult(
                id=benchmark_id,
                label=benchmark["label"],
                dimension=benchmark["dimension"],
                status="missing",
                score=None,
                stderr=None,
                samples=None,
                completed_samples=None,
                metric=None,
                weight=float(benchmark["weight"]),
                log_file=None,
                error="未生成 Inspect 日志",
            ))
            continue
        log = _load_log(log_path)
        score, stderr, metric = _extract_score(log, benchmark)
        results = log.get("results") or {}
        collected.append(BenchmarkResult(
            id=benchmark_id,
            label=benchmark["label"],
            dimension=benchmark["dimension"],
            status=str(log.get("status", "unknown")),
            score=score,
            stderr=stderr,
            samples=_as_int(results.get("total_samples")),
            completed_samples=_as_int(results.get("completed_samples")),
            metric=metric,
            weight=float(benchmark["weight"]),
            log_file=str(log_path.relative_to(root)),
            error=_extract_error(log, log_path),
        ))
    return manifest, collected


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _ci(stderr: float | None) -> str:
    return "—" if stderr is None else f"± {1.96 * stderr * 100:.2f}pp"


def render_report(manifest: dict[str, Any], results: list[BenchmarkResult]) -> tuple[str, dict[str, Any]]:
    scored = [item for item in results if item.score is not None and item.status == "success"]
    available_weight = sum(item.weight for item in scored)
    overall = (
        sum(item.score * item.weight for item in scored) / available_weight
        if available_weight > 0
        else None
    )
    complete = len(scored) == len(results)

    lines = [
        "# LLM Benchmark Report",
        "",
        f"- Model: `{manifest['model']['model_id']}`",
        f"- Suite: `{manifest['suite']}`",
        f"- Created: `{manifest['created_at']}`",
        f"- Overall quality: **{_percent(overall)}**" + ("" if complete else "（仅按已完成项目重新归一化）"),
        f"- Coverage: **{len(scored)}/{len(results)} benchmarks**",
        "",
        "| Dimension | Benchmark | Score | 95% CI | Samples | Status | Weight |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for item in results:
        sample_text = "—"
        if item.completed_samples is not None or item.samples is not None:
            sample_text = f"{item.completed_samples or 0}/{item.samples or '?'}"
        lines.append(
            f"| {item.dimension} | {item.label} | {_percent(item.score)} | {_ci(item.stderr)} "
            f"| {sample_text} | {item.status} | {item.weight * 100:.0f}% |"
        )

    failures = [item for item in results if item.status != "success" or item.score is None]
    if failures:
        lines.extend(["", "## Failures", ""])
        for item in failures:
            lines.append(f"- `{item.id}`: {item.error or '任务未成功或没有可用指标'}")

    lines.extend([
        "",
        "## Reproducibility",
        "",
        f"- Provider: `{manifest['model']['provider']}`",
        f"- Base URL: `{manifest['model']['base_url']}`",
        f"- Temperature: `{manifest['run']['temperature']}`",
        f"- Seed: `{manifest['run']['seed']}`",
        f"- Max connections: `{manifest['run']['max_connections']}`",
        f"- Max retries: `{manifest['run']['max_retries']}`",
        f"- Request timeout: `{manifest['run']['timeout']}s`",
        "",
        "> Overall 是显式权重的便利指标；跨模型比较时还应同时检查各子项、覆盖率和运行配置。",
        "",
    ])
    payload = {
        "schema_version": 1,
        "model": manifest["model"],
        "suite": manifest["suite"],
        "overall": overall,
        "complete": complete,
        "coverage": {"scored": len(scored), "total": len(results)},
        "benchmarks": [item.__dict__ for item in results],
    }
    return "\n".join(lines), payload


def write_report(run_dir: str | Path) -> tuple[Path, Path]:
    root = Path(run_dir).resolve()
    manifest, results = collect_results(root)
    markdown, payload = render_report(manifest, results)
    markdown_path = root / "report.md"
    json_path = root / "results.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [item.id for item in results if item.status != "success" or item.score is None]
    (root / "run-status.json").write_text(
        json.dumps({"failed": failed, "complete": not failed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return markdown_path, json_path
