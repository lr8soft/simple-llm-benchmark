from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import AppConfig, BenchmarkConfig


class RunnerError(RuntimeError):
    """Raised when the external evaluation runner cannot be used."""


def inspect_executable() -> str | None:
    on_path = shutil.which("inspect")
    if on_path:
        return on_path
    # Running an entry-point executable directly does not necessarily add the
    # virtualenv's Scripts directory to PATH (common in PyCharm on Windows).
    sibling = Path(sys.executable).with_name("inspect.exe" if os.name == "nt" else "inspect")
    return str(sibling) if sibling.exists() else None


def selected_benchmarks(config: AppConfig, only: Iterable[str] | None) -> list[BenchmarkConfig]:
    if not only:
        return list(config.benchmarks)
    requested = set(only)
    known = {benchmark.id for benchmark in config.benchmarks}
    unknown = requested - known
    if unknown:
        raise RunnerError(f"suite 中不存在：{', '.join(sorted(unknown))}")
    return [benchmark for benchmark in config.benchmarks if benchmark.id in requested]


def build_command(
    inspect: str,
    config: AppConfig,
    benchmark: BenchmarkConfig,
    log_dir: Path,
) -> list[str]:
    command = [
        inspect,
        "eval",
        benchmark.task,
        "--model",
        config.model.inspect_model,
        "--model-base-url",
        config.model.base_url,
        "--log-dir",
        str(log_dir),
        "--log-format",
        "eval",
        "--display",
        "plain",
        "--max-connections",
        str(config.run.max_connections),
        "--max-retries",
        str(config.run.max_retries),
        "--timeout",
        str(config.run.timeout),
        "--temperature",
        str(config.run.temperature),
        "--seed",
        str(config.run.seed),
    ]
    if benchmark.limit is not None:
        command.extend(["--limit", str(benchmark.limit)])
    for key, value in benchmark.task_args.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            rendered = str(value)
        command.extend(["-T", f"{key}={rendered}"])
    return command


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value).strip("-")[:80]


def _manifest(config: AppConfig, benchmarks: list[BenchmarkConfig]) -> dict:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": config.run.suite,
        "model": {
            "model_id": config.model.model_id,
            "base_url": config.model.base_url,
            "provider": config.model.provider,
        },
        "run": {
            "max_connections": config.run.max_connections,
            "max_retries": config.run.max_retries,
            "timeout": config.run.timeout,
            "temperature": config.run.temperature,
            "seed": config.run.seed,
        },
        "benchmarks": [asdict(item) for item in benchmarks],
    }


def create_run_dir(config: AppConfig, benchmarks: list[BenchmarkConfig]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.run.output_dir / f"{stamp}-{_safe_slug(config.model.model_id)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "manifest.json").write_text(
        json.dumps(_manifest(config, benchmarks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def run_suite(
    config: AppConfig,
    only: Iterable[str] | None = None,
    dry_run: bool = False,
) -> Path | None:
    benchmarks = selected_benchmarks(config, only)
    executable = inspect_executable() or "inspect"
    if not dry_run and inspect_executable() is None:
        raise RunnerError('未找到 Inspect CLI；请执行 pip install -e ".[inspect]"')

    if dry_run:
        for benchmark in benchmarks:
            command = build_command(executable, config, benchmark, Path("<run-dir>") / "logs" / benchmark.id)
            print(" ".join(_quote(part) for part in command))
        return None

    run_dir = create_run_dir(config, benchmarks)
    env = os.environ.copy()
    prefix = config.model.provider_env_prefix
    env[f"{prefix}_API_KEY"] = config.model.api_key
    env[f"{prefix}_BASE_URL"] = config.model.base_url

    failures: list[str] = []
    for benchmark in benchmarks:
        log_dir = run_dir / "logs" / benchmark.id
        log_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(executable, config, benchmark, log_dir)
        print(f"\n[{benchmark.id}] {benchmark.label}", flush=True)
        completed = subprocess.run(command, env=env, check=False)
        if completed.returncode != 0:
            failures.append(benchmark.id)

    (run_dir / "run-status.json").write_text(
        json.dumps({"failed": failures, "complete": len(failures) == 0}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def _quote(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return json.dumps(value, ensure_ascii=False)
    return value
