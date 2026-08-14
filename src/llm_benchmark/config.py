from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a benchmark configuration is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    base_url: str
    api_key: str
    provider: str = "benchmark"

    @property
    def inspect_model(self) -> str:
        return f"openai-api/{self.provider}/{self.model_id}"

    @property
    def provider_env_prefix(self) -> str:
        return self.provider.upper().replace("-", "_")


@dataclass(frozen=True)
class BenchmarkConfig:
    id: str
    label: str
    dimension: str
    task: str
    weight: float
    limit: int | None = None
    task_args: dict[str, Any] = field(default_factory=dict)
    metric_names: tuple[str, ...] = ("accuracy", "mean")
    scorer_contains: str | None = None


@dataclass(frozen=True)
class RunConfig:
    suite: str
    output_dir: Path
    max_connections: int = 8
    max_retries: int = 3
    timeout: int = 120
    temperature: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class AppConfig:
    version: int
    model: ModelConfig
    run: RunConfig
    benchmarks: tuple[BenchmarkConfig, ...]
    source_path: Path


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} 必须是对象")
    return value


def load_config(path: str | Path) -> AppConfig:
    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"配置文件不存在：{source_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 无法解析：{exc}") from exc

    root = _mapping(raw, "根配置")
    version = root.get("version")
    if version != 1:
        raise ConfigError("当前仅支持 version: 1")

    model_raw = _mapping(root.get("model"), "model")
    required_model = ("model_id", "base_url", "api_key")
    missing = [key for key in required_model if not model_raw.get(key)]
    if missing:
        raise ConfigError(f"model 缺少字段：{', '.join(missing)}")
    provider = str(model_raw.get("provider", "benchmark"))
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", provider):
        raise ConfigError("model.provider 只能包含小写字母、数字、_ 和 -")
    model = ModelConfig(
        model_id=str(model_raw["model_id"]),
        base_url=str(model_raw["base_url"]).rstrip("/"),
        api_key=str(model_raw["api_key"]),
        provider=provider,
    )

    run_raw = _mapping(root.get("run", {}), "run")
    suite_name = str(run_raw.get("suite", "quick-v1"))
    suites_raw = _mapping(root.get("suites"), "suites")
    if suite_name not in suites_raw:
        raise ConfigError(f"未找到 suite：{suite_name}")
    suite_raw = _mapping(suites_raw[suite_name], f"suites.{suite_name}")
    items = suite_raw.get("benchmarks")
    if not isinstance(items, list) or not items:
        raise ConfigError(f"suites.{suite_name}.benchmarks 必须是非空列表")

    benchmarks: list[BenchmarkConfig] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        data = _mapping(item, f"benchmarks[{index}]")
        benchmark_id = str(data.get("id", ""))
        if not benchmark_id or benchmark_id in seen:
            raise ConfigError(f"benchmark id 为空或重复：{benchmark_id!r}")
        seen.add(benchmark_id)
        try:
            weight = float(data["weight"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{benchmark_id}.weight 必须是数字") from exc
        if weight <= 0:
            raise ConfigError(f"{benchmark_id}.weight 必须大于 0")
        limit = data.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ConfigError(f"{benchmark_id}.limit 必须是正整数或 null")
        metric_names = data.get("metric_names", ["accuracy", "mean"])
        if not isinstance(metric_names, list) or not metric_names:
            raise ConfigError(f"{benchmark_id}.metric_names 必须是非空列表")
        benchmarks.append(
            BenchmarkConfig(
                id=benchmark_id,
                label=str(data.get("label", benchmark_id)),
                dimension=str(data.get("dimension", benchmark_id)),
                task=str(data.get("task", "")),
                weight=weight,
                limit=limit,
                task_args=_mapping(data.get("task_args", {}), f"{benchmark_id}.task_args"),
                metric_names=tuple(str(name) for name in metric_names),
                scorer_contains=(
                    str(data["scorer_contains"])
                    if data.get("scorer_contains") is not None
                    else None
                ),
            )
        )

    total_weight = sum(item.weight for item in benchmarks)
    if abs(total_weight - 1.0) > 1e-6:
        raise ConfigError(f"suite 权重之和必须为 1.0，当前为 {total_weight:g}")

    output_dir = Path(str(run_raw.get("output_dir", "runs")))
    if not output_dir.is_absolute():
        output_dir = source_path.parent / output_dir
    run = RunConfig(
        suite=suite_name,
        output_dir=output_dir.resolve(),
        max_connections=int(run_raw.get("max_connections", 8)),
        max_retries=int(run_raw.get("max_retries", 3)),
        timeout=int(run_raw.get("timeout", 120)),
        temperature=float(run_raw.get("temperature", 0.0)),
        seed=int(run_raw.get("seed", 42)),
    )
    if run.max_connections <= 0:
        raise ConfigError("run.max_connections 必须大于 0")
    if run.max_retries < 0:
        raise ConfigError("run.max_retries 不能小于 0")
    if run.timeout <= 0:
        raise ConfigError("run.timeout 必须大于 0")

    return AppConfig(version=version, model=model, run=run, benchmarks=tuple(benchmarks), source_path=source_path)
