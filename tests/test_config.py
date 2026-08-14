from pathlib import Path

import pytest

from llm_benchmark.config import ConfigError, load_config


ROOT = Path(__file__).parents[1]


def test_example_config_is_valid() -> None:
    config = load_config(ROOT / "benchmark.example.yaml")
    assert config.model.inspect_model == "openai-api/benchmark/your-model-id"
    assert len(config.benchmarks) == 6
    assert sum(item.weight for item in config.benchmarks) == pytest.approx(1.0)


def test_rejects_bad_weight_sum(tmp_path: Path) -> None:
    content = (ROOT / "benchmark.example.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(content.replace("weight: 0.20", "weight: 0.21", 1), encoding="utf-8")
    with pytest.raises(ConfigError, match="权重之和"):
        load_config(path)
