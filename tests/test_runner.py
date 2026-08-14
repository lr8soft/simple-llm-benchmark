from pathlib import Path

from llm_benchmark.config import load_config
from llm_benchmark.runner import build_command, inspect_executable


ROOT = Path(__file__).parents[1]


def test_command_does_not_contain_api_key() -> None:
    config = load_config(ROOT / "benchmark.example.yaml")
    benchmark = config.benchmarks[0]
    command = build_command("inspect", config, benchmark, Path("logs"))
    rendered = " ".join(command)
    assert "LLM_BENCHMARK_API_KEY" not in rendered
    assert "--model openai-api/benchmark/your-model-id" in rendered
    assert "-T fewshot=0" in rendered
    assert "--limit 500" in rendered
    assert "--max-retries 3" in rendered
    assert "--timeout 120" in rendered


def test_finds_inspect_next_to_venv_python_when_installed() -> None:
    # This test also covers PyCharm/direct-entry-point invocation where the
    # venv's Scripts directory is not present in PATH.
    found = inspect_executable()
    if found is not None:
        assert Path(found).name.lower() in {"inspect", "inspect.exe"}
