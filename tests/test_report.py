import json
from pathlib import Path

import pytest

from llm_benchmark.report import collect_results, render_report, write_report


def _run_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    log_dir = run_dir / "logs" / "alpha"
    log_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "created_at": "2026-08-14T00:00:00+00:00",
        "suite": "test-v1",
        "model": {
            "model_id": "test-model",
            "base_url": "https://example.test/v1",
            "provider": "benchmark",
        },
        "run": {
            "temperature": 0.0,
            "seed": 42,
            "max_connections": 4,
            "max_retries": 3,
            "timeout": 120,
        },
        "benchmarks": [{
            "id": "alpha",
            "label": "Alpha",
            "dimension": "Reasoning",
            "task": "example/alpha",
            "weight": 1.0,
            "limit": 10,
            "task_args": {},
            "metric_names": ["accuracy"],
            "scorer_contains": None,
        }],
    }
    log = {
        "status": "success",
        "results": {
            "total_samples": 10,
            "completed_samples": 10,
            "scores": [{
                "name": "choice",
                "scorer": "choice",
                "metrics": {
                    "accuracy": {"value": 0.7},
                    "stderr": {"value": 0.1},
                },
            }],
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (log_dir / "alpha.json").write_text(json.dumps(log), encoding="utf-8")
    return run_dir


def test_collect_and_render(tmp_path: Path) -> None:
    run_dir = _run_fixture(tmp_path)
    manifest, results = collect_results(run_dir)
    markdown, payload = render_report(manifest, results)
    assert payload["overall"] == pytest.approx(0.7)
    assert "70.00%" in markdown
    assert "± 19.60pp" in markdown


def test_write_report(tmp_path: Path) -> None:
    run_dir = _run_fixture(tmp_path)
    markdown_path, json_path = write_report(run_dir)
    assert markdown_path.exists()
    assert json_path.exists()
    assert "test-model" in markdown_path.read_text(encoding="utf-8")


def test_extracts_nested_inspect_metric(tmp_path: Path) -> None:
    run_dir = _run_fixture(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmarks"][0]["metric_names"] = ["if_metric.prompt_strict_acc"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    log_path = run_dir / "logs" / "alpha" / "alpha.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))
    log["results"]["scores"][0]["metrics"] = {
        "if_metric": {
            "value": {
                "prompt_strict_acc": 0.8,
                "prompt_strict_stderr": 0.05,
            }
        }
    }
    log_path.write_text(json.dumps(log), encoding="utf-8")
    _, results = collect_results(run_dir)
    assert results[0].score == pytest.approx(0.8)
    assert results[0].stderr == pytest.approx(0.05)
