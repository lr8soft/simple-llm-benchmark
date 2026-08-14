from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .report import ReportError, write_report
from .runner import RunnerError, inspect_executable, run_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-benchmark", description="OpenAI-compatible LLM benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="运行一个 benchmark suite")
    run.add_argument("-c", "--config", default="benchmark.example.yaml")
    run.add_argument("--bench", action="append", help="只运行指定 benchmark id；可重复")
    run.add_argument("--dry-run", action="store_true", help="仅打印 Inspect 命令，不请求 API")

    report = subparsers.add_parser("report", help="从已有 Inspect 日志生成报告")
    report.add_argument("run_dir")

    doctor = subparsers.add_parser("doctor", help="检查配置、密钥和依赖")
    doctor.add_argument("-c", "--config", default="benchmark.example.yaml")

    return parser


def _doctor(config_path: str) -> int:
    config = load_config(config_path)
    checks = [
        ("配置", True, str(config.source_path)),
        ("Python", sys.version_info[:2] >= (3, 11) and sys.version_info[:2] < (3, 14), sys.version.split()[0]),
        ("Inspect CLI", inspect_executable() is not None, inspect_executable() or "未安装"),
        ("inspect_evals", importlib.util.find_spec("inspect_evals") is not None, "Python package"),
        ("IFEval checker", importlib.util.find_spec("instruction_following_eval") is not None, "Python package"),
        ("API Key", bool(config.model.api_key), "已配置" if config.model.api_key else "未配置"),
    ]
    checks.append(("TLS verify", True, "开启" if config.model.tls_verify else "关闭（不安全）"))
    if any(item.id == "humaneval" for item in config.benchmarks):
        checks.append(("Docker (HumanEval)", shutil.which("docker") is not None, shutil.which("docker") or "未安装"))
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args.config)
        if args.command == "report":
            markdown, data = write_report(Path(args.run_dir))
            print(f"Markdown: {markdown}")
            print(f"JSON: {data}")
            return 0
        if args.command == "run":
            config = load_config(args.config)
            run_dir = run_suite(config, only=args.bench, dry_run=args.dry_run)
            if run_dir is not None:
                markdown, data = write_report(run_dir)
                print(f"\nRun: {run_dir}")
                print(f"Markdown: {markdown}")
                print(f"JSON: {data}")
            return 0
    except (ConfigError, RunnerError, ReportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2
