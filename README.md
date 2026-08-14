# LLM Benchmark

面向 OpenAI-compatible Chat Completions API 的可复现 LLM 评分工具。v0.1 使用 Inspect AI / Inspect Evals 运行官方任务，本项目负责统一配置、逐项执行和生成 Markdown/JSON 报告。

## v0.1 包含什么

- `quick-v1`：MMLU-Pro、GPQA Diamond、GSM8K、IFEval、HumanEval、MMMLU-ZH
- API Key 直接写在本地 YAML 中，但不会进入运行快照、报告或 dry-run 命令
- 并发、超时和 API 重试次数显式固定，避免无限重试
- 启动 Inspect 子进程时强制 Python UTF-8 模式，兼容中文 Windows 的 GBK 默认编码
- 每项 benchmark 单独记录日志；单项失败不会删除其他结果
- `--dry-run` 检查最终 Inspect 命令，不消耗 token
- 综合分采用配置里的显式权重，并保留全部原始子项
- 同时输出 `report.md` 和 `results.json`

> Quick suite 使用 Inspect 的固定 seed、shuffle 和 `--limit`，适合低成本内部横向比较；它不是官方完整榜单结果。要发表或与论文数字比较，请去掉各项 `limit`、固定依赖版本，并记录完整配置。

## 安装

建议使用 Python 3.11 或 3.12；当前 Inspect Evals 的多数任务也可在 3.13 工作。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[inspect,dev]"
```

IFEval 检查器使用 Inspect Evals 锁定的 fork 与 Git revision 安装；这是因为该依赖不发布在 PyPI。首次运行任务时，Inspect Evals 还会下载相应公开数据集。

## 配置与运行

复制并修改 `benchmark.example.yaml`，填写 `model_id`、`base_url` 和 `api_key`：

```powershell
llm-benchmark doctor -c benchmark.example.yaml
llm-benchmark run -c benchmark.example.yaml --dry-run
llm-benchmark run -c benchmark.example.yaml
```

只跑便宜的 smoke test：

```powershell
llm-benchmark run -c benchmark.example.yaml --bench gpqa_diamond
```

运行结束后，文件位于：

```text
runs/<timestamp>-<model-id>/
├── manifest.json
├── run-status.json
├── results.json
├── report.md
└── logs/
```

重新汇总已有日志：

```powershell
llm-benchmark report runs/<timestamp>-<model-id>
```

逐题检查可使用 Inspect 自带 viewer：

```powershell
inspect view --log-dir runs/<timestamp>-<model-id>/logs
```

## 公平比较约束

只有 suite、task 版本、prompt、few-shot、temperature、输出预算和抽样完全一致的运行才应直接排序。API 失败或缺失指标不会被算作 0 分；报告会把综合分标记为“仅按已完成项目重新归一化”，并展示 coverage，避免半套结果伪装成完整得分。

HumanEval 会执行模型生成的代码。正式运行前请确认 Inspect 的 sandbox 配置符合你的安全边界，不要在包含敏感文件或凭据的无隔离环境中直接执行未知代码。

## v0.2 方向

- 为 quick suite 生成并锁定分层抽样 manifest，而不只依赖 `--limit`
- C-Eval/OpenCompass 后端
- token、费用和 P50/P95 延迟独立报告
- 多模型 leaderboard 与 bootstrap 置信区间
