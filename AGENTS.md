# CoPatch-RL · AGENTS.md

> 本规范适用于该仓库内所有 Coding Agent 与团队成员。先阅读 [V2.1 实施方案](docs/copatch_rl_v2.md) 的一页总览和第 16–19 章，并核对 [阶段门槛](docs/PLAN.md)、[文件所有权](docs/OWNERSHIP.md) 与当前基线的 `src/copatch_rl/contracts.py`。模型和数据以实施方案第 5–7 章为准。

## 项目范围

- **1 Main + 3 Frozen Sub-Agent**（Planner / Coder / Reviewer），正式 Main 目标为 Qwen 4B，仅训练 Main LoRA。
- Harness 同时支持 Local 冻结模型与 Hybrid 在线 LLM；训练与评测分开，所有结果必须真实、可追溯。
- Bootstrap → Stage 1（全部代码实现）→ 资产下载与接口 Gate → Stage 2（SFT/GRPO + Dev）→ Stage 3（Held-out 与交付）。
- 非当前 Stage 的新需求不可擅自实现。Bootstrap 和 Stage 1 工作分支不得下载模型权重、正式数据或正式任务镜像。Stage 1 代码合并后的资产预检由 A 负责。

## 文件所有权

- **A：** `src/copatch_rl/contracts.py`、`src/copatch_rl/training/`、根目录 README/AGENTS/pyproject/uv.lock、全局配置与资产脚本。
- **B：** `src/copatch_rl/{core,harness,agents,models,app}/` 和 B 专属测试/演示文档。
- **C：** `src/copatch_rl/{execution,data,evaluation}/`、`examples/dev/` 和 C 专属测试。
- 跨模块公共接口改变先提交变更请求，由 A 修改后广播；其他人禁止擅自修改公共契约和依赖锁。
- 每人独立 Git worktree 和 Stage 分支；禁止多 Agent 直接在同一个目录同时写入或向 main 直接提交。

## Python 习惯与代码质量

- Python 3.11；4 空格；双引号；Ruff 行宽 100；`snake_case` 函数变量、`PascalCase` 类。
- 可读、直接、**最小必要抽象**：有状态才用类，无状态逻辑用函数；少做多层包装、无意义继承和多余文件。
- 公共接口加类型标注；内部数据优先 `dataclass`，跨模块 Schema 用 Pydantic；使用 `pathlib.Path` 和明确异常类型。
- 标准库 → 第三方 → 包内导入；不使用通配符导入，不修改 `sys.path` 规避工程结构。
- 通过参数配置、环境变量传入路径/Key；不硬编码本机目录、密钥或模型 revision。
- 错误必须显式处理：拒绝裸 `except:`、静默 fallback、伪造成功结果；日志脱敏。
- 算法类逻辑要检查 token/mask/logprob、shape/dtype、grad 与 Checkpoint，Sub-Agent 不能接收梯度。
- 修改前先列 `files_to_change` 与 `tests_to_run`，只做与任务直接相关的最小改动，不附带全局重构。

## 测试、证据和安全

- 每次修改后运行相关 `pytest`；提交前通过 `uv run ruff check src tests`、`uv run ruff format --check src tests`、`uv run pytest -q`。
- Mock 与真实模型调用分开记录；只有实际运行过的代码、测试和训练结果才可以称为“完成”。
- 候选代码只在 Docker 隔离区执行；禁止读取隐藏评分数据，禁止向模型提供训练 Reward 的私有细节。
- JSONL 保存事件事实；SQLite 保存可重建索引与任务元数据；重放和恢复必须有测试。
- 模型权重、完整缓存、私测、`.env`、SQLite 运行文件、私有轨迹不进入 Git；只共享下载命令、Manifest、修订号、hash 与相对路径。
- **禁止** 为通过测试而改 Benchmark 真值、剔除失败任务、在测试集上调超参，或用伪造日志替代真实 RL Rollout。

## PR 与阶段交接

交付时输出：`stage / owner / branch / base_commit / final_commit / summary / changed_files / tests_passed / evidence_paths / risks`，并提供独立交接文件 `docs/handoffs/stageX_{a|b|c}.md`。

所有变更先经过负责人代码审查、模块所有权与安全检查，再合并至 main。三个 PR 全部通过、统一集成测试通过后才能发布 Stage Gate。Stage 1 合并后由 A 统一下载权重与数据、验证接口并发布 `docs/ASSET_REGISTRY.md`，其余成员按登记自行下载。
