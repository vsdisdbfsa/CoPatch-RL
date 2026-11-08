# CoPatch-RL

CoPatch-RL 研究如何让 Main Agent 根据公开代码执行反馈，动态调用 Planner、Coder 和 Reviewer。三种 Sub-Agent 与 Qwen 底座冻结，训练阶段仅更新 Main 的 LoRA。Local 用于训练和主要对照，Hybrid 用于真实在线应用和独立迁移验证。

当前交付为 **Bootstrap**，包含共享契约、合法及非法 Mock 样例和锁定开发环境。Stage 1 负责实现运行时、执行、评测和训练入口。

## 开发环境

使用已安装的 Python 3.11 和 uv。从仓库根目录运行以下命令即可复现 Bootstrap 验收。

```bash
uv sync --locked --python 3.11 --no-python-downloads
uv run --locked python -c 'from copatch_rl.contracts import CONTRACT_VERSION; print(CONTRACT_VERSION)'
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
```

测试使用手写 fixture 和 FakeClient，无需模型、数据集、API 密钥、Docker 或 GPU。

## 开发入口

- [V2 实施方案](docs/copatch_rl_v2.md)（优先阅读一页总览和第 16–19 章）
- [功能范围与 Stage 1 接口](docs/SPEC.md)
- [阶段门槛](docs/PLAN.md)与[模块所有权](docs/OWNERSHIP.md)
- [架构决策](docs/ADR/0001-architecture.md)与[资产登记模板](docs/ASSET_REGISTRY.md)
- [仓库协作规范](AGENTS.md)
