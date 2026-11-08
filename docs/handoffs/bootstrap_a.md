# Bootstrap · A 交接

| 字段 | 记录 |
| --- | --- |
| Stage / Owner | Bootstrap / A |
| Branch | `bootstrap/a-init` |
| Base commit | 初始仓库，无前置提交 |
| Final commit | `git log -1 --format=%H -- docs/handoffs/bootstrap_a.md` |
| Summary | 初始化 Python 3.11 包与锁定环境，提供公共契约 v1、共享 FakeClient 和合法及非法样例，明确 Stage 1 所有权及接口边界。 |
| Changed files | `README.md`、`AGENTS.md`、`.gitignore`、`pyproject.toml`、`uv.lock`、`docs/{SPEC,PLAN,OWNERSHIP,ASSET_REGISTRY,copatch_rl_v2}.md`、`docs/ADR/0001-architecture.md`、本交接、`configs/assets.example.toml`、`src/copatch_rl/`、`tests/`。 |
| Evidence paths | [复现命令](../../README.md#开发环境)、[契约测试](../../tests/test_contracts.py)、[Mock 样例](../../tests/fixtures/bootstrap_cases.json)。 |
| Risks | Bootstrap 证据来自 macOS ARM64、Python 3.11.16 与 Mock。运行时、Docker 隔离、真实模型和训练的证据在对应阶段产出。 |

## 实际验收

| 命令 | 退出码与结果 |
| --- | --- |
| `uv sync --locked --python 3.11 --no-python-downloads` | `0`，全新虚拟环境安装成功 |
| `uv run --locked python -c 'from copatch_rl.contracts import CONTRACT_VERSION; print(CONTRACT_VERSION)'` | `0`，输出 `1` |
| `uv run --locked ruff check src tests` | `0`，全通过 |
| `uv run --locked ruff format --check src tests` | `0`，4 个文件符合格式 |
| `uv run --locked pytest -q` | `0`，10 个测试通过 |
| `uv lock --check` | `0`，锁文件与项目配置一致 |

安装和验收使用临时 `UV_PROJECT_ENVIRONMENT` 与 `UV_CACHE_DIR`。模型权重和正式数据下载为零。`docs/copatch_rl_v2.md` 沿用原文。

待负责人审查候选提交后发布 `stage1-base`，向 B/C 共享同一基线哈希。
