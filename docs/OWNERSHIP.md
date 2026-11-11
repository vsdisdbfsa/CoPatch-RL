# 文件所有权与交接

| 负责人 | 文件和目录 | Stage 1 责任 |
| --- | --- | --- |
| A | `src/copatch_rl/__init__.py`、`src/copatch_rl/contracts.py`、`src/copatch_rl/training/`、`scripts/assets/`、`configs/`、根目录工程配置与依赖锁、Bootstrap 公共文档、`tests/test_contracts.py`、`tests/test_training_*`、`tests/conftest.py`、`tests/fixtures/bootstrap_cases.json` | 公共契约及仓库任务扩展审核、Qwen 4B Main LoRA、GRPO 分组、资产工具和集成审查 |
| B | `src/copatch_rl/{core,harness,agents,models,app}/`、`tests/test_harness_*`、`tests/test_models_*`、应用专属测试、`docs/demo.md` | Main/Sub 运行与角色状态隔离、函数及仓库任务公开视图、Local/Hybrid 适配、JSONL/SQLite、CLI/Web |
| C | `src/copatch_rl/{execution,data,evaluation}/`、`tests/test_execution_*`、`tests/test_data_*`、`tests/test_eval_*`、`examples/dev/` | Docker 隔离、函数及仓库执行、快照复原、独立评分、来源去重与分层评测 |

## 一、分支与公共接口

- Bootstrap 由 A 独立完成。进入 Stage 1 后，三人从相同的已接受基线在独立 worktree 和独立分支工作，各自只改归属文件和测试。下一阶段从 A 发布的已接受提交出发。
- 跨模块接口变更由提议人提交原因、影响和兼容测试。A 维护公共契约和锁文件，通过 PR 审查合并后向 B/C 广播新基线。
- 阶段工作通过 PR 进入 `main`。A 核查文件所有权、测试和安全证据，完成集成复测后才发布阶段 Gate。

## 二、交接证据

每人提交 `docs/handoffs/stageX_{a|b|c}.md`。记录 `stage / owner / branch / base_commit / final_commit / summary / changed_files / tests_passed / evidence_paths / risks`，附实际命令与退出码，区分 Mock、Local 和 Hybrid。注明接口改动、阻塞项、复现步骤和资产下载情况。私测、密钥、权重、运行数据库和私有轨迹独立保管。

通用规范见 [AGENTS](../AGENTS.md)，阶段验收见 [PLAN](PLAN.md)。
