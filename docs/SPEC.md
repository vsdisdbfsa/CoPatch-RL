# CoPatch-RL 项目规格

## 一、目标与范围

在 Python 函数题和仓库级修复任务中验证 **动态编排是否因强化学习而改善**。Main 根据公开任务、候选代码、工具反馈和预算选择委派、测试、修复或提交。Planner、Coder、Reviewer 的调用由 Main 决定，固定流程用于评测对照。

系统交付 Python 应用、Main LoRA 的 SFT → GRPO 训练、独立评测和复现材料。正式 Main 目标为 Qwen 4B，具体仓库、变体与 revision 待选。Local 使用本地冻结 Sub-Agent 承担主训练和评测。三个角色可以共享 Sub 权重实例，各自使用独立 Prompt、Context、Session 和 KV 缓存。Hybrid 使用同一个 Main 和在线 Sub-Agent，承担应用演示及独立迁移评测。通用桌面操作、联合训练 Sub-Agent、长期记忆和分布式 RL 超出本期范围。

函数层使用 MBPP 的 Train/Dev 划分与 MBPP+ 的去重保留子集。仓库层使用 SWE-smith 按仓库族划分的任务，SWE-Gym 通过来源、许可和环境审核后才接入。12 道自建小工程题单列为探索结果。

## 二、运行与评分边界

1. B 的 Harness 管理 Agent Loop、角色上下文、工具权限、预算和终止状态。JSONL 记录原始事件，SQLite 保存元数据和可重建索引。
2. C 的执行器只在 Docker 中运行候选代码，将公开测试反馈与当前源码 hash 绑定。提交冻结候选快照，独立 Grader 再用私测评分。
3. 模型输入使用 `TaskSpec.public_view()`。私测索引、脚本、预期值和失败详情留在宿主评分侧。训练奖励由 Grader 交给 Trainer。
4. A 的 Trainer 冻结 Qwen 底座和三个 Sub-Agent，仅更新 Main LoRA。SFT 仅监督 Main 生成 token。正式 GRPO 每组从同题同初态独立采样 4 条轨迹，按实际生成段对齐前缀、token、旧 logprob 和 mask，计算组相对优势。

## 三、Stage 1 接口交接

Bootstrap 提供可导入的 [公共契约 v1](../src/copatch_rl/contracts.py)、[共享 Mock](../tests/conftest.py) 和 [合法及非法样例](../tests/fixtures/bootstrap_cases.json)。`ModelClient.generate(role, messages, tools, config, trace_ctx)` 接收公开消息和允许的 `ToolName`。适配器可用 `Action.argument_schema(tool_name)` 获取入参结构，并从宿主 `TraceContext` 填入 `main_step`。角色权限、文件范围、预算和执行安全由 Stage 1 的工具分派和执行器检查。

运行模块在 Stage 1 实现。以下签名是本阶段交接边界。

仓库任务沿用 v1 的公开视图与工具接口。恢复与独立评分需要的仓库来源、提交、镜像和私有用例索引可由宿主 Manifest 关联。若需扩展公开字段，由 A 审核契约和兼容测试后发布。

| 所有者 | Stage 1 接口 | 交接边界 |
| --- | --- | --- |
| B | `run_episode(task: TaskSpec, config: RunConfig) -> EpisodeRecord`，以及模型适配、会话、上下文和工具分派 | 把模型响应归一为 `ModelResponse`，校验 `Action` 并形成 `Observation`。记录 `MainGeneration` 的真实采样字段。没有采样数据时保留 `None`。 |
| C | `Executor.run_public_tests(workspace_id, suite_id)`、`Grader.score(submitted_snapshot_id, task_id) -> GraderResult`、`EvaluationRunner.run(condition_id, task_manifest)` | 提供函数和仓库任务的隔离执行、公开反馈、终局私测与分层对照。区分候选失败和基础设施故障。 |
| A | `TrainingAdapter.sample_group(task_id, policy_version, group_size) -> list[Trajectory]`，以及 SFT/GRPO、Reward 和 Checkpoint | 复用 B 的运行入口和 C 的终局评分，验证同题采样、Main-only mask、旧 logprob、参数冻结和恢复。 |

技术细节与最终指标见 [V2.1 实施方案](copatch_rl_v2.1.md)，阶段门槛见 [PLAN](PLAN.md)。
