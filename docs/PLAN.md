# CoPatch-RL 阶段计划

阶段以验收证据放行。完整任务、负责人和风险见 [V2 第 18–19 章](copatch_rl_v2.md#ch18)。

| 阶段 | 工作范围 | 放行条件 |
| --- | --- | --- |
| Bootstrap | A 初始化仓库、依赖锁和质量检查，定义公共契约、共享 Mock、资产登记模板和文件所有权。 | 干净环境 `uv sync --locked` 后可导入，Ruff 和 pytest 通过，Mock 样例验证合法动作、观察与错误分支。A 核对候选提交并发布 `stage1-base`，向 B/C 共享同一提交哈希。 |
| Stage 1 | A 实现训练与资产工具，B 实现 Harness、模型适配和应用，C 实现执行、数据和评测。各工作分支使用 FakeClient、合成输入和小型 fixture。 | 三份 PR 通过审查、全量质量检查、Mock 集成和实际 Docker 隔离测试。合并后由 A 下载固定版本模型和受控数据小样、登记资产、完成 Local 与评分器的真实接口预检，并在具备有效凭据时发出真实 Hybrid 请求，再发布 `stage1-accepted`。 |
| Stage 2 | 基于已接受版本构造真实训练轨迹、执行 SFT → GRPO 和 Dev 对照，核对评分和成本。 | 保存真实 Main LoRA 更新、梯度、mask、旧 logprob、重载与运行成本证据。冻结模型、Harness、数据 Manifest、预算和 `eval_protocol.json`，发布 `stage2-accepted`。 |
| Stage 3 | 按冻结协议跑 Held-out 对照与必要消融，完成真实 Hybrid 演示、复现和技术报告。 | 留存逐题结果、失败、成本和版本哈希。负责人在干净环境核验关键命令与产物，审核合并并发布最终版本。 |

Bootstrap 和 Stage 1 工作分支禁止下载正式模型或数据。Stage 1 合并后的预检验证资产加载、接口兼容性和最小训练路径。缺 GPU 时先验证加载与训练兼容原型，正式 SFT/GRPO 更新与调参在 Stage 2 执行。Held-out 只按冻结协议评分。

审查顺序建议为 C（执行和评分）→ B（Harness 和模型适配）→ A（训练和集成）。各阶段交接要求见 [OWNERSHIP](OWNERSHIP.md)。
