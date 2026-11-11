# Bootstrap V2.1 · A 交接

| 字段 | 记录 |
| --- | --- |
| Stage / Owner | Bootstrap V2.1 / A |
| Branch | `bootstrap/a-init` |
| Base commit | `b58e11f7be553e7bf64294f37c1de4afc9bd05a5`（原 `stage1-base`） |
| Final commit | `git log -1 --format=%H -- docs/handoffs/bootstrap_v2_1_a.md` |
| Summary | 同步 Qwen 4B 目标、Local 冻结角色配置、函数与仓库任务的数据来源和阶段预检。公共契约采用 v1。 |
| Changed files | `README.md`、`AGENTS.md`、`configs/assets.example.toml`、`docs/{copatch_rl_v2,SPEC,PLAN,OWNERSHIP,ASSET_REGISTRY}.md`、`docs/ADR/0001-architecture.md` 和本交接。 |
| Tests passed | Python 3.11.16 全新环境锁定安装成功，导入得到契约版本 `1`。锁文件、Ruff 检查及格式通过，pytest 10 项通过。另核对 TOML 与登记表的资产 ID、路径和角色绑定，并验证手写仓库任务的 v1 公开视图。 |
| Evidence paths | [复验命令](../../README.md#开发环境)、[契约测试](../../tests/test_contracts.py)、[资产模板](../../configs/assets.example.toml)、[登记表](../ASSET_REGISTRY.md)。 |
| Risks | 模型仓库、变体、revision 与校验和待预检。SWE-Gym 待审核，规划任务量待试点后确认。 |

本次验收使用手写任务和 Mock。模型权重、正式数据及任务镜像下载量为零。

## Stage 1 工作区

三个分支从本交接文件所在提交创建，供 A 审查 V2.1 候选版本。A 完成审查后统一发布正式基线。

| 负责人 | 分支 |
| --- | --- |
| A | `stage1/a-training` |
| B | `stage1/b-harness` |
| C | `stage1/c-eval` |
