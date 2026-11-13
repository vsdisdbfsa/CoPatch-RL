# Stage 1 · C 交接

## 1. 基本信息
- stage / owner / branch: Stage 1 / C / stage1/c-eval
- base_commit: 55de863
- final_commit: 17f2450
- 是否下载模型/数据: 否（仅本地开发镜像 copatch-public-pytest:dev；无正式镜像、无权重、无原始数据）

## 2. 目标与实际完成范围
- 目标：完成 execution / data / evaluation 三块全部代码，用 Mock 与手写 fixture 验收，不做真实训练与评测。
- 完成：16 个模块文件、18 个测试文件、303 条用例；代码无空壳，未改 Contracts 与 A/B 目录。
- 未完成：见第 6 节。

## 3. 修改文件清单（仅 C 归属目录）
| 目录 | 文件 | 内部任务 | commit |
|---|---|---|---|
| execution/ | workspace.py, sandbox.py, public_tests.py, repository_tests.py | C-01~C-04 | 27f1ed0 |
| execution/ | grader.py | C-05 | 3ccc8e7 |
| execution/ | fixtures.py, examples/dev/count_positive/ | C-14 | 17f2450 |
| data/ | ingest.py | C-06 | c96e5e1 |
| data/ | dedupe.py | C-07 | 857ec07 |
| data/ | validate.py | C-08 | 22096e6 |
| data/ | trajectories.py | C-09 | 4805f45 |
| evaluation/ | conditions.py | C-10 | 787153e |
| evaluation/ | runner.py | C-11 | 3213b68 |
| evaluation/ | metrics.py | C-12 | 8b9c0de（+ bda9d31 格式） |
| evaluation/ | reports.py | C-13 | 170197e |

未修改：contracts.py、pyproject.toml、uv.lock、tests/conftest.py、tests/test_contracts.py、A/B 目录。

## 4. 新增或更改接口
- 新增 C 内部类型（均未进 Contracts，未修改 v1）：
  PublicResult、RepositorySuite、RepositoryDiagnostic、IngestResult、SplitPolicy、
  TaskRecord、Summary、Validation、Event、Report 六件产物格式。
- 无公共契约变更。需要 A/B 确认的接口请求见第 7 节。

## 5. 实际验证命令与退出码
| 命令 | 结果 |
|---|---|
| `.\.venv\Scripts\python.exe -m pytest -q` | 259 passed, 44 skipped（42 Docker opt-in + 2 Windows 符号链接） |
| `$env:COPATCH_DOCKER_TESTS='1'; 同上` | 259 passed, 2 skipped |
| `pytest --collect-only -q` | 303 条用例 |
| `ruff check src tests` | 退出码 0 |
| `ruff format --check src tests` | 34 files already formatted |

全部为 Mock / 手写 fixture 证据；无 Local、无 Hybrid、无真实模型调用。

## 6. 阻塞问题与已知风险
1. 仓库级独立评分不可信：pytest 与候选同进程，报告可被内省；本轮一律 diagnostic_only，不产出 SUCCESS。
2. 三类资产 Manifest 仅实现"任务"一类；SFT 与 GRPO 无真实轨迹，未做登记（字段格式已按 §7.4 设计）。
3. 报告分层按"每个 condition 一份"实现，未在单份报告内按 layer/mode 拆分。
4. 快照持久化与重启恢复未实现；当前仅内存内快照与提交重放。
5. 未与 B 的真实 Harness 联调（runner 只接注入的 Mock harness）。
6. 报告图表未实现；绘图依赖需 A 批准后方可加入。
7. 2 项 Windows 符号链接测试因本机权限跳过。

## 7. 接口请求（待 A/B 确认）
A：
1. `PublicResult` 是否提升为公共契约；当前为 C 内部 dataclass。
2. 仓库级 TaskSpec 扩展是否以关联 Manifest 承载（当前不改 public_view）。
3. 是否批准 matplotlib 作为报告绘图依赖。
B：
1. 未知 workspace / suite 建议返回 `PERMISSION_DENIED`（C 内部已按此实现）。
2. `Observation` 如何承载 `code_hash=None`，不得用当前 hash 顶替。
3. C-09 事件信封需与 B 的 Event Store 字段对齐（清单见 `data/trajectories.py` 的 `adapt_event`）。
4. `EvaluationRunner` 需要 B 提供 `run_episode(condition, task, seed) -> Outcome` 注入点。

## 8. 证据路径
- 本文件
- `tests/` 下 18 个测试文件（执行/数据/评测/集成）
- `examples/dev/count_positive/`：初态含边界缺陷，公开测试可复现失败

## 附：内部任务拆分说明
C-01 ~ C-14 是按方案 §18.3 的 C 完成清单拆出的执行单元，我个人做的任务拆分，不是方案的 Stage；
与模块、commit 的对应关系见第 3 节。每步的细节保留在各 commit message 中。