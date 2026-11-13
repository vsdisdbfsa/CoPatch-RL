# Stage 1 · C 交接

- stage / owner / branch: Stage 1 / C / stage1/c-eval
- base_commit: 55de863 · final_commit: 22096e6
- 是否下载模型/数据：否（仅用已有开发镜像 copatch-public-pytest:dev）
- 证据命令：见文末"复现命令"

## C-01 ~ C-03 执行底座
files: execution/{workspace,sandbox,repository_tests}.py + 4 个 test_execution_* 文件
tests: 61 collected（含 11 容器）
unverified: 2 项 Windows 符号链接权限

## C-04 公开测试
files: execution/public_tests.py, examples/dev/public_repository/
tests: 全量 98/31 (default) · 127/2 (DOCKER=1) · 129 collected
unverified: 仓库报告可信边界（pytest 同进程）→ 只能 diagnostic_only
pending A/B: PublicResult 公共化、Observation 映射、未知 suite → PERMISSION_DENIED

## C-05 独立评分
files: execution/grader.py, tests/test_execution_grader.py, tests/fixtures/private/
tests: 19 = 12 逻辑 + 7 容器 · 全量 146/2
unverified: 仓库可信成功（本轮不产出 SUCCESS）
backlog: 快照持久化、跨进程评分边界

## C-06 三源导入
files: data/ingest.py, tests/test_data_ingest.py
tests: 20 passed
key: 公开清单六字段白名单；SWE-smith patch=注错；SWE-Gym 固定 pending

## C-07 去重与划分
files: data/dedupe.py, tests/test_data_dedupe.py
tests: 12 passed
key: 默认 SplitPolicy(7/1/2)，与实际比例待 A 冻结

## C-08 题目校验
files: data/validate.py, tests/test_data_validate.py
tests: 19 = 14 逻辑 + 5 容器 · 全量 197/2
unverified: 真实许可/依赖、语义题意、真实仓库恢复

## 复现命令
.\.venv\Scripts\python.exe -m pytest -q                    → 156 passed, 43 skipped
$env:COPATCH_DOCKER_TESTS='1'; 同上                        → 197 passed, 2 skipped
uv run ruff check src tests / format --check               → 退出码 0