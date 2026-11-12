# 资产登记模板

V2.1 Bootstrap 提供资产登记模板。Stage 1 的三个代码 PR 合并、无模型验收通过后，由 A 选择并校验真实资产，再填写本表。数据来源沿用 V2.1 方案，实际版本和可用数量以预检登记为准。

| asset_id | role | source | license | pinned_revision | sha256 或 repo hash | size | logical_path | download_cmd | verified_at | verifier | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `main_qwen` | 正式 Main 底座 | Qwen 4B，仓库和变体待选 | 待核对 | 待锁定 | 待校验 | 待测 | `models/main_qwen` | 待登记 |  |  | 未验收 |
| `frozen_sub` | Local Planner / Coder / Reviewer | 型号和部署待试点，可共享权重 | 待核对 | 待锁定 | 待校验 | 待测 | `models/frozen_sub` | 待登记 |  |  | 未验收 |
| `mbpp` | 函数 Train / Dev | `google-research-datasets/mbpp` | 待核对 | 待锁定 | 待校验 | 待测 | `datasets/raw/mbpp` | 待登记 |  |  | 未验收 |
| `mbpp_plus` | 函数 Held-out 子集 | `evalplus/mbppplus` | 待核对 | 待锁定 | 待校验 | 待测 | `datasets/raw/mbppplus` | 待登记 |  |  | 未验收 |
| `swe_smith` | 仓库 Train / Dev / Held-out | `SWE-bench/SWE-smith-py` | 待核对 | 待锁定 | 待校验 | 待测 | `datasets/raw/swe_smith` | 待登记 |  |  | 未验收 |
| `swe_gym` | 仓库任务候选 | `SWE-Gym/SWE-Gym` | 待核对 | 待锁定 | 待校验 | 待测 | `datasets/raw/swe_gym` | 待登记 |  |  | 待审核 |
| `coding_preflight` | 受控函数和仓库小样 | 待选 | 待核对 | 待锁定 | 待校验 | 待测 | `datasets/raw/preflight` | 待登记 |  |  | 未验收 |

- 模型预检记录架构与加载类、tokenizer 和模板 hash、dtype、推理后端、LoRA 目标模块与重载结果。
- 数据预检记录来源版本、可用任务量、划分和许可。仓库小样另记上游提交与快照 hash、镜像 digest、平台、依赖锁与 runner 版本。跨来源仓库族须在 Train、Dev、Held-out 之间隔离。
- Hybrid 登记在线供应商及实际返回模型，凭据通过环境变量提供。

本机通过 `COPATCH_ASSETS_ROOT` 设置 Git 仓库外的资产根目录，逻辑路径见 [配置模板](../configs/assets.example.toml)。A 完成预检后，B/C 在各自环境按登记的 revision 和命令获取并核验资产。Git 只共享登记、脚本、Manifest、哈希和配置模板。
