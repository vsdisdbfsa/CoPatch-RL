# 资产登记模板

Bootstrap 仅提供模板。Stage 1 的三个代码 PR 合并、无模型验收通过后，由 A 选择并校验真实资产，再填写本表。

| asset_id | role | source | license | pinned_revision | sha256 或 repo hash | size | logical_path | download_cmd | verified_at | verifier | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `main_qwen` | Main 底座 | 待定 | 待核对 | 待锁定 | 待校验 | 待测 | `models/main_qwen` | 待登记 |  |  | 未验收 |
| `frozen_sub` | Planner / Coder / Reviewer | 待定 | 待核对 | 待锁定 | 待校验 | 待测 | `models/frozen_sub` | 待登记 |  |  | 未验收 |
| `coding_preflight` | 预检数据小样 | 待定 | 待核对 | 待锁定 | 待校验 | 待测 | `datasets/raw/preflight` | 待登记 |  |  | 未验收 |

本机通过 `COPATCH_ASSETS_ROOT` 设置 Git 仓库外的资产根目录，逻辑路径见 [配置模板](../configs/assets.example.toml)。A 完成预检后，B/C 在各自环境按登记的 revision 和命令获取并核验资产。Git 只共享登记、脚本、Manifest、哈希和配置模板。
