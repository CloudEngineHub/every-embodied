# ATEC2026 轻量代码

这里保存可以直接放入教程仓库的轻量代码，不包含 Isaac Sim、模型权重、训练缓存或第三方完整仓库。

## 文件

- `taskd_reference_adapter.py`：演示 Logic-TARS 风格的 16D locomotion action 到 24D official action 的适配，以及基于 score 的 Task D 阶段划分。
- `external_repo_audit/clone_references.sh`：按固定 commit 获取公开外部参考仓库。
- `external_repo_audit/external_repos.yaml`：仓库、commit、许可证和迁移用途清单。

## 边界

这些文件用于理解接口和复现资料组织，不是官方提交包，也不包含能够保证高分的 policy。
