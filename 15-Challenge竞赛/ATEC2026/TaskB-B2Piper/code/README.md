# Task B 轻量代码

这里的脚本用于公开发布前审计和观测接口检查，不依赖 Isaac Sim、Isaac Lab、模型权重或官方机器人资产。

## 文件

- `inspect_task_b_observation.py`：读取脱敏 JSON，打印观测树、键名、类型和列表形状。
- `build_public_manifest.py`：为小于 100 MiB 的文件计算 SHA-256，并生成发布清单。
- `scan_public_release.sh`：检查凭据、绝对路径、超大文件和缓存目录。
- `fetch_reference_repos.sh`：把公开参赛参考仓库拉到仓库外的缓存目录，避免误提交完整第三方代码。

## 使用边界

这些脚本不执行 Task B 控制、不读取仿真真值、不生成目标坐标，也不宣称完成抓取或垃圾桶放置。正式策略仍需接入官方 runner，并按官方版本重新核对 observation key、action space、动作缩放和评测器。

## 最小 smoke test

```bash
python3 -m py_compile inspect_task_b_observation.py build_public_manifest.py
bash -n fetch_reference_repos.sh
bash -n scan_public_release.sh
```
