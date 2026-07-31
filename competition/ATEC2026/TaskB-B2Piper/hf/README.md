---
language:
- zh
license: other
pretty_name: ATEC2026 Task B reproducibility archive
tags:
- embodied-ai
- robotics
- atec2026
- b2-piper
---

# Hugging Face 公开归档

对应的 Datawhale 公开数据集仓库：

- [Datawhale/atec2026-task-b-reproducibility](https://huggingface.co/datasets/Datawhale/atec2026-task-b-reproducibility)

## 内容边界

HF 归档与 GitHub 教程保持一致，保存轻量文档、配置、检查脚本、结果边界和发布清单。它不包含：

- 未获得再分发许可的外部数据集。
- 私有 checkpoint、私有日志和比赛原始视频。
- 访问令牌、机器绝对路径和内部服务地址。

本次已额外上传两类可公开复现材料：

- `source/ATEC2026_Simulation_Challenge_20260518.zip`：官方公开仓库的 2026-05-18 源码、机器人资产和 baseline 快照，保留官方 MIT LICENSE。
- `TaskB-B2Piper/videos/` 与 `TaskB-B2Piper/frames/`：本地 Task B 调试/评估预览，均标注为 debug-only，不代表官方分数或完整抓取成功。

源快照 SHA-256：`84ccfdc3903e4e03a5de8a7dedd90314b15ed09382c136bbeee6a858dae802d9`。

大文件如果未来获得授权，应单独写明来源、许可证、SHA-256、版本和下载方式，不直接把它们塞进 GitHub 教程。

## 从 GitHub 复现

```bash
git clone https://github.com/datawhalechina/every-embodied.git
cd every-embodied/competition/ATEC2026/TaskB-B2Piper
bash code/scan_public_release.sh .
python3 code/build_public_manifest.py . --output public_manifest.json
```

## 从 HF 获取

```bash
git clone https://huggingface.co/datasets/Datawhale/atec2026-task-b-reproducibility
```

HF 仓库中的教程文件不是官方 Task B 环境的替代品。请先按官方仓库安装环境，再使用本目录做边界审计和实验记录。

## 复现能力边界

上传后，公开归档可以复现教程中的观测审计、发布扫描、官方源码快照定位和视频证据阅读；完整 Task B 仿真仍需要本地安装与官方版本匹配的 Isaac Sim/Isaac Lab、GPU 驱动和 Python 环境。HF 归档不包含私有环境缓存，也不声称只靠 `git clone` 就能无条件启动仿真。

推荐固定官方源码版本：`4000378a9a6fc6ce3e57bcdd20a1582f6854e0dc`。若使用本 HF 快照，请以压缩包内的 `readme.md`、`LICENSE` 和任务脚本为准。
