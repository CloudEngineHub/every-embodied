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

- 官方仿真资产和机器人模型。
- 未获得再分发许可的外部数据集。
- 私有 checkpoint、私有日志和比赛原始视频。
- 访问令牌、机器绝对路径和内部服务地址。

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
