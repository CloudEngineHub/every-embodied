# AgiBot World 2026 竞赛资源与复现入口

本页汇总竞赛报名、世界模型赛题、数据集和基线代码入口。开始实验前，先阅读竞赛主页中的任务定义与提交规则，再根据所选赛题下载对应资源。

## 报名与任务说明

报名官网：

https://agibot-world.com/challenge2026

https://github.com/AgibotTech/ACoT-VLA?tab=readme-ov-file

https://arxiv.org/abs/2601.11404





## 世界模型赛题

https://huggingface.co/spaces/agibot-world/ICRA26WM

40G数据：https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2026/tree/main/WorldModel

安装配置说明：https://github.com/AgibotTech/AgiBotWorldChallengeICRA2026-WorldModelBaseline?tab=readme-ov-file

## 最小验证

克隆基线仓库后，按其环境文件创建独立 Python 环境，并先运行仓库提供的帮助命令或最小推理示例。能够正确加载配置、列出任务并读取一条样本，说明代码与数据路径已经连通；正式训练前再检查显存、磁盘和输出目录。

## 常见问题

- 数据下载中断：使用下载工具的续传选项，不要删除已经完成的分片。
- 找不到数据：核对配置中的数据根目录是否指向下载后的实际目录。
- 模型无法加载：确认基线代码版本与权重版本一致，并查看项目发布页的依赖约束。
