# ATEC2026 线上赛赛后开源复盘

本目录整理 DatawhaleEAI / Every Embodied 在 ATEC2026 线上赛中的两条 L0 任务实践记录。它不是“满分秘笈”，而是一次真实参赛工程的赛后开源：保留能复现的代码骨架、提交包结构、调试日志摘要、失败路线和最终判断，方便后来者学习 Isaac Lab 竞赛、机器人策略训练和提交工程。

## 目录结构

```text
ATEC2026/
  README.md
  L0-机器人徒步/
    README.md
    assets/
    code/l0_b2piper_option_moe/
  L0-桌面整理TaskE/
    README.md
    code/act_seed1_best_submission/
    docs/
    hf_model_package/
```

## 已整理任务

| 子目录 | 对应任务 | 机器人 | 最终公开内容 | 备注 |
|---|---|---|---|---|
| `L0-机器人徒步/` | 赛道1 L0 机器人徒步 | `B2Piper` | 复现教程、曲线图、option-MoE 代码骨架、HF 链接 | 线上有效分从 baseline 附近起步，教程重点是 RL 评估闭环 |
| `L0-桌面整理TaskE/` | 赛道2 L0 桌面整理 Task E | `Piper` | ACT 提交代码骨架、技术复盘、Workspace Memory、HF 上传说明 | 线上最好回报为 `15.00`，不是满分方案 |

## 公开结论

- L0 机器人徒步：官方 baseline 是最重要的起点，能帮助跑通提交链路；继续冲榜必须让训练 reward 对齐 TaskA 长距离路线进度，而不是盲目长训。
- L0 桌面整理 Task E：最终保护方案是 ACT seed1 best，线上用户回报 `15.00`。我们尝试过 XSA-ACT、ACT seed2、PCA/GraspGen-style、AnyGrasp/GraspNet、SAM3、pi0.5/OpenPI 等路线；赛后复盘显示短期最可靠的是 ACT 提交包，而不是尚未稳定的规划抓取或 VLA 分支。
- 公开检索：截至 2026-07-18，未发现明确开源的 ATEC2026 Task E 满分/18 分完整提交包。能检索到的主要是官方仓库、赛事介绍和通用机器人项目，不能当成可直接提交的高分方案。

## 官方与外部链接

- ATEC 官网：[https://www.atecup.com/](https://www.atecup.com/)
- ATEC2026 页面：[https://www.atecup.com/competitions/ATEC2026](https://www.atecup.com/competitions/ATEC2026)
- 官方仿真挑战仓库：[https://github.com/atecup/ATEC2026_Simulation_Challenge](https://github.com/atecup/ATEC2026_Simulation_Challenge)
- Every Embodied 仓库：[https://github.com/datawhalechina/every-embodied](https://github.com/datawhalechina/every-embodied)

## 模型权重

GitHub 目录默认不直接存放大权重。权重建议放到 Hugging Face，并在对应子目录 README 中维护链接和 SHA256。

- `L0-机器人徒步/` 已记录 Hugging Face 模型仓库链接。
- `L0-桌面整理TaskE/` 已记录 Hugging Face 模型仓库和复现数据集仓库；数据集仓库包含 filtered HDF5 的 100 MiB 分片、日志归档、恢复脚本和 SHA256。

## 本地清理建议

在确认 GitHub push 成功、Hugging Face 权重上传成功、并且 `WORKSPACE_MEMORY` 已备份后，可以删除本地训练中间产物、日志视频和大型 checkpoint。不要删除以下轻量归档：

- 本目录；
- 对应 Hugging Face 模型仓库；
- `L0-桌面整理TaskE/docs/WORKSPACE_MEMORY_TaskE_完整调试记录.md`；
- SHA256 清单。
