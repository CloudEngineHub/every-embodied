# ATEC2026 线上赛：赛后复盘、公开方案与复现入口

本目录是 Every Embodied 第 15 章中的 ATEC2026 统一入口，集中收录任务定义、L0 任务实践、Task B 垃圾收集复盘、Task D 公开方案分析、Task E 桌面整理复现、轻量代码和公开资源说明。

这不是冠军方案，也不保证任何线上分数。教程把一次真实具身智能竞赛拆成可阅读、可检查、可复现的工程材料，并明确区分官方证据、局部调试证据和外部选手自报结果。

## 目录结构

```text
15-Challenge竞赛/ATEC2026/
├── README.md
├── L0-机器人徒步/                 # Task A / B2Piper locomotion 复现与评估
├── L0-桌面整理TaskE/              # Task E / Piper 桌面抓取与 ACT 复盘
├── TaskB-B2Piper/                 # Task B / B2-Piper 垃圾收集专题
├── docs/                          # Task A/B/D/E 共性边界与发布文档
├── code/                          # 轻量 adapter、参考仓库审计和检查脚本
└── references/                    # 外部公开方案、固定 commit 与许可证索引
```

## 任务总览

| 任务 | 机器人/能力 | 本目录入口 | 公开结论边界 |
|---|---|---|---|
| Task A / L0 | B2Piper 越野徒步 | [`L0-机器人徒步/`](./L0-机器人徒步/) | 重点是官方 baseline、局部评估、轨迹跟踪和 checkpoint 选择 |
| Task B | B2-Piper 垃圾收集 | [`TaskB-B2Piper/`](./TaskB-B2Piper/) | 公开完整调试与观测边界；没有核验过的完整最终 policy |
| Task D | 推箱越障 | [`docs/03_公开方案与Logic-TARS.md`](./docs/03_公开方案与Logic-TARS.md) | 重点参考公开 Logic-TARS 的状态机、LiDAR 和动作适配 |
| Task E / L0 | Piper 桌面整理 | [`L0-桌面整理TaskE/`](./L0-桌面整理TaskE/) | ACT seed1 best 线上最好回报为 `15.00`，不是满分方案 |

## 推荐阅读

1. [任务定义与观测边界](./docs/01_任务定义与观测边界.md)
2. [问题清单与解决方案](./docs/02_问题清单与解决方案.md)
3. [公开方案与 Logic-TARS 解读](./docs/03_公开方案与Logic-TARS.md)
4. [复现、HF 与 GitHub 发布](./docs/04_复现、HF与GitHub发布.md)
5. [归档对账与本地清理](./docs/05_归档对账与本地清理.md)
6. [Task B 专题归档](./TaskB-B2Piper/README.md)

## 公开资源

- 官方仿真仓库：[atecup/ATEC2026_Simulation_Challenge](https://github.com/atecup/ATEC2026_Simulation_Challenge)
- B2Piper L0 模型包：[Datawhale/atec2026-b2piper-l0](https://huggingface.co/Datawhale/atec2026-b2piper-l0)
- Task B 公开复现归档：[Datawhale/atec2026-task-b-reproducibility](https://huggingface.co/datasets/Datawhale/atec2026-task-b-reproducibility)
- Task E 模型：[Datawhale/atec2026-task-e-act-seed1-best](https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best)
- Task E 数据与日志：[Datawhale/atec2026-task-e-reproducibility](https://huggingface.co/datasets/Datawhale/atec2026-task-e-reproducibility)

## 复现约定

- GitHub 只保存教程、轻量代码、实验边界、固定外部 commit 和脱敏 Workspace Memory。
- 大数据、模型权重、官方源码快照、视频和日志按许可证与 SHA-256 记录在 Hugging Face。
- 完整仿真仍需要匹配的 Isaac Sim/Isaac Lab、GPU、Python 依赖、官方 runner 和评测器；不能把 GitHub 教程当成一键启动环境。
- 环境相机、真值状态、oracle prefix、脚本夹爪和阶段状态机可以用于诊断，但必须标注协议边界，不能冒充正式 policy 输入或完整成功证据。
- 原始训练缓存、私有日志、token、服务器凭据和未授权 SDK 不进入公开仓库。
