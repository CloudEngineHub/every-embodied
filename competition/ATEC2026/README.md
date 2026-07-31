# ATEC2026 仿真挑战赛：赛后复盘、开源方案与可复现工程

本目录记录 Datawhale / Every Embodied 在 ATEC2026 仿真挑战中的公开复盘。内容覆盖任务定义、传感器和动作接口、调试过程、失败原因、公开选手方案、Logic-TARS 代码解读，以及 GitHub 与 Hugging Face 的发布边界。

这不是冠军方案，也不保证任何线上分数。它的目标是把一次真实的具身智能比赛工程拆成可以检查、可以复现、可以继续改进的学习材料。

## 这一章解决什么问题

读完本目录，可以回答下面几个容易混淆的问题：

- 垃圾收集、推箱越障、机器人徒步分别属于哪一个任务；
- 官方观测里有哪些相机和 LiDAR，能不能自行添加全局相机；
- 为什么打开场景后看不到垃圾、机器人或障碍物；
- B2Piper、B2wPiper 的动作维度为什么不同；
- 为什么通用 locomotion policy 不能直接解决垃圾抓取或推箱；
- 哪些公开选手代码可以借鉴，哪些只能当感知或工程参考；
- 如何把 GitHub 轻量代码和 Hugging Face 大文件拆开发布。

## 目录

| 文档 | 内容 |
|---|---|
| [任务定义与观测边界](./docs/01_任务定义与观测边界.md) | Task A/B/D/E 的关系、全局相机边界、动作与观测契约 |
| [问题清单与解决方案](./docs/02_问题清单与解决方案.md) | 比赛中遇到的场景、相机、得分、训练、提交和复现问题 |
| [公开方案与 Logic-TARS 解读](./docs/03_公开方案与Logic-TARS.md) | 公开仓库分级、Logic-TARS 的 Task D 方案和迁移边界 |
| [复现、HF 与 GitHub 发布](./docs/04_复现、HF与GitHub发布.md) | 目录规划、smoke test、权重发布、公开前检查 |
| [中文 Workspace Memory](./docs/WORKSPACE_MEMORY_ATEC2026_赛后复盘.md) | 后续继续工作时需要保留的事实、假设和实验纪律 |

轻量代码在 [code/](./code/)；外部方案索引在 [references/](./references/)。

## 官方资料

- 官方仿真挑战仓库：[atecup/ATEC2026_Simulation_Challenge](https://github.com/atecup/ATEC2026_Simulation_Challenge)
- 官方比赛页面：[ATEC2026](https://www.atecup.com/competitions/100017)
- Every Embodied 仓库：[datawhalechina/every-embodied](https://github.com/datawhalechina/every-embodied)

## 已有 Datawhale Hugging Face 资源

- [Datawhale/atec2026-b2piper-l0](https://huggingface.co/Datawhale/atec2026-b2piper-l0)：B2Piper L0 徒步 baseline / Option-MoE 模型包。
- [Datawhale/atec2026-task-e-act-seed1-best](https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best)：Task E ACT 模型权重。
- [Datawhale/atec2026-task-e-reproducibility](https://huggingface.co/datasets/Datawhale/atec2026-task-e-reproducibility)：Task E 完整复现资料和大文件。

本目录新增的 Task D 外部方案审计、适配代码和发布清单会先进入 GitHub；需要保存的完整源码快照、日志和大文件放入 Datawhale 的公开 Hugging Face 数据集，并在这里记录版本与 SHA256。

本次公开复现包：[Datawhale/atec2026-simulation-challenge-reproducibility](https://huggingface.co/datasets/Datawhale/atec2026-simulation-challenge-reproducibility)

## 一句话结论

- 如果研究 **Task D 推箱越障**，优先阅读 [Logic-TARS/ATEC2026](https://github.com/Logic-TARS/ATEC2026)，重点看 61D/16D policy 接口、16D 到 24D 官方动作适配、状态机、LiDAR 箱体跟踪和 stuck recovery。
- 如果研究 **Task B 垃圾收集**，建议把 [ZSN2024/ATEC2026_Simulation_Challenge](https://github.com/ZSN2024/ATEC2026_Simulation_Challenge) 的 B2wPiper Stage1 训练链路与 [yma867/ATEC2026_Simulation_Challenge_RIL](https://github.com/yma867/ATEC2026_Simulation_Challenge_RIL/tree/main/taskb_perception) 的 RGB-D 感知接口组合起来，再自行完成行走、抓取和投放闭环。
- 这些仓库是公开选手实现或 fork，不等于官方冠军方案。凡是自报排名，都在文档中标成“自报”，不替代官方成绩证明。
