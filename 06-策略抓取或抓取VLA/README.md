# 机器人策略、抓取与视觉语言动作模型

本专题介绍从行为克隆到视觉语言动作模型的机器人策略学习路线。建议先理解模型族谱和数据接口，再选择一个完整实验；论文导读用于比较不同模型如何处理时序记忆、三维空间、动作表征和在线适应。

## 基础阅读

- [视觉语言动作模型综述](01VLA相关总结综述.md)
- [RT-1、RT-2 与 RT-X](大模型控制、VLA、VLM/03RT系列论文解读与代码分析/01RT系列论文解读与代码分析.md)
- [OpenVLA 环境、评估与原理](大模型控制、VLA、VLM/02OpenVLA复现/02openvla复现.md)

## 完整实验

- [SmolVLA 在 LIBERO 上训练与评估](大模型控制、VLA、VLM/01SmolVLA-LIBERO/01SmolVLA-libero.md)
- [MuJoCo 中采集数据并训练 ACT、pi_0 与 SmolVLA](大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA/README.md)
- [DiT4DiT 在 LIBERO 上训练与评估](大模型控制、VLA、VLM/05DiT4DiT-LIBERO/01DiT4DiT-LIBERO训练与评估.md)
- [策略诊断与物理成功评估](大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA/09策略诊断与物理成功评估.md)

## 前沿方法导读

| 主题 | 章节 |
| --- | --- |
| 时序记忆与视觉证据 | [EventVLA](大模型控制、VLA、VLM/06EventVLA视觉证据记忆导读/README.md)、[VisualThink-VLA](大模型控制、VLA、VLM/15-VisualThink-VLA视觉证据推理导读/README.md) |
| 开源模型与工程框架 | [WALL-OSS](大模型控制、VLA、VLM/07WALL-OSS开源VLA模型导读/README.md)、[WALL-X](大模型控制、VLA、VLM/08WALL-X开源工程框架导航/README.md)、[EVA-Client](大模型控制、VLA、VLM/19-EVA-Client真机部署与评测工程导航/README.md) |
| 三维与物理理解 | [3DVLA](大模型控制、VLA、VLM/09-3DVLA三维空间实例增强VLA导读/README.md)、[PhysBrain](大模型控制、VLA、VLM/10-PhysBrain物理常识增强VLA导读/README.md) |
| 强化学习与在线适应 | [PRTS](大模型控制、VLA、VLM/11-PRTS强化学习原生VLA导读/README.md)、[LWD](大模型控制、VLA、VLM/14-LWD真机机群强化学习导读/README.md)、[Agentic-VLA](大模型控制、VLA、VLM/16-Agentic-VLA在线适应导读/README.md)、[Dexbotic-RLinf](大模型控制、VLA、VLM/17-Dexbotic-RLinf工程化VLA后训练导读/README.md) |
| 动作表征与高效推理 | [Galaxea G0.5](大模型控制、VLA、VLM/12-Galaxea-G0.5自回归VLA导读/README.md)、[Dexora](大模型控制、VLA、VLM/13-Dexora高自由度双臂灵巧VLA导读/README.md)、[DM0.5 与 OpenDM](大模型控制、VLA、VLM/18-DM0.5高性能推理与OpenDM导读/README.md) |

## 完成标准

- 能画出图像、语言、机器人状态、动作块和环境反馈之间的数据流。
- 能区分预训练模型评估、微调、短链路检查和完整闭环评估。
- 能为一次策略实验记录数据版本、模型配置、任务分母和逐回合结果。
