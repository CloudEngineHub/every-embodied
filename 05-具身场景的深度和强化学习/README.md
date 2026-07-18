# 具身场景的深度和强化学习

本目录收集强化学习、模仿学习和深度学习在具身场景中的复现教程。建议大家先从任务边界和仿真平台开始理解，再进入具体代码和训练命令。

## 章节入口

- [多机器人搬运家具强化学习](01多机器人搬运家具强化学习.md)：CooHOI 协作式人形机器人搬运任务入门。
- [HIMLoco 四足机器人运动控制 Isaac Lab 复现](02HIMLoco-IsaacLab复现/README.md)：从论文理解、旧 Isaac Gym 栈排障到 Isaac Lab 新栈 smoke test。
- [AGILE 人形机器人 Loco-Manipulation Isaac Lab 复现](03AGILE人形机器人Loco-Manipulation复现/README.md)：理解 AGILE 的人形机器人 RL 工作流、官方任务边界、本地 Isaac Sim 5.1 复刻视频和评估导出链路。
- [Robots That Know What to Ask 奖励对齐导读](04-Robots-That-Know-What-to-Ask奖励对齐导读/README.md)：拆解 ASQ 如何识别示教中的欠指定奖励特征，并用自然语言解释引导人类补充 corrective demonstrations。

## 学习建议

如果大家刚接触机器人强化学习，建议先看 HIMLoco 章节，理解 proprioception、privileged critic 和 sim-to-real 的基本思路；再看 AGILE 章节，学习更完整的人形机器人任务配置、teacher-student 蒸馏、评估报告和 Sim2MuJoCo 复核流程。对 VLA、世界模型后训练或人类示教数据感兴趣的同学，可以继续看 Robots That Know What to Ask，重点理解 reward misalignment、欠指定特征和 human-in-the-loop 奖励修正。
