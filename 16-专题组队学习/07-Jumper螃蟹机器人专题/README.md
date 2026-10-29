# Jumper 螃蟹机器人：运动策略与创作专题

从金钢科技的 22 自由度螃蟹机器人 Jumper（跳跳）出发，一起学习机器人模型、六足步态、强化学习、仿真评估和外观场景创作。专题参考 [MicroDuck 专题](../05-MicroDuck运动策略与任务开发专题/README.md) 的组队学习与 Notebook 形式，先读懂并运行公开工具，再设计自己的动作和场景。

## 学习路线

| 阶段 | 内容 | Notebook | 产出 |
| --- | --- | --- | --- |
| 1 | 环境与资源 | [环境检查与项目地图](03-Notebook/01_环境检查与项目地图.ipynb) | 记录 Python、Git、GPU 和仓库版本 |
| 2 | 机器人与六足步态 | [模型与步态](03-Notebook/02_模型与六足步态.ipynb) | 找到模型、任务入口并解释三足步态 |
| 3 | 训练、回放与评估 | [动作训练与评估](03-Notebook/03_动作训练与评估.ipynb) | 保存配置、checkpoint、回放和评估记录 |
| 4 | 外观、场景与作品 | [外观场景与作品](03-Notebook/04_外观场景与作品.ipynb) | 设计 `.skin`、`.map` 并记录检查结果 |
| 5 | 上游动作包演示 | [跳跃与螃蟹舞](03-Notebook/05_官方动作包_跳跃与舞蹈.ipynb) | 播放并复现预训练动作 |

Notebook 默认只执行本地信息检查和教学示例；安装、下载、训练和部署命令以待执行步骤展示。Ubuntu 工作站的环境、CPU 仿真及实验过程见 [工作站复现记录](01-学习资料/Ubuntu工作站复现记录.md)。建议先打开第 1 个 Notebook，确认环境，再根据上游对应版本文档配置。

## Ubuntu 实跑结果

已完成环境检查、CPU 短训、GPU 续训和 GPU 从零训练。**自训三足策略已在仿真中走起来**：从零训练 1000 次迭代后的 checkpoint，在 200 Hz 物理的 10 秒回放中，0.3 m/s 指令对应平均前向速度 0.278 m/s，0.5 m/s 指令对应 0.459 m/s，均未重置；[查看最终策略视频](assets/tripod-fresh999.mp4)。此前从 CPU checkpoint 续训的策略在 0.3 m/s 指令下接近原地。日志、checkpoint、数据与复现步骤见[记录](01-学习资料/Ubuntu工作站复现记录.md)及[训练 Notebook](03-Notebook/03_动作训练与评估.ipynb)。

另有上游预训练动作包的[行走](assets/official-walk-follow.mp4)、[跳跃](assets/official-jump.mp4)和[螃蟹舞](assets/official-dance.mp4)演示，详细记录见[动作包 Notebook](03-Notebook/05_官方动作包_跳跃与舞蹈.ipynb)。上游动作包与自训三足 checkpoint 的来源不同。

## 公开资料

- [Jumper 主仓库与中文介绍](https://github.com/KingKongRobotics/jumper/blob/main/README.zh.md)：模型、训练、回放和动作包。
- [项目指南](https://github.com/KingKongRobotics/jumper/blob/main/docs/PROJECT_GUIDE.md)：环境、任务和当前集成边界。
- [动作训练教程](https://github.com/KingKongRobotics/jumper/blob/main/docs/TUTORIAL.md)：三足步态、跳跃、舞蹈、打包和回放。
- [Jumper Design](https://github.com/KingKongRobotics/jumper-design)：外观、场景及包格式。
- [Jumper Alebrije](https://github.com/madfam-org/jumper-alebrije)：社区外观与场景示例。
- [跳跳产品介绍](https://kingkong.tech/jumper)；[六足机器人硬件与固件参考](https://github.com/gitcfyang/hexapod_robot)（独立项目）。

资料核对日期：2026-10-29。上游更新后，以仓库对应版本的文档和代码为准。上游维护者自有内容采用 Apache-2.0，第三方材料及生成文件的许可需分别查看；这里不复制其代码或资源。模型和训练工具的开放不等于已提供完整实体制造资料。

## 组队与记录

见 [招募文案](招募文案.md) 和 [实验记录模板](01-学习资料/实验记录模板.md)。建议每次分享保留版本、配置、数据、视频与失败案例，区分公开策略回放、短训链路检查、正式评估及真机实验。
