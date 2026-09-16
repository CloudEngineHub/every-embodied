---
license: apache-2.0
library_name: onnx
tags:
  - robotics
  - reinforcement-learning
  - microduck
  - mujoco
  - ball-kick
---

# Microduck BallKick 4096x6000

该仓库保存 every-embodied 教程复现的 Microduck 右脚踢球策略。训练基于 `pollen-robotics/microduck_rl` 提交 `29e887ecfbf5d37144759e5a9f8a176dfb83d547`，任务为 `Mjlab-BallKick-Flat-MicroDuck`。

## 训练记录

- 并行环境：4096
- PPO 迭代：6000
- 累计 transition：589,824,000
- 训练耗时：2 小时 6 分 15 秒
- 最终 mean reward：89.42
- 最终 mean episode length：250.0
- 最终 `ball_forward_velocity` reward：11.4758

这些数值只描述本次运行，不应脱离代码提交、任务配置和随机种子作为跨项目榜单结果。

## 文件

- `model_5999.pt`：最终 RSL-RL checkpoint。
- `ball_kick_right_6000.onnx`：供 Microduck 61 维观测、14 维动作运行时使用的右脚策略。
- `microduck_ballkick_visual_qa.json`：训练、浏览器闭环、接触扫描与演示视频的机器可读记录。

## 校验值

```text
cf3f34ed68dec9137260d20f4e2cfdd507415df651d86fe31f2ac9daa8073c7c  model_5999.pt
bd03168c2554de6f95a69903143997c3ff85fec73c28a010225868216d3ea518  ball_kick_right_6000.onnx
```

## 使用边界

该策略不接收图像或球位置，只执行一次盲踢。教程网页使用独立的视觉检测接口和有限状态机完成找球、接近、脚位对齐与策略切换，并提供头部第一视角、近距离第二视角和居中第三视角。浏览器演示中的检测框由 MuJoCo 真值投影得到，用于验证控制接口，不代表训练或评测了视觉检测器。

完整说明、命令与视频见 [every-embodied](https://github.com/datawhalechina/every-embodied) 的 Microduck 双足强化学习章节。
