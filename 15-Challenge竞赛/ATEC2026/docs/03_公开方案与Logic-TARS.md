# 3. 公开方案与 Logic-TARS 解读

## 3.1 公开方案检索结论

截至 2026-07-31，GitHub 上可以确认有官方仓库、公开 fork 和选手仓库，但没有找到 ATEC 官方公布的线上赛冠军/亚军完整源码。下面的评价按“代码是否存在、与任务接口是否接近、是否有完整控制闭环”区分，不把自报成绩当成官方成绩。

| 仓库 | 任务 | 公开内容 | 迁移判断 |
|---|---|---|---|
| [Logic-TARS/ATEC2026](https://github.com/Logic-TARS/ATEC2026) | Task D | B2W + Piper；课程训练；61D/16D policy；16D→24D adapter；状态机、LiDAR、航向修正、卡死恢复 | **Task D 首选参考** |
| [ZSN2024/ATEC2026_Simulation_Challenge](https://github.com/ZSN2024/ATEC2026_Simulation_Challenge) | Task B | B2wPiper Stage1 PPO、command scheduler、obs/action adapter、训练/导出/smoke 命令 | **Task B 训练骨架** |
| [yma867/ATEC2026_Simulation_Challenge_RIL](https://github.com/yma867/ATEC2026_Simulation_Challenge_RIL/tree/main/taskb_perception) | Task B | YOLO、ByteTrack、RGB-D 深度反投影、世界坐标输出 | **Task B 感知参考** |
| [Lhy6900/ATEC2026_Simulation_Challenge](https://github.com/Lhy6900/ATEC2026_Simulation_Challenge/tree/main/demo) | G1 / GR00T | whole-body adapter、blind locomotion、teleop | 机器人接口不同，不能直接替换 B2/B2W policy |
| [StevenLiudw/Clear_ATEC2026_Simulation_Challenge](https://github.com/StevenLiudw/Clear_ATEC2026_Simulation_Challenge/tree/main/demo) | Task B 辅助 | locomotion smoke test、ACT/RL demo | 适合 baseline 对照 |

## 3.2 Logic-TARS 为什么最值得看

Logic-TARS 的 README 自报排名为 `32/100`。这个数字只能作为仓库作者的公开说明，不能当作官方成绩证明；真正有迁移价值的是它把训练接口、部署接口和高层控制逻辑拆开了。

### 训练与部署接口

其核心契约可以抽象为：

```text
policy observation: 61D
policy action:      16D = 12D leg + 4D wheel
official action:    24D = 12D leg + 4D wheel + 8D Piper arm
```

这说明手臂动作不是被 policy 随意忽略，而是在 deployment adapter 中明确固定或置零。这样做可以让 locomotion policy 继承已有底盘能力，同时满足官方提交接口。

### 课程训练

公开脚本体现了递进式路线：

```text
flat locomotion
  -> rough straight walking
  -> rough omni B2W
  -> Task D fine-tuning
  -> export
  -> official submission adapter
```

这条路线适合借鉴训练组织方式，但不能直接复制 checkpoint。不同机器上的 Isaac Sim、机器人资产、关节默认值和随机种子都会影响策略行为。

### Task D 高层状态机

其提交入口包含以下阶段：

```text
approach box
  -> push box
  -> navigate platform
  -> climb finish
```

阶段切换同时参考当前得分、箱体 LiDAR 检测、距离、bearing 和机器人进度。相比单一速度命令，状态机更容易定位“走路失败”“接触失败”“平台失败”分别发生在哪一段。

### 61D 观测的任务信息

公开代码在基础 locomotion 观测后加入任务信息，包括：

- 归一化当前分数；
- 分数阶段 one-hot；
- 箱体是否被 LiDAR 看到；
- 箱体 bearing；
- 箱体距离归一化值。

这种设计的关键是把“任务已经完成了哪一段”和“箱子现在在哪里”同时送进 policy，而不是只给机器人本体状态。

## 3.3 可以搬过来的部分

### 高优先级

1. 16D 到 24D 的动作适配契约；
2. score-aware phase switching；
3. LiDAR 箱体 bearing/distance 的最小检测接口；
4. heading lock 和速度修正；
5. 长时间 score 不增长时的 stuck recovery；
6. flat 到 rough 再到 Task D fine-tuning 的训练顺序。

### 需要重新确认

1. 我们当前分支到底使用 B2Piper 还是 B2wPiper；
2. 61D 的每一维与本地 observation 的精确顺序；
3. wheel action 的 scale、符号和四个轮子的顺序；
4. Piper 8D 固定动作在当前官方版本是否仍然允许置零；
5. score 阈值与当前官方评分器是否一致；
6. LiDAR ray 数量、角度范围和高度过滤参数；
7. 外部 checkpoint 是否可获得，以及其许可证和官方资产依赖。

## 3.4 不能直接搬过来的部分

- 不能把 `submission/solution.py` 直接复制到 B2Piper；
- 不能把 Task D 的箱体状态机当成 Task B 垃圾抓取控制器；
- 不能假设全局相机或仿真世界坐标在官方提交中可用；
- 不能把自报 `32/100` 写成 Datawhale 的官方排名；
- 不能把缺少 checkpoint 的源码仓库包装成开箱即用的高分方案。

## 3.5 外部仓库固定版本

本教程的外部方案清单固定到以下公开 commit，便于未来代码发生变化后仍能复核：

| 仓库 | commit | 许可证 |
|---|---|---|
| Logic-TARS/ATEC2026 | `b78c4afd1b84302fe8f88bcfd287eac64c33692c` | MIT |
| ZSN2024/ATEC2026_Simulation_Challenge | `ee4e0eb97928754d9404a3acd5d644020ac7794c` | MIT |
| yma867/ATEC2026_Simulation_Challenge_RIL | `e56a2a9e39c5231a91c0a8b1cce8ab1bc0e72403` | MIT |

获取固定版本：

```bash
bash code/external_repo_audit/clone_references.sh
```

脚本只克隆公开源码，不下载模型权重或仿真缓存。外部代码仍然属于原作者，使用时应保留其许可证和上游链接。
