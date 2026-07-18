# ATEC2026 赛道2 L0 桌面整理 Task E 赛后复盘

本文整理 ATEC2026 线上赛赛道2 L0「桌面整理」Task E 的完整参赛流程。任务要求 Piper 桌面机械臂识别三种随机摆放物体，完成抓取并放入指定篮筐区域。我们最终线上最好回报为 `15.00`，不是满分；本开源目录的价值在于复盘真实工程路径、保留可提交代码骨架和失败路线证据，避免后来者重复踩坑。

## 任务目标

Task E 的核心流程是：

```text
视觉感知 -> 目标识别/定位 -> 运动规划或策略输出 -> 抓取 -> 运输 -> 篮筐释放
```

线上提交为文件式源码上传。我们最终保护的提交包来自 ACT seed1 best_loss 分支：

```text
code/act_seed1_best_submission/
  solution.py
  act/detr/*.py
  requirements.txt
  run.sh
  server.py
  UPLOAD_README.md
```

注意：`policy_act.pt` 未放入 GitHub，需要从本地提交包或 Hugging Face 模型仓库下载后放到同一目录。

## 最终分数与提交包

| 项目 | 结果 |
|---|---|
| 任务 | L0 桌面整理 Task E |
| 机器人 | `Piper` |
| 最好线上回报 | `15.00` |
| 最终保护方案 | ACT seed1 best_loss |
| 本地历史现象 | 单次可到 `18`，但多 seed 方差大 |
| 是否满分稳定 | 否 |

最终保护包来源：

```text
/home/ubuntu/Documents/01Proj/13atec/ATEC2026_Simulation_Challenge/submissions/task_e_act_seed1_best_20260608_upload
```

关键 SHA256：

```text
policy_act.pt  282614de9673dc01e229557448e380b6a02c196b5c3953cec77b2aebd2305a8e
solution.py    a08790e7512c914e5b78132303e1c6098c8aba3b01a373933e6ddc617fa3a485
```

## 我们尝试过的路线

### 1. 官方 / scripted oracle 数据生成

官方 Task E scripted oracle 对 object_3 更容易成功，但 object_1 / object_2 存在抓取接触、Piper TCP、USD root/几何中心和运输 primitive 不匹配问题。我们最终通过 per-object primitive 和闭环 servo 生成了三物体演示数据：

```text
datasets/atec_task_e_obj321_servo_100demos
```

这批数据支持 ACT 训练，但它不是一套可直接转成满分规划控制器的规则方案。

### 2. ACT / XSA-ACT

ACT 是最终最可用的提交路线。关键修正包括：

- 修正 Task E action 语义：动作是 `(joint_target - default_joint_pos) / 0.5`，不是速度或简单 delta。
- `solution_act.py` 支持自动加载 checkpoint 中的 `model_args`。
- XSA attention 改造可以作为 ACT 变体，但线上结果并不稳定。
- training loss 与任务成功率相关性弱，必须按 seed 实测。

线上试榜结果显示：

| 方案 | 线上回报 |
|---|---:|
| XSA-ACT 上传版 | `6.00` |
| ACT seed2 45000 | `12.00` |
| ACT seed2 50000 | `6.00` |
| ACT seed1 best | `15.00` |

因此本目录保护的是 ACT seed1 best，而不是单 seed 曾经满分的 seed2 checkpoint。

### 3. GraspNet / AnyGrasp / SAM3 / GraspGen-style PCA

我们接入并验证过多种规划抓取路线：

- TunTunClaw / GraspNet；
- AnyGrasp SDK；
- SAM3 图像分割；
- GraspGen-style PCA/AABB 几何抓取；
- basket-center release；
- pre-grasp approach / guarded insertion；
- object_1 方块“含进去再闭合”动作；
- object_3 香蕉低位 cradle/servo 搬运。

这些路线证明了“感知和候选抓取位姿”不是完全不可行，object_1 / object_3 在若干 seed 可以被夹起。但最终 submit-style `demo/solution_pca.py` 仍然在 IK、finger-center 追踪、接触保持和运输释放上不稳定，不能替代 ACT 做最后提交。

重要教训：

```text
AnyGrasp / GraspNet 不是一键满分控制器；
它只能给 grasp pose prior，后面仍需要 Piper IK、TCP 标定、夹爪接触保持和篮筐释放 primitive。
```

### 4. pi0.5 / OpenPI

我们系统尝试过 pi0.5/OpenPI：

- fixed-base 16D padding 旧分支；
- native8 state/action 分支；
- 20-demo / 100-demo；
- LoRA；
- pi05_base；
- raw unfiltered + absolute target action；
- action expert / action head 分支；
- zero noise / chunk 执行步数调参。

结果都没有超过 ACT，多个分支 seed11/12/13 为 `0/0/0` 或极低分。赛后排查显示，pi0.5 并不是“模型本身弱”，而是当前数据表达和相位建模不适合这个 Task E 轨迹：

- 过滤数据删除了大量 hold / 稳定帧，破坏 action chunk 时间语义；
- 单相机/任务 prompt 与真实抓取相位绑定弱；
- feed-forward 策略在 home/高位观测附近缺少连续相位，动作会在 home 与中段轨迹间抖动；
- 即使 raw absolute target 1-demo overfit，Isaac rollout 仍无法形成有效 pick/place。

因此 pi0.5 当前不作为提交候选。

## 本目录内容

```text
L0-桌面整理TaskE/
  README.md
  code/act_seed1_best_submission/
    UPLOAD_README.md
    solution.py
    requirements.txt
    run.sh
    server.py
    act/detr/*.py
  docs/
    ATEC2026_TaskE_规则物理配置与提交说明_20260604.md
    WORKSPACE_MEMORY_TaskE_完整调试记录.md
  hf_model_package/
    README.md
    policy_act.sha256
    task_e_act_seed1_best_20260608_upload.zip.sha256
```

`docs/WORKSPACE_MEMORY_TaskE_完整调试记录.md` 是完整工作记忆，包含大量中间失败路线、视频路径、checkpoint 路径和排障结论。它很长，但对复盘非常重要。

## 如何复现提交包

将 `policy_act.pt` 放入：

```text
L0-桌面整理TaskE/code/act_seed1_best_submission/policy_act.pt
```

然后源码上传时保持以下结构：

```text
solution.py
policy_act.pt
act/detr/backbone.py
act/detr/detr_vae.py
act/detr/position_encoding.py
act/detr/transformer.py
act/detr/utils.py
requirements.txt
```

如果平台按服务目录上传，也可带上：

```text
server.py
run.sh
```

## 为什么不直接把 policy_act.pt 放进 GitHub

`policy_act.pt` 约 64 MB，虽然低于 GitHub 单文件硬限制，但二进制权重不适合塞进教程仓库历史。推荐做法：

1. GitHub 存代码、README、SHA256 和提交说明；
2. Hugging Face 存权重和完整提交压缩包；
3. README 中维护 HF 链接和校验值。

## Hugging Face 链接

权重与完整提交压缩包已上传到 Datawhale Hugging Face：

```text
https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best
```

对应上传 commit：

```text
https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best/commit/4583bda9e3118e75987a0f91de62c155a70a2bed
```

上传命令、文件清单和 SHA256 见 `hf_model_package/README.md`。

## 能否安全删除本地内容

可以删，但建议按顺序：

1. 确认本目录已经 push 到 GitHub；
2. 确认 Hugging Face 已上传 `policy_act.pt` 和提交 zip；
3. 确认 SHA256 对得上；
4. 保留本目录和 `docs/WORKSPACE_MEMORY_TaskE_完整调试记录.md`；
5. 再删除 `/home/ubuntu/Documents/01Proj/13atec/ATEC2026_Simulation_Challenge` 下的大型训练输出、datasets、runs、logs/videos、openpi checkpoints。

不要在确认 HF 权重可下载前删除唯一的本地 `policy_act.pt`。
