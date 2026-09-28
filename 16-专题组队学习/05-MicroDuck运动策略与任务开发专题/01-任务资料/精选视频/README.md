# MicroDuck 任务演示总览

这里按任务拆分演示，避免把不同策略、算法或物理交互剪成一个结果。每本 Notebook 的 GPU 与 BPU 视频各自对应同一任务；展开条目可以看完整任务列表和额外版本。

> **视频口径：** GPU GIF 是已保存的 Notebook 推理回放；BPU GIF 是 Notebook 先前运行留下的板端闭环视频缓存，不表示读者打开 GitHub 时正在实时调用 RDK。重新运行 BPU 单元需要对应任务 HBM、RDK 服务和仿真工作站。浏览器物理扰动是人工拖拽/外力交互，不是训练出的抗扰恢复策略。梯面目前只有阶段性接触结果，不是连续爬梯或登顶。

## Notebook：每项任务一组

| Notebook 与任务 | GPU 推理 | 先前保存的 BPU 闭环回放 |
| --- | --- | --- |
| `01` 篮球平衡 | ![篮球平衡 GPU 回放](./assets/notebook-replays/basketball_gpu.gif) | ![篮球平衡 BPU 缓存回放](./assets/notebook-replays/basketball_bpu_cached.gif) |
| `02` 浏览器物理扰动 | ![物理扰动 GPU 回放](./assets/notebook-replays/perturbation_gpu.gif) | ![物理扰动 BPU 缓存回放](./assets/notebook-replays/perturbation_bpu_cached.gif) |
| `03` 高跷行走 | ![高跷行走 GPU 回放](./assets/notebook-replays/stilt_gpu.gif) | ![高跷行走 BPU 缓存回放](./assets/notebook-replays/stilt_bpu_cached.gif) |
| `04` 摆动旋转 | ![摆动旋转 GPU 回放](./assets/notebook-replays/swing_gpu.gif) | ![摆动旋转 BPU 缓存回放](./assets/notebook-replays/swing_bpu_cached.gif) |
| `05` 球平衡 FastSAC | ![球平衡 GPU 回放](./assets/notebook-replays/ball_balance_gpu.gif) | ![球平衡 BPU 缓存回放](./assets/notebook-replays/ball_balance_bpu_cached.gif) |
| `06` 梯面攀爬 | ![梯面攀爬 GPU 阶段回放](./assets/notebook-replays/ladder_gpu.gif) | ![梯面攀爬 BPU 缓存阶段回放](./assets/notebook-replays/ladder_bpu_cached.gif) |
| `07` 行走/导航接入 | ![行走 GPU 回放](./assets/notebook-replays/walking_gpu.gif) | ![行走 BPU 缓存回放](./assets/notebook-replays/walking_bpu_cached.gif) |

上述 7 本 Notebook 共保存了 64 个代码单元的执行计数与输出。完整执行快照在 Notebook 中；为避免重复存储大段 MP4，视频输出改为链接到本目录的压缩 GIF。运行状态和可选步骤见 [Notebook 说明](../../03-Notebook/README.md)。

## 其它任务版本

以下是 Notebook 双视频之外的公开策略、本地回放和阶段性实验。每段都保留自己的来源与结果边界。

<details>
<summary>行走、绕障与命令编舞</summary>

![4096 环境、6000 次迭代的行走策略回放](../07-RDK端侧与网页部署/assets/local_videos/microduck_4096env_6000iter_walk.gif)

训练策略的命令编舞回放：

![同一行走策略的 12 秒命令编舞](../07-RDK端侧与网页部署/assets/local_videos/microduck_command_dance.gif)

MuJoCo 障碍几何中的行走回放：

![障碍行走回放](./assets/task-demos/walking-obstacle-policy.gif)
</details>

<details>
<summary>篮球平衡</summary>

![公开策略篮球平衡预览](../01-MicroDuck篮球平衡强化学习/assets/preview.gif)
</details>

<details>
<summary>浏览器物理扰动</summary>

![浏览器物理扰动官方演示](../02-mjswan-MicroDuck浏览器物理扰动/assets/microduck_official.gif)

![94 秒人工拖拽演示压缩版](./assets/task-demos/browser-manual-drag-full.gif)

拖拽是人为施加的外力；不要据此声称策略已经学会了恢复动作。
</details>

<details>
<summary>高跷行走：25 cm 与 200 cm</summary>

![25 cm 高跷仿真回放](./assets/task-demos/stilts-25cm.gif)

![200 cm 高跷仿真回放](./assets/task-demos/stilts-200cm.gif)
</details>

<details>
<summary>摆动旋转：公开版本与本地回放</summary>

![公开策略 alpha050 摆动回放](./assets/task-demos/swing-public-alpha050.gif)

![本地策略 alpha050 摆动回放](./assets/task-demos/swing-local-alpha050.gif)

alpha050 回放不代表连续完成完整 360 度旋转。
</details>

<details>
<summary>球平衡：公开版本与本地 5000 iteration</summary>

![MotrixLab 公开球平衡回放](./assets/task-demos/ball-balance-official.gif)

![本地 FastSAC 5000 iteration 回放](./assets/task-demos/ball-balance-local-5000iter.gif)

本地 5000 iteration 是短训演示，不代表训练已收敛或超过公开策略。
</details>

<details>
<summary>梯面攀爬：参考、V1/V2 与 V3 阶段审计</summary>

![参考视频：MicroDuck 梯面攀爬](./assets/task-demos/ladder-reference.gif)

![V1 本地部分攀爬](./assets/task-demos/ladder-v1-partial.gif)

![V2 梯脚附近的 warm-start 阶段回放](./assets/task-demos/ladder-v2-bootstrap-preview.gif)

![V3 model 1100 阶段回放](./assets/task-demos/ladder-v3-model-1100.gif)

![V3 model 1238 重置阶段回放](./assets/task-demos/ladder-v3-model-1238.gif)

![组合片中的横档接触阶段](./assets/task-demos/ladder-stage-contact-suite.gif)

这些视频覆盖真实横档接触、入口失败和阶段审计；当前不能表述为连续爬完整架梯子或成功登顶。
</details>

<details>
<summary>技能串联总览</summary>

![MicroDuck 技能串联总览压缩版](./assets/task-demos/microduck-skill-suite-overview.gif)

这是多任务剪辑总览，不是一个统一策略，也不能代替上面每个任务的独立回放。
</details>

## 素材与复现

- 所有提交的视频预览均为 GIF；Notebook 输出中的 MP4 已替换为 GIF 相对链接。
- 原始 MP4 不进入主仓库，减小克隆和浏览开销。各任务的环境、模型、指标和完整回放说明见对应任务 README。
- RDK X5 的板端任务视频属于每任务独立推理；Ubuntu 负责物理仿真和编码，BPU 负责动作推理。当前页面中标注“缓存”的视频来自此前保存的执行输出。
