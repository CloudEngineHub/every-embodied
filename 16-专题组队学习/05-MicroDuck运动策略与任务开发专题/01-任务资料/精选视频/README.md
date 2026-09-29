# MicroDuck 任务演示总览

这里按任务拆分演示，每本 Notebook 的 GPU 与 BPU 视频各自对应同一任务；展开条目可以看完整任务列表和额外版本。

**视频说明：** GPU GIF 展示 Notebook 中保存的策略推理回放；BPU GIF 展示 RDK X5 执行策略推理、Ubuntu 运行仿真的板端闭环记录。生成新 BPU 视频时，运行对应单元并连接任务 HBM、RDK 服务和仿真工作站。浏览器扰动演示通过鼠标拖动施加外力；梯面 Notebook 展示横档接触阶段，逐级攀爬演示见下方社区复现。

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

## 梯面攀爬：逐级踩横档

社区复现视频展示小鸭子逐级踩上梯子横档：[播放 11.77 秒攀爬视频](https://huggingface.co/HannesVonEssen/microduck-climb/resolve/main/media/preview.mp4)。同一项目提供攀爬与起身策略、ONNX、PPO 检查点、环境代码和梯子模型：[模型与演示页](https://huggingface.co/HannesVonEssen/microduck-climb) · [训练与复现代码](https://github.com/Vottivott/microduck-playground/tree/main/experiments/desk-climb)。本专题的本地横档接触阶段回放见下方“梯面攀爬”条目。

## 高跷行走：25 cm 与 200 cm

![25 cm 高跷行走](./assets/task-demos/stilts-25cm.gif)

![200 cm 高跷行走](./assets/task-demos/stilts-200cm.gif)

社区还发布了 10 cm 至 2 m 的八档高跷策略，每档包含独立视频、ONNX 和训练检查点：[高跷策略与视频合集](https://huggingface.co/HannesVonEssen/microduck-stilts)。其中 [200 cm 仿真视频](https://huggingface.co/HannesVonEssen/microduck-stilts/resolve/main/200cm/preview.mp4)展示了交替支撑行走。

## 其它任务版本

以下收录 Notebook 双视频之外的任务策略、本地回放和阶段性实验。

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

![篮球平衡策略预览](../01-MicroDuck篮球平衡强化学习/assets/preview.gif)
</details>

<details>
<summary>浏览器物理扰动</summary>

![浏览器物理扰动演示](../02-mjswan-MicroDuck浏览器物理扰动/assets/microduck_official.gif)

![94 秒人工拖拽演示压缩版](./assets/task-demos/browser-manual-drag-full.gif)

通过鼠标拖动小鸭子施加外力，观察仿真中的受力与运动变化。
</details>

<details>
<summary>摆动旋转策略与本地回放</summary>

![alpha050 摆动策略回放](./assets/task-demos/swing-public-alpha050.gif)

![本地策略 alpha050 摆动回放](./assets/task-demos/swing-local-alpha050.gif)

查看 alpha050 策略的摆动与旋转动作回放。
</details>

<details>
<summary>球平衡：MotrixLab 策略与本地 FastSAC</summary>

![MotrixLab 球平衡回放](./assets/task-demos/ball-balance-official.gif)

![本地 FastSAC 5000 iteration 回放](./assets/task-demos/ball-balance-local-5000iter.gif)

本地 FastSAC 5000 iteration 短训策略回放。
</details>

<details>
<summary>梯面攀爬：社区逐级攀爬与本地接触阶段</summary>

社区复现提供攀爬策略、起身策略、PPO 检查点、训练代码和梯子模型。演示中小鸭逐级踩上横档：[观看 11.77 秒攀爬视频](https://huggingface.co/HannesVonEssen/microduck-climb)；[直接打开 MP4](https://huggingface.co/HannesVonEssen/microduck-climb/blob/main/media/preview.mp4)；[训练与复现代码](https://github.com/Vottivott/microduck-playground/tree/main/experiments/desk-climb)。

![参考视频：MicroDuck 梯面攀爬](./assets/task-demos/ladder-reference.gif)

![V1 本地部分攀爬](./assets/task-demos/ladder-v1-partial.gif)

![V2 梯脚附近的 warm-start 阶段回放](./assets/task-demos/ladder-v2-bootstrap-preview.gif)

![V3 model 1100 阶段回放](./assets/task-demos/ladder-v3-model-1100.gif)

![V3 model 1238 重置阶段回放](./assets/task-demos/ladder-v3-model-1238.gif)

![组合片中的横档接触阶段](./assets/task-demos/ladder-stage-contact-suite.gif)

本地回放记录横档接触、靠近梯脚和不同训练阶段；社区逐级爬梯视频展示完整攀爬动作。
</details>

<details>
<summary>技能串联总览</summary>

![MicroDuck 技能串联总览压缩版](./assets/task-demos/microduck-skill-suite-overview.gif)

这是多任务剪辑总览；各任务的独立回放见上方对应条目。
</details>

## 素材与复现

- 所有提交的视频预览均为 GIF；Notebook 输出中的 MP4 已替换为 GIF 相对链接。
- 原始 MP4 不进入主仓库，减小克隆和浏览开销。各任务的环境、模型、指标和完整回放说明见对应任务 README。
- RDK X5 的板端任务视频属于每任务独立推理；Ubuntu 负责物理仿真和编码，BPU 负责动作推理。标注为“已保存”的视频可直接预览；运行 Notebook 的 BPU 单元可生成新的板端闭环视频。
