# MicroDuck 任务演示总览

这里按任务展示演示，GPU 推理在前、BPU 推理在后。展开条目可以看更多版本；训练、模型来源和实现步骤见各任务教程。

GPU 在工作站计算动作；BPU 在 RDK X5 计算动作，工作站负责仿真与画面渲染。网页展示的是已保存的任务视频，重跑步骤见 [Notebook 说明](../../03-Notebook/README.md)。

## Notebook：每项任务一组

| Notebook 与任务 | GPU 推理 | BPU 推理 |
| --- | --- | --- |
| `01` 篮球平衡 | ![篮球平衡 GPU 推理](./assets/notebook-replays/basketball_gpu.gif) | ![篮球平衡 BPU 推理](./assets/notebook-replays/basketball_bpu.gif) |
| `02` 浏览器物理扰动 | ![物理扰动 GPU 推理](./assets/notebook-replays/perturbation_gpu.gif) | ![物理扰动 BPU 推理](./assets/notebook-replays/perturbation_bpu.gif) |
| `03` 高跷行走 | ![高跷行走 GPU 推理](./assets/notebook-replays/stilt_gpu.gif) | ![高跷行走 BPU 推理](./assets/notebook-replays/stilt_bpu.gif) |
| `04` 摆动旋转 | ![摆动旋转 GPU 推理](./assets/notebook-replays/swing_gpu.gif) | ![摆动旋转 BPU 推理](./assets/notebook-replays/swing_bpu.gif) |
| `05` 球平衡 FastSAC | ![球平衡 GPU 推理](./assets/notebook-replays/ball_balance_gpu.gif) | ![球平衡 BPU 推理](./assets/notebook-replays/ball_balance_bpu.gif) |
| `06` 梯面攀爬 | ![梯面攀爬与登桌 GPU 推理](./assets/notebook-replays/ladder_gpu.gif)<br>逐级爬梯，登桌站稳 | ![爬梯与登桌 BPU 推理](./assets/notebook-replays/ladder_bpu.gif)<br>双策略切换，登桌站稳 |
| `07` 行走策略接入 | ![行走 GPU 推理](./assets/notebook-replays/walking_gpu.gif) | ![行走 BPU 推理](./assets/notebook-replays/walking_bpu.gif) |

上述 7 本 Notebook 保存了 65 个代码单元的执行计数与输出，视频均来自本轮 GPU/BPU 闭环；加上导航、HelloWorld 与跨缝跳跃，00–09 共 10 本、102 个代码单元均保留运行输出。任务表现和不稳定原因见 [Notebook 说明](../../03-Notebook/README.md)。页面展示压缩 GIF，21 段完整 MP4 与报告见 [视频目录](../../03-Notebook/recordings/README.md)。

## LightNav-0 多场景导航

[第 8 本 Notebook](../../03-Notebook/08_LightNav0_视觉语言导航_GPU_多场景.ipynb) 在 GPU 上接入预训练导航模型，读取第一视角图像与语言指令。三个场景分别生成视频；运行步骤、控制接口与模型来源见[导航教程](../../02-可运行代码/lightnav-learning/README.md)。

| 官方住宅 | 自建客厅 | 自建走廊 |
| --- | --- | --- |
| ![官方住宅语言导航](./assets/lightnav/official-apartment.gif)<br>住宅里寻找左侧桌子 | ![自建客厅语言导航](./assets/lightnav/living-room.gif)<br>客厅里接近扶手椅 | ![自建走廊语言导航](./assets/lightnav/corridor.gif)<br>走廊里寻找盆栽 |

## 跨缝跳跃

![MicroDuck 跨缝跳跃](../08-MicroDuck跨缝跳跃/assets/long_jump_continued.gif)

高平台起跳，跨过间隙，在低平台落地站稳。[第 9 本 Notebook](../../03-Notebook/09_跨缝跳跃_PPO_GPU_ONNX.ipynb)提供原始模型、10 轮 PPO 续训和 ONNX 的完整视频；模型来源、接口与奖励见[任务教程](../08-MicroDuck跨缝跳跃/README.md)。

## 梯面攀爬与登桌

![MicroDuck 逐级爬梯、登桌与站稳](./assets/notebook-replays/ladder_gpu.gif)

逐级踩横档，登桌后起身站稳。双策略切换、训练方法和模型来源见[梯面攀爬教程](../06-MicroDuck梯面攀爬强化学习导读/README.md)。

## 高跷行走：25 cm 与 200 cm

![25 cm 高跷行走](./assets/task-demos/stilts-25cm.gif)

![200 cm 高跷行走](./assets/task-demos/stilts-200cm.gif)

不同高度下交替支撑行走。形态课程、模型来源和尺寸配置见[高跷行走教程](../03-MicroDuck高跷行走强化学习复现/README.md)。

## 其它任务版本

以下收录各任务的更多动作和训练阶段。

<details>
<summary>行走、绕障与命令编舞</summary>

![平地行走](../07-RDK端侧与网页部署/assets/local_videos/microduck_4096env_6000iter_walk.gif)

速度与朝向命令：

![同一行走策略的 12 秒命令编舞](../07-RDK端侧与网页部署/assets/local_videos/microduck_command_dance.gif)

障碍场景：

![绕障行走](./assets/task-demos/walking-obstacle-policy.gif)
</details>

<details>
<summary>篮球平衡</summary>

![篮球平衡](../01-MicroDuck篮球平衡强化学习/assets/preview.gif)
</details>

<details>
<summary>浏览器物理扰动</summary>

![浏览器物理扰动演示](../02-mjswan-MicroDuck浏览器物理扰动/assets/microduck_official.gif)

![94 秒人工拖拽演示压缩版](./assets/task-demos/browser-manual-drag-full.gif)

通过鼠标拖动小鸭子施加外力，观察仿真中的受力与运动变化。
</details>

<details>
<summary>摆动旋转</summary>

![摆动旋转](./assets/task-demos/swing-public-alpha050.gif)

![摆动旋转训练演示](./assets/task-demos/swing-local-alpha050.gif)

摆动与旋转动作。
</details>

<details>
<summary>球平衡与 FastSAC 训练</summary>

![球平衡](./assets/task-demos/ball-balance-official.gif)

![FastSAC 球平衡训练](./assets/task-demos/ball-balance-local-5000iter.gif)

FastSAC 球面平衡训练。
</details>

<details>
<summary>梯面攀爬：动作分析与训练阶段</summary>

完整爬梯与登桌见上方演示；以下依次记录动作参考和 V1/V2/V3 训练阶段。实现过程见[梯面攀爬教程](../06-MicroDuck梯面攀爬强化学习导读/README.md)。

![参考视频：MicroDuck 梯面攀爬](./assets/task-demos/ladder-reference.gif)

![V1 本地部分攀爬](./assets/task-demos/ladder-v1-partial.gif)

![V2 梯脚接近](./assets/task-demos/ladder-v2-bootstrap-preview.gif)

![V3 横档接触](./assets/task-demos/ladder-v3-model-1100.gif)

![V3 起始姿态](./assets/task-demos/ladder-v3-model-1238.gif)

![组合片中的横档接触阶段](./assets/task-demos/ladder-stage-contact-suite.gif)

横档接触、梯脚接近和起始姿态。
</details>

<details>
<summary>技能串联总览</summary>

![MicroDuck 技能串联总览压缩版](./assets/task-demos/microduck-skill-suite-overview.gif)

多任务剪辑；各任务的完整演示见上方对应条目。
</details>

## 素材与复现

- 页面使用 GIF 预览；Notebook 保留图像、轨迹图和本轮 MP4 播放入口。
- 完整 MP4 与 JSON 报告独立收录在 `03-Notebook/recordings/`，不重复嵌入 Notebook。各任务的环境、模型、指标与复现步骤见对应任务 README。
- 运行 Notebook 可生成新的任务 MP4 与评测报告。
