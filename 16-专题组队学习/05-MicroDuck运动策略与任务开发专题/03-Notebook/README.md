# MicroDuck 直播 Notebook 学习线

本专题主线有 8 本 Notebook，对应 8 天学习，另有第 9 本跨缝跳跃拓展实验。前 7 本的运动任务视频顺序是 GPU 推理在前、RDK X5 BPU 推理在后；第 8 本 LightNav-0 多场景导航只使用工作站 GPU，不连接开发板。

组队学习按 [8 天学习计划](../学习计划.md)推进，每天对应下方一本 Notebook，围绕任务学习、实验与交流。

月度组队学习使用同一组 Notebook，每个 Task 安排 2–3 天，共 21 天，日期与作业截止时间见[月度招募文案](../月度组队学习招募文案.md)。两套学习安排分别保留。

Notebook 保存了单元输出，适合在 GitHub 上先阅读结果，再在 Ubuntu/Jupyter 环境按需重跑。00–09 共 10 本 Notebook、102 个代码单元均已执行并保存输出；本轮还在开发板开机后重新执行了前 7 本的 GPU/BPU 闭环。运动任务重新计算关节动作并生成视频，导航记录实际模型预测；可选训练是否执行，见各单元状态。

## 本轮验证

[验证记录](./验证记录.json)保存模型校验值、真实推理次数、数值对照和任务表现。前 7 本的模型检查与 GPU 闭环使用同一模型，缺少模型、环境错误、旧视频或推理提前结束都会报错。

| 实验 | GPU / 浮点策略 | RDK X5 BPU |
| :-- | :-- | :-- |
| HelloWorld | CPU ONNX + MuJoCo：550 步，前进、左转、停止 | 不连接开发板 |
| 篮球 | 1500 步、30 秒，无摔倒或重置 | 新校准 INT16 HBM，1500 步、30 秒，无摔倒或重置 |
| 高跷、摆动、行走 | 各 500 步、10 秒，无终止重置 | 各 500 步、10 秒，无终止重置 |
| 物理扰动 | 500 步，施加扰动后发生 1 次重置 | 500 步，发生 1 次重置 |
| Motrix 球平衡 | 恢复 5002 轮训练模型，900 步、18 秒，无摔倒或重置 | 900 步、18 秒，无摔倒或重置 |
| 梯面攀爬 | 1500 步、30 秒，登桌后连续站稳 15.74 秒 | 新校准双模型，1500 步、30 秒，登桌后连续站稳 17.24 秒 |
| LightNav-0 | 三个场景实际预测与闭环行走；保留停止原因和目标距离 | 不连接开发板 |
| 跨缝跳跃 | 原始模型、10 轮续训候选与 ONNX 分别评测；按落地与站稳结果选择模型 | 不连接开发板 |

7 个运动任务的 BPU 视频均重新生成，报告核对运行编号、板端 HBM SHA 和实际控制步数。RDK 计算每一步动作，Ubuntu 将动作送入 MuJoCo/MotrixSim 并录制物理仿真画面。全套共 21 段真实推理视频，已逐帧解码检查；表中结果对应本次固定场景和种子，扰动实验保留完整重置过程。

### 不稳定的原因

[诊断记录](./诊断记录.json)保存源模型对照、同一真实观测下的浮点/BPU 动作误差，以及爬梯的重复试验：

- Motrix 球平衡恢复训练到 5002 轮的 checkpoint 与归一化，PyTorch/ONNX 动作最大误差约 `1.5e-6`；板端移除错误的 `[-1, 1]` 动作截断，按关节目标的实际单位下发。短程训练只生成候选，不自动替换默认稳定模型。
- 篮球重新使用 506 组真实观测、LSTM hidden/cell 配对数据做 max 校准，并启用 INT16。状态输出先反量化，再按下一步输入的独立比例量化，修正了直接按 INT8 复制状态的问题。
- 爬梯保留原有接触、原始动作槽、滤波与增益规则；物理接触使用 CPU 后端，策略分别由 GPU/BPU 计算。两套模型移出固定归一化，拆分全连接权重为主值与残差后校准 INT16。两次板端对照均完成登桌与站稳。
- 回合结束时清空 BPU LSTM 记忆，避免仿真自动重置后再次手动重置；视频的回合时限延长到录制结束之后，失败重置与超时重置分别计数。

## MotrixLab 启动入口更新（2026-10-25）

新版 MotrixLab 使用 `mxlab train`、`mxlab play` 和 `mxlab export-onnx`，不再通过 `scripts/train.py` 等脚本启动。05 号 Notebook 和生成模板已适配，配套启动器会自动选择当前环境的新旧入口，不需要改用 Windows，也不需要补建 `train.py`。安装、Ubuntu/WSL 与 PowerShell 命令、10 轮训练示例和视频录制见[球平衡教程第 4 节](../01-任务资料/05-MotrixLab-MicroDuck球平衡与FastSAC/README.md#4-安装与运行使用-mxlab-入口)。已有视频与运行记录保留其原始结果。

版本对照：新版命令固定为 **MotrixLab `0.4.0` / 提交 `24796cff1c00`**；已有训练与视频使用的旧版代码基线为 **`0.4.0b0` / 提交 `d093f94fbd87`**。完整提交号、Python 要求、依赖锁定及版本检查命令见[版本与复现基线](../01-任务资料/05-MotrixLab-MicroDuck球平衡与FastSAC/README.md#版本与复现基线)。

## 第一次运行：Python 控制小鸭子

先运行 [00_MuJoCo策略控制_HelloWorld.ipynb](./00_MuJoCo策略控制_HelloWorld.ipynb)，再进入任务 Notebook。它是入门练习，不改变原有 8 天或月度学习安排。

1. 下载完整仓库，按[最小环境说明](../02-可运行代码/hello-world/README.md)安装依赖并启动 Jupyter。
2. 打开 HelloWorld，按顺序执行所有单元；行走模型自动下载并校验，无需填写模型路径。
3. 在同一个 MuJoCo 场景中执行 `sim.forward()`、`sim.turn()` 和 `sim.stop()`，查看实时画面、响应曲线与新生成的视频。

Notebook 直接展开控制循环：`sim.observation()` 读取当前物理状态，ONNX Runtime 计算动作，`sim.data.ctrl` 接收关节位置目标，`mujoco.mj_step()` 推进物理仿真，再读取下一步状态。修改的是模型观测中的速度命令，而非机器人位置。停止命令归零后，策略继续控制关节保持站立。

本例使用 CPU 推理与原生 MuJoCo，无需训练框架、显卡或 RDK。实际运行了 550 个控制步：前进约 0.48 m，左转约 44°，停止后平均速度接近零。[执行记录](../01-任务资料/07-RDK端侧与网页部署/assets/hello_world/validation.json)保留模型校验值和测量结果。

其它 Notebook 的模型检查单元先确认输入输出接口；随后任务闭环单元会启动对应策略与仿真器，重新计算动作并生成视频。各任务需要自己的模型和运行环境，按任务教程准备；仅播放保存的 GIF 不会启动仿真。

## Notebook 清单

| Notebook | 主线 |
| :-- | :-- |
| [00_MuJoCo策略控制_HelloWorld.ipynb](./00_MuJoCo策略控制_HelloWorld.ipynb) | 自动下载行走模型、前进/转向/停止、实时 ONNX → MuJoCo（入门） |
| `01_篮球平衡_PPO_ONNX_BPU_MuJoCo.ipynb` | 速度命令、球场景、PPO、策略导出 |
| `02_浏览器物理扰动_回放与接口.ipynb` | MuJoCo 状态、外力扰动、网页回放 |
| `03_高跷行走_课程与ONNX_BPU.ipynb` | BAM、形态课程、动作平滑、部署契约 |
| `04_摆动旋转_ONNX_BPU_MuJoCo.ipynb` | 摆动任务和仓库内置 ONNX 示例 |
| `05_球平衡_FastSAC_ONNX_BPU.ipynb` | FastSAC 与另一套任务接口的对照 |
| `06_梯面攀爬_接触与部署模板.ipynb` | 爬梯、登桌、起身双策略切换和课程训练 |
| `07_RDK网页与多策略_BPU验收.ipynb` | 行走策略直接接入、RDK TCP、多策略切换、BPU 验收 |
| `08_LightNav0_视觉语言导航_GPU_多场景.ipynb` | 预训练模型、图像与语言、MPC、住宅/客厅/走廊 |
| `09_跨缝跳跃_PPO_GPU_ONNX.ipynb` | 起跳与落地、10 轮 PPO 续训、GPU 视频和 ONNX 对齐（拓展） |

## 跨缝跳跃拓展

[第 9 本 Notebook](./09_跨缝跳跃_PPO_GPU_ONNX.ipynb)从高平台跨过 30 cm 间隙，在低 25 cm 的目标平台落地。先运行原始模型，再实际续训 10 轮并导出 ONNX，比较三段独立生成的视频。模型和环境准备见[配套代码](../02-可运行代码/long-jump-learning/README.md)，接口与奖励见[任务教程](../01-任务资料/08-MicroDuck跨缝跳跃/README.md)。该拓展使用 GPU 仿真和 ONNX 接口验证，不连接 RDK，不改变现有学习日程。

13 个代码单元已全部运行并保存输出；续训实际运行了 64 环境 × 10 轮 PPO 更新。Notebook 分别展示原始模型和续训候选的表现，再按落地、站稳和 12 秒回合评测选择部署模型。短程续训可能降低任务表现，候选未通过时不会替换原始模型。Notebook 内保留轨迹图、训练结果和 ONNX 数值对照，GitHub 预览使用本次生成的 GIF。

## 第 8 天：视觉语言导航

[08_LightNav0_视觉语言导航_GPU_多场景.ipynb](./08_LightNav0_视觉语言导航_GPU_多场景.ipynb) 使用预训练 LightNav-0，不训练导航模型。
在 GPU［图形处理器］上运行导航模型，连接已有 MicroDuck 行走策略，分别体验官方住宅、自建客厅和自建走廊。
每个场景生成独立视频、模型预测记录与轨迹图。环境准备和默认路径见[配套说明](../02-可运行代码/lightnav-learning/README.md)。
按学习计划在第 8 天运行，结合视频理解导航模型与行走策略如何协作。

导航 Notebook 已在 Ubuntu 从头执行：12 个代码单元全部保留输出，包含三个场景的第一视角、实际预测、视频和轨迹图。
视频由本次模型推理生成，仓库保留独立 MP4；运行后在对应单元内直接播放。

每本 Notebook 的视频见[演示总览](../01-任务资料/精选视频/README.md)。行走/导航直接接入已有策略；扰动用于物理交互；梯面 GPU/BPU 最新视频均完成爬梯、登桌与站稳。其它任务提供对应算法的训练/续训入口，实际运行情况见保存的单元输出。

## 启动

```bash
cd 02-可运行代码/microduck-playground-stilts
uv sync
uv run --with jupyter jupyter lab ../../03-Notebook
```

在 Ubuntu 工作站上启动 Jupyter。Notebook 会自动向上查找 `02-可运行代码`；如果工作目录不是专题目录，设置 `MICRODUCK_TOPIC_ROOT`。

## GPU 与 BPU 视频

前 7 本 Notebook 先运行 GPU 闭环；连接开发板后再启用 BPU 单元。BPU 计算动作，工作站负责物理仿真和编码。启用后在 `03-Notebook/outputs/bpu_videos/` 生成任务 MP4 和 JSON 报告。第 8 本直接运行 GPU 导航模型，各场景视频保存在 `outputs/lightnav_gpu/`。

```python
os.environ["MICRODUCK_RUN_BPU"] = "1"  # 开发板与策略服务就绪后启用
os.environ["MICRODUCK_COMPILE_BPU"] = "1"  # 按需编译 HBM 候选
```

BPU 默认关闭；开启后连接失败会报错，不会自动播放旧视频冒充新推理。

Ubuntu/Jupyter 的执行版保留单元内 MP4 播放。仓库发布副本同时保存 GIF 预览、MP4 相对链接与播放器，其它代码、文本与图像输出保持原样。完整 MP4 和运行报告位于 [recordings/](./recordings/README.md)；下载仓库后可在 Jupyter 中播放，GitHub 阅读时可通过各单元的 MP4 链接打开视频。

首次打开下载的 Notebook，Jupyter 可能暂不显示保存的 HTML 播放器。确认代码来源后，将 Notebook 设为受信任（Trust Notebook）再打开；也可直接点击 MP4 链接播放，不必重新训练或连接开发板。

本次 21 段 MP4 合计约 24.2 MiB，最大一段约 4 MiB；10 本发布版 Notebook 合计约 2.2 MiB。视频独立保存，不把 Base64 视频数据重复塞进 Notebook。文件校验值、体积与配套报告见 [视频清单](./recordings/manifest.json)。模型检查点、HBM、训练缓存与后续生成的 `outputs/` 不随资料提交。

`archive_inference_outputs.py --destination outputs/published_notebooks` 生成包含 `recordings/` 的发布副本，不覆盖正在使用的执行版。更新视频时只收录核对过的本轮结果，不自动把所有历史运行视频加入仓库。

## 模型与 BPU 注入

Notebook 不内置大 checkpoint。使用自己训练或下载的策略：

```powershell
$env:MICRODUCK_ONNX = "C:\models\microduck_policy.onnx"
$env:RDK_HOST = "192.168.8.128"
$env:RDK_BPU_HBM = "/home/sunrise/models/microduck_policy.hbm"
```

运动任务使用 CUDA/mjlab/ONNX 环境；Jupyter 可使用独立环境。必要时设置 `MICRODUCK_POLICY_PYTHON` 为运动环境的 Python。`MICRODUCK_MODEL_ROOT/converted/` 按任务选择模型，或用 `MICRODUCK_ONNX` 明确指定；扰动使用行走模型，爬梯使用 climber/get-up 模型对。

Notebook 选择对应任务 HBM，通过 RDK 的 `8766` 控制服务上传并切换 `8765` 策略服务。模型旁的 `.input_transform.json` 同步上传并与 HBM 校验值绑定。爬梯通过 `switch_pair` 同时启动 `8875` 的 climber 和 `8876` 的 get-up 服务；两者分别使用配套归一化配置与 INT16 HBM。

RDK 控制服务脚本为 `02-可运行代码/microduck-playground-stilts/scripts/rdk_bpu_policy_supervisor.py`，只允许切换本专题登记的 7 个任务模型；每次 Notebook 运行 BPU 视频单元都会覆盖对应的 `*_bpu_latest.mp4` 和 JSON 报告。

## 训练与部署

PPO/FastSAC 在工作站训练，BPU 负责推理。可选训练由环境变量启用，单元输出记录实际运行情况；smoke 验证采样、更新和导出，策略质量通过任务评测检查。`07_RDK网页与多策略_BPU验收.ipynb` 直接接入已有行走策略；浏览器扰动用于物理交互；第 8 本接入预训练 LightNav-0，不训练导航模型。

球平衡 Notebook 提供 FastSAC smoke 入口，默认从恢复的训练 checkpoint 继续 10 轮，也可用 `MICRODUCK_TRAIN_MODE=scratch` 从零开始。候选模型单独保存，默认闭环继续使用已验证的稳定模型。BPU 校准从实际仿真轨迹采样，再由 Ubuntu Docker 中的 `hb_mapper makertbin --model-type onnx` 编译 HBM。

本轮前 7 本重跑使用已训练策略，没有重复执行可选训练；篮球、球平衡与爬梯的 HBM 已在本轮另行完成校准编译，再由 Notebook 上传或切换并实测。第 9 本实际执行了 10 轮续训。校准候选和旧模型分别保留。

开发板侧需要匹配版本的 HBRT/BPU runtime 和对应任务 HBM；ONNX 到 HBM 的编译在配套的 Ubuntu Docker 工具链中完成，不要求板端安装编译器。真实重跑时还需要启动 RDK 服务并检查 HBM 与 ONNX 的观测维度、归一化、输入布局和动作后处理一致。

GPU 视频脚本与 `gpu_onnx_runtime.py` 在配套代码的 `scripts/` 中；爬梯使用 `bpu_official_ladder_pair_video.py` 和 desk-climb 环境，模型获取、GPU 运行和继续训练见[爬梯教程](../01-任务资料/06-MicroDuck梯面攀爬强化学习导读/README.md#6-爬梯与登桌复现步骤)。BPU 使用 RDK 服务和 `hbm_runtime`，运行前核对 HBM 的量化、输入布局、输出形状和动作后处理。

## 展示策略

Notebook 中保存的 GIF 用于网页和 GitHub 预览，配套 MP4/JSON 随 `recordings/` 一起提交。重跑生成的新视频保存到 `outputs/`，核对后再更新发布副本。网页物理交互仍可单独运行任务资料目录中的网页服务。
