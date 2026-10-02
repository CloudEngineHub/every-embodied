# MicroDuck 直播 Notebook 学习线

本专题有 8 本 Notebook，对应 8 天学习。前 7 本的运动任务视频顺序是 GPU 推理在前、RDK X5 BPU 推理在后；第 8 本 LightNav-0 多场景导航只使用工作站 GPU，不连接开发板。

组队学习按 [8 天学习计划](../学习计划.md)推进，每天对应下方一本 Notebook，围绕任务学习、实验与交流。

Notebook 保存了单元输出，适合在 GitHub 上先阅读结果，再在 Ubuntu/Jupyter 环境按需重跑。前 7 本保存了 64 个代码单元的执行计数与输出；第 8 本新增的 12 个代码单元已在 Ubuntu 从头运行并保存输出，合计 76 个代码单元，没有存档的 Python 异常。可选训练是否执行，见对应单元的输出。

## Notebook 清单

| Notebook | 主线 |
| :-- | :-- |
| `01_篮球平衡_PPO_ONNX_BPU_MuJoCo.ipynb` | 速度命令、球场景、PPO、策略导出 |
| `02_浏览器物理扰动_回放与接口.ipynb` | MuJoCo 状态、外力扰动、网页回放 |
| `03_高跷行走_课程与ONNX_BPU.ipynb` | BAM、形态课程、动作平滑、部署契约 |
| `04_摆动旋转_ONNX_BPU_MuJoCo.ipynb` | 摆动任务和仓库内置 ONNX 示例 |
| `05_球平衡_FastSAC_ONNX_BPU.ipynb` | FastSAC 与另一套任务接口的对照 |
| `06_梯面攀爬_接触与部署模板.ipynb` | 爬梯、登桌、起身双策略切换和课程训练 |
| `07_RDK网页与多策略_BPU验收.ipynb` | 行走策略直接接入、RDK TCP、多策略切换、BPU 验收 |
| `08_LightNav0_视觉语言导航_GPU_多场景.ipynb` | 预训练模型、图像与语言、MPC、住宅/客厅/走廊 |

## 第 8 天：视觉语言导航

[08_LightNav0_视觉语言导航_GPU_多场景.ipynb](./08_LightNav0_视觉语言导航_GPU_多场景.ipynb) 使用预训练 LightNav-0，不训练导航模型。
在 GPU［图形处理器］上运行导航模型，连接已有 MicroDuck 行走策略，分别体验官方住宅、自建客厅和自建走廊。
每个场景生成独立视频、模型预测记录与轨迹图。环境准备和默认路径见[配套说明](../02-可运行代码/lightnav-learning/README.md)。
按学习计划在第 8 天运行，结合视频理解导航模型与行走策略如何协作。

导航 Notebook 已在 Ubuntu 从头执行：12 个代码单元全部保留输出，包含三个场景的第一视角、实际预测、视频和轨迹图。
视频由本次模型推理生成，不嵌入大文件；运行后在对应单元内直接播放。

每本 Notebook 的视频见[演示总览](../01-任务资料/精选视频/README.md)。行走/导航直接接入已有策略；扰动用于物理交互和接口验证；梯面 GPU 视频已完成逐级爬梯、登桌与站稳，BPU 视频记录梯面接触。其它任务提供对应算法的训练/续训入口，实际运行情况见保存的单元输出。

## 启动

```bash
cd 02-可运行代码/microduck-playground-stilts
uv sync
uv run --with jupyter jupyter lab ../../03-Notebook
```

在 Ubuntu 工作站上启动 Jupyter。Notebook 会自动向上查找 `02-可运行代码`；如果工作目录不是专题目录，设置 `MICRODUCK_TOPIC_ROOT`。

## GPU 与 BPU 视频

前 7 本 Notebook 先展示 GPU 推理视频，再展示 BPU 推理视频。BPU 计算动作，工作站负责物理仿真和编码。重新执行 BPU 单元会在工作站 `03-Notebook/outputs/bpu_videos/` 生成该任务 MP4 和 JSON 报告，并覆盖该任务的 `*_bpu_latest` 文件。第 8 本直接运行 GPU 导航模型，各场景视频保存在 `outputs/lightnav_gpu/`。

为了控制仓库体积，前 7 本 Notebook 内的视频输出使用 GIF 相对链接，原始 MP4 不嵌入 ipynb，也不提交。7 项运动任务的 GPU/BPU GIF 和 3 个导航场景的 GIF 都在演示总览页；Notebook 其余代码、文本与图像输出仍保留。导航 Notebook 重跑后可在单元内播放新生成的 MP4。

## 模型与 BPU 注入

Notebook 不内置大 checkpoint。使用自己训练或下载的策略：

```powershell
$env:MICRODUCK_ONNX = "C:\models\microduck_policy.onnx"
$env:RDK_HOST = "192.168.8.128"
$env:RDK_BPU_HBM = "/home/sunrise/models/microduck_policy.hbm"
```

如果只想检查板端运行时，可以运行已经验证的 X5 MobileNet BPU 样例：

```python
RDK_BPU_SMOKE_MODEL = "/opt/tros/humble/lib/dnn_benchmark_example/config/X5/mobilenetv1_224x224_nv12.bin"
RDK_BPU_SMOKE_INPUT_BYTES = 75264
```

MobileNet 用于检查连通和 BPU 运行时。运行 MicroDuck 任务时，Notebook 选择对应任务 HBM，通过 RDK 的 `8766` 控制服务切换 `8765` 策略服务。

RDK 控制服务脚本为 `02-可运行代码/microduck-playground-stilts/scripts/rdk_bpu_policy_supervisor.py`，只允许切换本专题登记的 7 个任务模型；每次 Notebook 运行 BPU 视频单元都会覆盖对应的 `*_bpu_latest.mp4` 和 JSON 报告。

## 训练与部署

PPO/FastSAC 在工作站训练，BPU 负责推理。可选训练由环境变量启用，单元输出记录实际运行情况；smoke 验证采样、更新和导出，策略质量通过任务评测检查。`07_RDK网页与多策略_BPU验收.ipynb` 直接接入已有行走策略；浏览器扰动用于物理交互；第 8 本接入预训练 LightNav-0，不训练导航模型。

球平衡 Notebook 提供 FastSAC smoke 入口；启用后先小规模训练、导出 ONNX，再用 Ubuntu Docker 中的 `hb_mapper makertbin --model-type onnx` 编译 HBM，上传 X5 进行推理。

开发板侧需要匹配版本的 HBRT/BPU runtime 和对应任务 HBM；ONNX 到 HBM 的编译在配套的 Ubuntu Docker 工具链中完成，不要求板端安装编译器。真实重跑时还需要启动 RDK 服务并检查 HBM 与 ONNX 的观测维度、归一化、输入布局和动作后处理一致。

GPU 视频脚本与 `gpu_onnx_runtime.py` 在配套代码的 `scripts/` 中；爬梯使用 `bpu_official_ladder_pair_video.py` 和 desk-climb 环境，模型获取、GPU 运行和继续训练见[爬梯教程](../01-任务资料/06-MicroDuck梯面攀爬强化学习导读/README.md#6-爬梯与登桌复现步骤)。BPU 使用 RDK 服务和 `hbm_runtime`，运行前核对 HBM 的量化、输入布局、输出形状和动作后处理。

## 展示策略

Notebook 中保存的 GIF 用于网页和 GitHub 预览；重跑生成的 MP4/JSON 留在 Ubuntu `outputs/`，不提交进仓库。网页物理交互仍可单独运行任务资料目录中的网页服务。
