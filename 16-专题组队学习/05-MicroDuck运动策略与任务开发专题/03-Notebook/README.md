# MicroDuck 直播 Notebook 学习线

本专题有 7 本按任务区分的 Notebook。视频顺序是 GPU 推理在前、RDK X5 BPU 推理在后；每本都有独立任务标签和独立回放，不会用同一段视频冒充不同任务。

Notebook 保存的是已执行结果快照，适合在 GitHub 上先阅读输出，再在 Ubuntu/Jupyter 环境按需重跑。发布时检查了 64 个代码单元：全部有执行计数和输出，没有存档的 Python 异常。该检查不等于在读者机器上重新执行，也不代表每个训练单元都实际启动了新训练。

## Notebook 清单

| Notebook | 主线 |
| :-- | :-- |
| `01_篮球平衡_PPO_ONNX_BPU_MuJoCo.ipynb` | 速度命令、球场景、PPO、策略导出 |
| `02_浏览器物理扰动_回放与接口.ipynb` | MuJoCo 状态、外力扰动、网页回放 |
| `03_高跷行走_课程与ONNX_BPU.ipynb` | BAM、形态课程、动作平滑、部署契约 |
| `04_摆动旋转_ONNX_BPU_MuJoCo.ipynb` | 摆动任务和仓库内置 ONNX 示例 |
| `05_球平衡_FastSAC_ONNX_BPU.ipynb` | FastSAC 与另一套任务接口的对照 |
| `06_梯面攀爬_接触与部署模板.ipynb` | 接触、落脚目标、课程和未完成项边界 |
| `07_RDK网页与多策略_BPU验收.ipynb` | 导航直接接入、RDK TCP、多策略切换、BPU 验收 |

每本 Notebook 的 GPU 预览和历史 BPU 预览见[按任务划分的演示总览](../01-任务资料/精选视频/README.md)。行走/导航按要求直接接入已有策略，不训练；扰动 Notebook 是物理回放与接口验证；梯面 Notebook 当前用于接触任务检查，不能据此宣称已稳定爬梯。其它 Notebook 提供对应算法的训练/续训入口，但烟雾训练、可选训练是否实际运行，以该 Notebook 保存的单元输出为准。

## 启动

```powershell
cd 02-可运行代码\microduck-playground-stilts
uv sync
uv run --with jupyter jupyter lab ..\..\03-Notebook
```

在 Ubuntu 工作站上启动 Jupyter。Notebook 会自动向上查找 `02-可运行代码`；如果工作目录不是专题目录，设置 `MICRODUCK_TOPIC_ROOT`。

## GPU 与 BPU 视频

每本 Notebook 都先输出对应任务的 GPU 回放，再输出此前保存的 BPU-in-the-loop 回放。BPU 动作由 X5 HBM 推理产生，工作站负责 MuJoCo 物理仿真和编码；但仓库中的 GIF 是先前运行保存的缓存结果，不表示打开 GitHub 时板端正在线运行。重新执行 BPU 单元会在工作站 `03-Notebook/outputs/bpu_videos/` 生成该任务 MP4 和 JSON 报告，并覆盖该任务的 `*_bpu_latest` 文件。

为了控制仓库体积，Notebook 内的视频输出改成 GIF 相对链接，原始 MP4 不嵌入 ipynb，也不提交。7 项任务的 GPU/BPU GIF 都在演示总览页；Notebook 其余代码与文本结果快照仍保留。

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

这里的 MobileNet 只用于验证“工作站能连接开发板、开发板能调用 BPU”；它不是 MicroDuck 策略模型。真正运行任务时，Notebook 会按任务选择对应 HBM，并通过 RDK 上预登记的 `8766` 控制服务切换 `8765` 策略服务，不依赖 Ubuntu 到 RDK 的 SSH 密钥。不能把上面的 MobileNet `.bin` 当作任务策略 HBM。

RDK 控制服务脚本为 `02-可运行代码/microduck-playground-stilts/scripts/rdk_bpu_policy_supervisor.py`，只允许切换本专题登记的 7 个任务模型；每次 Notebook 运行 BPU 视频单元都会覆盖对应的 `*_bpu_latest.mp4` 和 JSON 报告。

## 训练与部署边界

PPO/FastSAC 的训练在工作站进行，BPU 只负责推理。部分 Notebook 的训练单元由环境变量控制，默认可跳过；保存了输出不等于训练必然执行。烟雾训练只验证采样、更新和导出链路，不代表策略已收敛。`07_RDK网页与多策略_BPU验收.ipynb` 按要求不启动训练，直接接入已有行走/导航策略；浏览器扰动是回放任务，不训练新的恢复策略。

球平衡 Notebook 提供 FastSAC smoke 入口；如启用，默认小规模运行后可导出 ONNX，再用 Ubuntu Docker 中的 `hb_mapper makertbin --model-type onnx` 编译 HBM。X5 上不需要安装 `hb_mapper`；普通 ONNX 也不能直接当 HBM 执行。

开发板侧需要匹配版本的 HBRT/BPU runtime 和对应任务 HBM；ONNX 到 HBM 的编译在配套的 Ubuntu Docker 工具链中完成，不要求板端安装编译器。真实重跑时还需要启动 RDK 服务并检查 HBM 与 ONNX 的观测维度、归一化、输入布局和动作后处理一致。

任务视频使用专题中的 RDK BPU 服务和 `hbm_runtime`，不使用 `CPUExecutionProvider` 产生动作。不要把 ONNX 直接改名成 HBM；必须实际编译，并核对量化、输入布局、输出形状和动作后处理。

## 展示策略

Notebook 中保存的 GIF 用于网页和 GitHub 预览；重跑生成的 MP4/JSON 留在 Ubuntu `outputs/`，不提交进仓库。网页物理交互仍可单独运行任务资料目录中的网页服务。
