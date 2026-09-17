# MicroDuck 直播 Notebook 学习线

这里把专题拆成 7 本 Notebook，每本都沿着同一条可讲解、可验收的路径：

1. 认识任务和 actor 的输入输出契约；
2. 使用对应任务的训练/导出入口，并检查观测归一化、动作顺序和 action clip；
3. 在本地用 ONNX Runtime 做数值 smoke test；
4. 把策略 ONNX 编译成 X5 HBM；
5. 让 RDK X5 BPU 逐步产生动作，由 Ubuntu MuJoCo 渲染并生成任务专属 MP4；
6. 通过 SSH 检查板端模型形状、服务日志和推理返回码。

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

## 启动

```powershell
cd 02-可运行代码\microduck-playground-stilts
uv sync
uv run --with jupyter jupyter lab ..\..\03-Notebook
```

在 Ubuntu 工作站上启动 Jupyter。Notebook 会自动向上查找 `02-可运行代码`；如果工作目录不是专题目录，设置 `MICRODUCK_TOPIC_ROOT`。

## 板端视频

视频已经生成在 Ubuntu 的 `03-Notebook/outputs/bpu_videos/`，每个 MP4 都有同名 JSON 报告：

| 文件 | 对应任务 |
| :-- | :-- |
| `walking_bpu_latest.mp4` | 导航/行走 |
| `perturbation_bpu_latest.mp4` | 浏览器物理扰动 |
| `stilt_bpu_latest.mp4` | 高跷行走 |
| `swing_bpu_latest.mp4` | 摆动旋转 |
| `ladder_bpu_latest.mp4` | 梯面攀爬 |
| `basketball_bpu_latest.mp4` | 篮球平衡 |
| `ball_balance_bpu_latest.mp4` | Motrix FastSAC 球平衡 |

这些不是参考 GIF 或 CPU 回放：Ubuntu 执行物理仿真和视频编码，RDK X5 通过 `hbm_runtime` 为每个控制步返回动作。重新运行某本 Notebook 的 BPU 视频单元会覆盖对应的 `*_bpu_latest.mp4` 和 JSON，不会创建新文件。

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

## 固定演示模型

非导航 Notebook 保留训练 smoke 入口：既有运动任务使用 PPO/RSL-RL 训练入口，球平衡使用 MotrixLab FastSAC。球平衡默认使用 64 个并行环境和小 replay buffer 做 5 个 smoke iteration；训练后导出最新 ONNX，再用 Ubuntu Docker 中的 `hb_mapper makertbin --model-type onnx` 编译最新 HBM。smoke 只验证链路，不代表重新训练后已经达到完整训练效果。

`07_RDK网页与多策略_BPU验收.ipynb` 按要求不启动训练，直接接入已有导航 ONNX 和 `walking.bin`。

当前已确认的开发板是 `sunrise@192.168.8.128`：Ubuntu 22.04.5、aarch64、BPU Platform 1.3.6、HBRT 3.15.55.0，板端有 `hrt_model_exec`、`hrt_bin_dump`、`hobot_dnn` 和 `/dev/bpu`。板端没有 `hb_mapper`，普通 `.onnx` 不能直接交给 BPU；ONNX 到 HBM 的编译由 Ubuntu Docker 中的 X5 工具链完成。

任务视频使用专题中的 RDK BPU 服务和 `hbm_runtime`，不使用 `CPUExecutionProvider` 产生动作。不要把 ONNX 直接改名成 HBM；必须实际编译，并核对量化、输入布局、输出形状和动作后处理。

## 展示策略

直播默认在 Notebook 中直接展示 `outputs/bpu_videos/` 的真实 MP4。网页交互仍可单独运行任务资料目录里的网页服务；视频展示本身不依赖浏览器 WebAssembly 或本机 GPU。`outputs/` 已被忽略，不会把视频重新提交进仓库。
