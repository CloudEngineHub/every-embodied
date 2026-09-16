# MicroDuck 直播 Notebook 学习线

这里把专题拆成 7 本 Notebook，每本都沿着同一条可讲解、可验收的路径：

1. 认识任务和 actor 的输入输出契约；
2. 用现有训练入口导出 ONNX，并检查观测归一化、动作顺序和 action clip；
3. 在本地用 ONNX Runtime 做数值 smoke test；
4. 用短视频或关键帧展示 MuJoCo 回放，必要时再启动网页交互；
5. 通过 SSH 探测 RDK X5 工具，并在具备 HBM 与板端命令后验收 BPU 推理。

## Notebook 清单

| Notebook | 主线 |
| :-- | :-- |
| `01_篮球平衡_PPO_ONNX_BPU_MuJoCo.ipynb` | 速度命令、球场景、PPO、策略导出 |
| `02_浏览器物理扰动_回放与接口.ipynb` | MuJoCo 状态、外力扰动、网页回放 |
| `03_高跷行走_课程与ONNX_BPU.ipynb` | BAM、形态课程、动作平滑、部署契约 |
| `04_摆动旋转_ONNX_BPU_MuJoCo.ipynb` | 摆动任务和仓库内置 ONNX 示例 |
| `05_球平衡_FastSAC_ONNX_BPU.ipynb` | FastSAC 与另一套任务接口的对照 |
| `06_梯面攀爬_接触与部署模板.ipynb` | 接触、落脚目标、课程和未完成项边界 |
| `07_RDK网页与多策略_BPU验收.ipynb` | RDK TCP、多策略切换、BPU 验收 |

## 启动

```powershell
cd 02-可运行代码\microduck-playground-stilts
uv sync
uv run --with jupyter jupyter lab ..\..\03-Notebook
```

从专题根目录启动也可以；Notebook 会自动向上查找 `02-可运行代码`。如果工作目录不是专题目录，设置 `MICRODUCK_TOPIC_ROOT`。

## 模型与 BPU 注入

Notebook 不内置大 checkpoint。使用自己训练或下载的策略：

```powershell
$env:MICRODUCK_ONNX = "C:\models\microduck_policy.onnx"
$env:RDK_HOST = "192.168.8.128"
$env:RDK_BPU_HBM = "/home/sunrise/models/microduck_policy.hbm"
```

如果只是想先直接运行 Notebook，保持 `RDK_BPU_HBM` 为空即可。Notebook 会使用这个已经在 X5 上验证过的默认 BPU 样例：

```python
RDK_BPU_SMOKE_MODEL = "/opt/tros/humble/lib/dnn_benchmark_example/config/X5/mobilenetv1_224x224_nv12.bin"
RDK_BPU_SMOKE_INPUT_BYTES = 75264
```

这里的 MobileNet 只用于验证“工作站能通过 SSH 调到开发板、开发板能调用 BPU”；它不是 MicroDuck 策略模型。真正运行策略时，才填写与 `[1, 61] float32 -> [1, 14]` 接口匹配的 X5 `.hbm` 文件，例如 `RDK_BPU_HBM = "/home/sunrise/models/microduck_policy.hbm"`。不能把上面的 MobileNet `.bin` 路径填进 `RDK_BPU_HBM`，因为两者输入格式不同。

## 固定演示模型

Notebook 在 Ubuntu 工作站上默认从已验证的行走策略恢复：

```python
MICRODUCK_WORKSPACE = "/home/ubuntu/workspaces/microduck_rl"
MICRODUCK_TRAIN_RUN = "2026-09-03_20-05-56_every-embodied-4096x6000"
MICRODUCK_TRAIN_CHECKPOINT = "model_5999.pt"
MICRODUCK_TRAIN_ENVS = "64"
MICRODUCK_TRAIN_ITERATIONS = "10"
```

每次运行会继续训练 10 个 PPO iteration，并覆盖 Notebook 目录下的 `outputs/microduck_demo_latest.pt` 与 `outputs/microduck_demo_latest.onnx`。原始 checkpoint 不会覆盖；训练日志仍按时间写入 `logs/rsl_rl/velocity/`，便于回查。10 个 iteration 只用于验证训练链路，不代表策略能力已经重新收敛。

当前已确认的开发板是 `sunrise@192.168.8.128`：Ubuntu 22.04.5、aarch64、BPU Platform 1.3.6、HBRT 3.15.55.0，板端有 `hrt_model_exec`、`hrt_bin_dump`、`hobot_dnn` 和 `/dev/bpu`。板端没有 `hb_mapper`，普通 `.onnx` 不能直接交给 BPU；ONNX 到 HBM 的编译需要另行准备与 X5 SDK 匹配的编译环境。未设置 `RDK_BPU_HBM` 时，Notebook 会运行板端已验证的 X5 MobileNet BPU 样例；设置后再上传 `[1,61]` 的 float32 零观测并调用 `hrt_model_exec infer`，只有退出码为 0 才标记 `PASS-rdk-bpu`。

当前仓库原有 RDK 服务器使用 `CPUExecutionProvider`，因此 Notebook 会把 CPU 结果和 BPU 结果分开标记。不要把板端已有的 `.onnx` 文件直接改名成 `.hbm`；必须实际生成 HBM，并核对量化、输入布局、输出形状和动作后处理。

## 展示策略

直播默认展示短视频或关键帧，避免网页 WebAssembly、浏览器 GPU 和 OpenGL 窗口打断讲解。要做交互演示时，运行任务资料目录里的网页服务；要做闭环回放时，在 Notebook 中执行对应的 MuJoCo 命令并把产物写入 `outputs/`。`outputs/` 已被忽略，不会把视频重新提交进仓库。
