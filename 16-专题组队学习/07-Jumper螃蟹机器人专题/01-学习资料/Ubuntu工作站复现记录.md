# Ubuntu 工作站复现记录

记录日期：2026-10-29。上游 Jumper 提交 `b2e58669848a3ba4f209597cd6bebdfe9d0533a4`；工作站 Python 3.12.3，NVIDIA RTX PRO 6000 Blackwell。实验选择 `native:cpu`，避免与启动时已占用 GPU 的工作负载争抢资源。

专题中的日志副本去除了终端颜色控制字符及行尾空格；工作站保留原始日志。

源码位于 `/data/Data14TB/robotics_shared/repos/jumper-everyembodied`，隔离环境位于 `/data/Data14TB/envs/jumper-everyembodied`。Python 依赖按上游 `pip install -e .` 安装，PyTorch 使用 CPU 轮子。工具链包含独立目录下的 CycloneDDS 11.0.1、wasm-bindgen 0.2.128，仓库的 Rockchip 头文件通过上游下载脚本校验。

上游 `gates.py` 的 A–F 六项已通过。首次 F 项提示找不到 `stdbool.h`；为检查进程增加 `BINDGEN_EXTRA_CLANG_ARGS=-I/usr/lib/gcc/x86_64-linux-gnu/13/include` 后通过。检查结果还包括 18 个注册任务、MuJoCo 3.11.0 和正确的本仓库 vendored 包路径。Warp 能枚举 GPU；这不表示本次 CPU 版 PyTorch 是 GPU 训练环境。

先执行上游 flat 任务的 3 次迭代 smoke，实际成功完成，使用 native CPU、64 个环境。模型组装日志提示场景采用父模型 timestep 0.002 和默认积分器；这是本次实际配置，比较时应保留该信息。可复现命令：

```bash
cd /data/Data14TB/robotics_shared/repos/jumper-everyembodied
PY=/data/Data14TB/envs/jumper-everyembodied/bin/python
OMP_NUM_THREADS=4 "$PY" scripts/train.py --task jumper.flat --backend native --device cpu --num_envs 64 --max-iterations 3 --headless --no-tensorboard
```

零动作基线以 `--agent zero` 在 CPU 模拟 250 步并使用 OSMesa 录制：[基线视频](../assets/zero-baseline.mp4)，时长 5.03 秒。它展示初始物理仿真，不代表训练策略。

```bash
MUJOCO_GL=osmesa OMP_NUM_THREADS=2 "$PY" scripts/play.py --task jumper.tripod --backend native --device cpu --agent zero --headless --steps 250 --video zero-baseline.mp4 --video-width 640 --video-height 480 --speed 0
```

## 三足训练与测量回放

训练 300 次迭代，64 个环境、4 物理线程，共 460,800 环境步，日志报告 4 分 25 秒；最终 mean reward -5.29、线速度跟踪奖励 0.3666，课程等级 0。日志：[tripod-training.txt](tripod-training.txt)。

```bash
OMP_NUM_THREADS=4 "$PY" scripts/train.py --task jumper.tripod --backend native --device cpu --num_envs 64 --cpu_threads 4 --max-iterations 300 --headless --no-tensorboard
```

checkpoint：工作站源码目录中的 `logs/jumper/jumper.tripod/2026-10-29_10-59-46/model_299.pt`，专题内备份为 [tripod-cpu-299.pt](../02-可运行代码/checkpoints/tripod-cpu-299.pt)；SHA-256：`fec2913fe14a7c45317f9820e6e40ba8b88f9e3a1bcb69cad90b63316ab7727f`。本次日志中的环境种子为 `None`，运行目录没有保存参数文件，因此不能据此重现完全相同的随机轨迹。

```bash
CKPT=logs/jumper/jumper.tripod/2026-10-29_10-59-46/model_299.pt
MUJOCO_GL=osmesa OMP_NUM_THREADS=2 "$PY" scripts/play.py --task jumper.tripod --backend native --device cpu --checkpoint "$CKPT" --headless --steps 500 --command 0.1 0 0 --measure --evaluation-out evaluation.json --replay-info-out replay-info.json --video tripod-299.mp4 --video-width 640 --video-height 480 --speed 0
```

开启观测噪声，固定前向指令 0.1 m/s，500 步、10 秒，0 次 episode reset；平均前向速度 -0.0013 m/s，前向平均绝对误差 0.1013 m/s。回放物理 1000 Hz、控制 50 Hz。该短训策略没有在此次测试中跟上前进指令，不应称为成功行走。视频：[tripod-299.mp4](../assets/tripod-299.mp4)；[评估数据](evaluation.json)、[回放配置](replay-info.json)、[关节测量](measure.csv)、[测量图](../assets/measure.png)。这是一次回放，不能推断多种子成功率。

## GPU 续训与诊断

2026-10-29 在 RTX PRO 6000 Blackwell 上安装 PyTorch 2.8.0+cu128、tensordict 0.10.0，`pip check` 和上游六项门槛均通过。训练从 CPU `model_299.pt` 恢复，用 Warp CUDA、4096 环境再训练 1000 次迭代，日志报告 11 分 29 秒及 98,304,000 环境步。最终课程等级 1、mean reward 28.79、线速度跟踪奖励 0.6038，单看奖励不能判断走路。完整日志为 [GPU 续训日志](tripod-gpu-training.txt)。

```bash
OMP_NUM_THREADS=4 "$PY" scripts/train.py --task jumper.tripod --backend warp --device cuda:0 --num_envs 4096 --checkpoint logs/jumper/jumper.tripod/2026-10-29_10-59-46/model_299.pt --max-iterations 1000 --headless --no-tensorboard
```

最终 checkpoint：[tripod-gpu-1298.pt](../02-可运行代码/checkpoints/tripod-gpu-1298.pt)，SHA-256 `8ee65fdfcabfb6f6e9a725762ff93897220e1e0fee69cd6cad1acbeb5f6a182a`；工作站原件在 `logs/jumper/jumper.tripod/2026-10-29_11-31-32/model_1298.pt`。`--max-iterations` 在恢复时表示新增迭代数。

用固定前进指令 0.3 m/s、500 步、10 秒、观测噪声开启，CPU 原生后端及训练相同的 200 Hz 物理频率回放：平均前向速度 -0.00080 m/s，前向平均绝对误差 0.30080 m/s，0 次 reset。此 checkpoint 仍未走起来。[视频](../assets/tripod-gpu-final.mp4)、[评估 JSON](gpu-final-evaluation.json)、[回放配置](gpu-final-replay-info.json)。

```bash
CKPT=logs/jumper/jumper.tripod/2026-10-29_11-31-32/model_1298.pt
MUJOCO_GL=osmesa OMP_NUM_THREADS=2 "$PY" scripts/play.py --task jumper.tripod --backend native --device cpu --checkpoint "$CKPT" --headless --steps 500 --physics-hz 200 --command 0.3 0 0 --measure --evaluation-out gpu-final-evaluation.json --replay-info-out gpu-final-replay-info.json --video tripod-gpu-final.mp4 --video-width 640 --video-height 480 --speed 0
```

排查中，`model_500` 在 GPU Warp 和 CPU 原生、200 Hz 和 1000 Hz 的回放都接近原地；相关 JSON 与 [固定指令检查脚本](../02-可运行代码/audit_fixed_command.py)保存在专题中。该脚本检查了命令管理器在第 0、49、99 步均接收 `[0.3, 0, 0]`（见 [日志](command-audit.txt)）；这排除了命令管理器未收到指令，但不代表所有观测环节已完全验证。上游代码也讨论了站立局部最优，是否就是本次原因还需对照实验。

## GPU 从零训练：三足策略开始行走

为排查 CPU checkpoint 续训的影响，使用相同上游任务配置重新随机初始化，并使用 Warp CUDA、4096 环境从零训练。`model_200.pt` 的 200 Hz 原生 CPU 回放在 0.3 m/s 固定前向指令、观测噪声开启时，10 秒平均前向速度 **0.2724 m/s**、0 次 reset；单独录视频的回放为 **0.2878 m/s**、0 次 reset。不同回放未固定种子，数值会有差别。以 1000 Hz 物理复测一次得到 **0.2793 m/s**、0 次 reset。数据分别见 [200 Hz 测试](fresh200-eval.json)、[视频回放测试](fresh200-video-evaluation.json)、[1000 Hz 测试](fresh200-cpu1000-03.json)。

这是本专题自训的三足策略；画面见 [自训三足行走](../assets/tripod-fresh200.mp4)，策略见 [tripod-gpu-fresh200.pt](../02-可运行代码/checkpoints/tripod-gpu-fresh200.pt)，SHA-256 `bbbe88e4087b2381de242f966c5582233bf5341ee44d8109b11ef26935bb9cfd`。

```bash
OMP_NUM_THREADS=4 "$PY" scripts/train.py --task jumper.tripod --backend warp --device cuda:0 --num_envs 4096 --max-iterations 1000 --headless --no-tensorboard
CKPT=logs/jumper/jumper.tripod/2026-10-29_11-43-57/model_200.pt
MUJOCO_GL=osmesa OMP_NUM_THREADS=2 "$PY" scripts/play.py --task jumper.tripod --backend native --device cpu --checkpoint "$CKPT" --headless --steps 500 --physics-hz 200 --command 0.3 0 0 --measure --evaluation-out fresh200-evaluation.json --replay-info-out fresh200-info.json --video tripod-fresh200.mp4 --video-width 640 --video-height 480 --speed 0
```

从零训练与续训的初始化、批量和随机状态不同；两者差异说明从零训练在本次测试中走起来，不能单凭一次对照确定续训失败的唯一原因，也不代表真机表现。

### 完整 1000 次训练与最终 checkpoint

从零训练正常完成 1000 次迭代，共 98,304,000 环境步；最终日志 mean reward 85.70、平均动作标准差 0.28。[完整日志](tripod-gpu-fresh.txt)。最终 checkpoint [tripod-gpu-fresh999.pt](../02-可运行代码/checkpoints/tripod-gpu-fresh999.pt) 的 SHA-256 为 `5d20b95dce5c0ba683b944618f109b3e00b8fe405701c6759b4f3b9458bbfe0d`。

以下都是 200 Hz 物理、50 Hz 控制、噪声开启、10 秒、单环境回放：

| 指令 vx (m/s) | 平均实际 vx (m/s) | vx 平均绝对误差 (m/s) | 重置 | 数据 |
| --- | --- | --- | --- | --- |
| 0.3 | 0.2780 | 0.0220 | 0 | [重复回放](fresh999-repeat.json) |
| 0.3 | 0.2759 | 0.0243 | 0 | [视频回放](fresh999-video.json) |
| 0.5 | 0.4593 | 0.0407 | 0 | [较快指令](fresh999-fast.json) |

视频：[最终 checkpoint 行走](../assets/tripod-fresh999.mp4)；[回放参数](fresh999-info.json)、[关节测量 CSV](fresh999-measure.csv)、[测量图](../assets/fresh999-measure.png)。两次 0.3 m/s 回放未固定种子，属于独立运行，不是覆盖不同任务与环境的成功率统计。复现时将上面的 `CKPT` 改为 `logs/jumper/jumper.tripod/2026-10-29_11-43-57/model_999.pt`，其他命令相同；测量 0.5 m/s 时替换 `--command 0.3 0 0`。
