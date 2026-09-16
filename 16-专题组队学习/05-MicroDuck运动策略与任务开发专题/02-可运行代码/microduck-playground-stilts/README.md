# MicroDuck 可运行代码包

这是本专题配套的 MicroDuck 运行代码快照，来源于本地
`microduck-playground-stilts`。它保留了 `mjlab` 任务、MicroDuck 机器人模型、BAM 执行器、训练脚本、评测脚本和配置测试，适合在独立 Python 环境中运行。

## 快速开始

要求：Python 3.12、CUDA GPU、`uv`，以及能够安装 `mjlab==1.3.0` 和
`warp-lang==1.12.0` 的网络环境。进入当前目录后执行：

```powershell
uv sync

# 先做低成本训练 smoke，不要一开始就使用完整并行规模
uv run train Mjlab-SwingPump-MicroDuck `
  --env.scene.num-envs 64 `
  --agent.max-iterations 5
```

如果只想检查配置和物理不变量，可以先运行：

```powershell
uv run --with pytest pytest tests -q
```

训练、评测、导出和视频回放脚本集中在 `scripts/`。梯面任务优先阅读
`docs/stilt_training_plan.md` 和任务资料中的梯面 README，再执行
`scripts/prepare_ladder_footstep_bootstrap.py`、
`scripts/evaluate_ladder_checkpoint.py` 等脚本。

## 代码导航

| 目录或文件 | 作用 |
| --- | --- |
| `src/mjlab_microduck/robot/` | MJCF 机器人模型、关节参数、形态配置和硬件常量 |
| `src/mjlab_microduck/actuator/` | BAM 执行器和摩擦模型 |
| `src/mjlab_microduck/tasks/` | 行走、球踢、拾取、摆动、高跷、梯面等任务 |
| `scripts/export.py` | checkpoint 到 ONNX 和单环境视频导出 |
| `scripts/infer_policy.py` | 策略推理和动作回放 |
| `scripts/evaluate_*.py` | 针对不同任务的评测入口 |
| `tests/` | 观测维度、任务注册、接触、课程和导出契约检查 |
| `hardware/stilts/generate_stilts.py` | 参数化生成高跷几何 |

## 重要说明

1. `logs/`、`.venv/`、checkpoint、ONNX 和视频输出没有随代码包提交；运行结果应保存在本地实验目录，并在专题 README 或实验记录中登记路径。
2. 训练任务和评测任务必须使用匹配的观测维度、槽位顺序、归一化和控制频率。旧 checkpoint 的 51 维输入不能直接接到 61 维接口。
3. 当前仓库的多数任务配置是独立的 Actor/Critic MLP，常见结构为 `512 → 256 → 128`；不要仅根据教学 PPT 把当前代码描述成已经启用 LSTM。
4. 梯面任务的短训练和 warm-start 只能证明环境、接触和训练入口可用，不能替代连续换档和顶部落台的跨 seed 成功率。

在 Windows 中文区域设置下，部分旧测试直接读取 UTF-8 的 `pyproject.toml` 时可能受系统默认编码影响；可先设置 `PYTHONUTF8=1`。梯面 footstep 任务的当前入口配置使用 `0.14 m/s`，测试已与该配置对齐。

## 许可证

软件和硬件许可证、第三方声明随本代码包保留在根目录。复制或发布
MicroDuck 模型、素材和实验补丁时，应同时保留原始署名和相应许可证说明。
