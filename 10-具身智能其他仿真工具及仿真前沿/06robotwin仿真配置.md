# RoboTwin 仿真配置、任务运行与评估诊断

RoboTwin 是面向双臂操作的数据生成与策略评估基准。本章给出一条从环境检查、任务配置到结果诊断的最短可复现路径，重点说明相机视场、演示采集与闭环评估之间的关系。

- 项目仓库：[RoboTwin](https://github.com/TianxingChen/RoboTwin)
- 适用读者：已经具备 Python 环境和图形驱动，希望运行官方任务或接入自定义策略的学习者
- 本章产物：一次可重复的任务启动、一个带固定参数的评估命令，以及一份分阶段失败记录

## 1. 环境与目录

先将源码、资产和结果目录显式写入环境变量，避免命令依赖某台机器的个人路径。

```bash
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/RoboTwin}"
export ROBOTWIN_ASSETS="${ROBOTWIN_ASSETS:-$ROBOTWIN_ROOT/assets}"
export ROBOTWIN_OUTPUT="${ROBOTWIN_OUTPUT:-$ROBOTWIN_ROOT/outputs}"

cd "$ROBOTWIN_ROOT"
mkdir -p "$ROBOTWIN_OUTPUT"
```

按照官方说明安装依赖后，先验证核心包、图形设备和任务配置是否可读：

```bash
python - <<'PY'
import importlib

for name in ("numpy", "torch"):
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'unknown')}")
PY
```

若项目提供环境检查或任务枚举脚本，应先运行该脚本，再进入数据采集或评估。这样可以把依赖问题与策略问题分开定位。

## 2. 相机参数

任务配置中的 `fovy` 表示垂直视场角，单位通常为度。数值越大，相机覆盖的上下范围越宽，但目标在图像中的像素占比会下降。

| 相机示例 | `fovy` | 影响 |
| --- | ---: | --- |
| L515 | 45 | 视野较宽，适合观察完整工作区 |
| D435 | 37 | 目标占比更高，适合近距离操作 |

调整相机时应同时检查：

1. 双臂、夹爪和目标物体是否在初始帧中可见；
2. 抓取、抬升和放置阶段是否发生遮挡；
3. 训练数据与评估环境是否使用相同的相机名称、分辨率和视场角；
4. 图像预处理是否保留目标区域，而不是在裁剪时丢失关键接触点。

## 3. 任务运行顺序

不同版本的脚本名称可能变化，运行时以当前仓库的帮助信息为准。推荐按以下顺序建立最小闭环：

```bash
# 1. 查看入口和参数
python <task_entry.py> --help

# 2. 使用一个固定任务和固定随机种子做单回合检查
python <task_entry.py> \
  --task <task_name> \
  --seed 0 \
  --episodes 1 \
  --output_dir "$ROBOTWIN_OUTPUT/smoke"

# 3. 确认单回合可运行后，再扩大正式评估分母
python <task_entry.py> \
  --task <task_name> \
  --seed 0 \
  --episodes 50 \
  --output_dir "$ROBOTWIN_OUTPUT/eval_seed0"
```

这里的 `<task_entry.py>` 和 `<task_name>` 是占位参数，需要替换为当前版本仓库中的实际入口与任务名。运行前应先用 `--help` 核对当前版本接口。

## 4. 评估记录

仅记录最终成功率不足以解释双臂任务的失败位置。建议每回合保存以下字段：

```json
{
  "task": "task_name",
  "seed": 0,
  "episode": 0,
  "success": false,
  "stage": {
    "approach": true,
    "contact": true,
    "grasp": false,
    "lift": false,
    "place": false
  },
  "termination": "grasp_missed",
  "video": "episode_000.mp4"
}
```

正式评估至少同时保存配置文件、逐回合结果和视频索引。这样可以比较不同策略、随机种子和相机配置，并准确定位任务失败阶段。

## 5. 常见失败与定位方法

### 5.1 夹爪闭合但没有抓住物体

依次检查：

- 夹爪动作的开合方向和数值范围；
- 动作是否在正确的控制频率下保持足够时间；
- 末端执行器坐标系与策略输出坐标系是否一致；
- 碰撞体、摩擦系数和物体质量是否与数据生成环境一致。

### 5.2 机械臂在目标附近反复绕行

这通常与动作归一化、控制周期或观测时序有关。应同时记录原始动作、反归一化动作、关节目标和末端位姿，定位误差是来自策略输出还是控制器执行。

### 5.3 训练回放正常，闭环评估失败

优先检查训练与评估的相机顺序、状态向量顺序、语言指令、动作维度和统计量。任何一个字段错位，都可能让离线损失正常而闭环行为失效。

## 6. 完成标准

完成本章后，应能回答以下问题：

- 当前使用的任务入口、配置文件和随机种子是什么；
- 相机名称、分辨率和 `fovy` 是否与训练数据一致；
- 策略动作如何映射到双臂和夹爪控制接口；
- 失败发生在接近、接触、抓取、抬升还是放置阶段；
- 正式评估结果、视频和配置是否可以由同一条命令重新生成。
