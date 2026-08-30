# OmniGibson 家庭场景仿真快速入门

OmniGibson 是基于 Isaac Sim 的机器人仿真环境，也是 BEHAVIOR-1K 家庭活动基准使用的仿真平台。本节介绍其配置结构和最小环境创建流程，安装命令与资产授权以[官方安装文档](https://stanfordvl.github.io/BEHAVIOR-1K/omnigibson/getting_started/installation.html)和[官方仓库](https://github.com/StanfordVL/BEHAVIOR-1K)为准。

## 学习目标

- 理解场景、对象、机器人和任务四类配置。
- 创建一个包含 Fetch 机器人和基础对象的环境。
- 完成环境步进并正确关闭仿真器。
- 知道何时需要 BEHAVIOR-1K 场景与对象资产。

## 前置条件

OmniGibson 依赖 Isaac Sim，对操作系统、显卡驱动和图形运行环境有明确要求。开始前应先在官方安装页核对当前版本要求，并为项目创建独立环境。模型、场景和对象资产应放在源码仓库之外。

```bash
export PROJECT_ROOT=$HOME/projects/BEHAVIOR-1K
export DATA_ROOT=$HOME/datasets/behavior-1k

git clone https://github.com/StanfordVL/BEHAVIOR-1K.git "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
```

随后按照官方安装脚本完成 OmniGibson 与资产安装。安装完成后先验证导入：

```bash
python -c "import omnigibson as og; print(og.__file__)"
```

命令应输出当前环境中 `omnigibson` 包的路径。

## 配置结构

OmniGibson 通过一个配置字典组合环境：

```python
import omnigibson as og

cfg = {
    "scene": {
        "type": "Scene",
        "floor_plane_visible": True,
    },
    "objects": [
        {
            "type": "PrimitiveObject",
            "name": "target_box",
            "primitive_type": "Cube",
            "rgba": [0.2, 0.7, 1.0, 1.0],
            "scale": [0.25, 0.25, 0.1],
            "fixed_base": True,
            "position": [1.0, 0.0, 0.1],
        }
    ],
    "robots": [
        {
            "type": "Fetch",
            "name": "fetch",
            "obs_modalities": ["rgb", "depth"],
        }
    ],
    "task": {
        "type": "DummyTask",
        "termination_config": {},
        "reward_config": {},
    },
}
```

- `scene` 定义空间和场景资产。
- `objects` 定义可交互对象、初始位姿和物理属性。
- `robots` 定义机器人、传感器模态和控制接口。
- `task` 定义奖励、终止条件和任务逻辑。

## 创建与步进环境

将上面的配置保存为 `examples/omnigibson_minimal.py`，再追加：

```python
env = og.Environment(cfg)

for _ in range(240):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

og.shutdown()
```

运行：

```bash
cd "$PROJECT_ROOT"
python examples/omnigibson_minimal.py
```

环境应加载地面、对象和 Fetch 机器人，并连续执行环境步进。随机动作只用于检查观察、动作和物理步进接口，不代表任务策略。

## 使用数据集对象

官方快速入门中的 `DatasetObject` 会根据 `category` 和 `model` 从 BEHAVIOR-1K 资产中查找对象。使用这类对象前必须完成相应资产安装；自定义 USD 文件则通过 `USDObject` 和 `usd_path` 加载。

## 检查点

1. `import omnigibson` 成功。
2. 环境创建时没有缺失场景或对象资产。
3. `env.action_space.sample()` 的维度与机器人动作空间一致。
4. `env.step()` 返回观察、奖励、终止标记和信息字典。
5. 脚本退出前调用 `og.shutdown()` 释放仿真资源。

## 常见问题

### 找不到对象或场景

检查资产是否安装、配置中的类别与模型名是否存在，以及运行环境是否能够读取资产目录。

### 无图形界面服务器无法启动

先按 Isaac Sim 官方文档验证无图形界面运行，再运行 OmniGibson。不要把图形驱动故障与任务配置故障混在同一次排查中。

### 环境能运行但任务没有成功

本节使用随机动作验证接口。正式任务需要选择控制器或策略，并使用对应任务的终止条件和评价协议。

## 参考资料

- [OmniGibson 官方安装文档](https://stanfordvl.github.io/BEHAVIOR-1K/omnigibson/getting_started/installation.html)
- [OmniGibson 官方快速入门](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/docs/getting_started/quickstart.md)
- [BEHAVIOR-1K 官方仓库](https://github.com/StanfordVL/BEHAVIOR-1K)
