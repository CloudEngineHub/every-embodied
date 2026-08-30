# ManiSkill 与 LeRobot 的仿真到真实数据链路

本章搭建 `lerobot-sim2real` 的最小运行环境，并用 ManiSkill 随机动作示例检查仿真、渲染和 Python 依赖。这个检查确认环境能够创建并执行动作；真实机器人迁移还需要相机标定、动作空间对齐、数据归一化和安全限幅。

## 1. 环境与目录

```bash
export WORKSPACE=/path/to/workspace
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

mamba create -n lerobot-sim2real python=3.11 -y
mamba activate lerobot-sim2real
git clone https://github.com/StoneT2000/lerobot-sim2real.git
```

项目依赖一个较早的 LeRobot 代码接口。为避免导入路径变化，先使用项目验证过的提交：

```bash
cd "$WORKSPACE"
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout a989c795587d122299275c65a38ffdd0a804b8dc
pip install -e .

cd "$WORKSPACE/lerobot-sim2real"
pip install -e .
```

PyTorch 应根据机器的 GPU 驱动和官方安装矩阵单独安装，不要直接复制与本机不匹配的 CUDA 版本。

## 2. Vulkan 渲染检查

无显示服务器运行时，ManiSkill 依赖 Vulkan 完成离屏渲染。先确认系统能够识别 Vulkan 设备：

```bash
vulkaninfo --summary
```

如果命令不存在，安装系统对应的 Vulkan 工具包；如果输出中没有目标 GPU，先修复驱动和设备权限，再运行仿真。

## 3. 运行最小示例

```bash
mamba activate lerobot-sim2real
cd "$WORKSPACE/lerobot-sim2real"
python -m mani_skill.examples.demo_random_action
```

程序能够创建环境、连续执行动作且不出现渲染异常，即通过最小链路检查。进一步的数据采集、训练和部署命令以[上游项目说明](https://github.com/StoneT2000/lerobot-sim2real)为准。

## 4. 迁移检查点

- 仿真与真机的观测字段、图像尺寸和时间戳一致。
- 策略输出与真实机器人控制接口采用同一动作定义和单位。
- 数据归一化统计只由训练集计算，并在部署时复用。
- 真机运行前配置关节、速度、力矩和工作空间限制。
