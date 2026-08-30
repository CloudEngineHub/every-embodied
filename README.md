<div align="center">
  <img src="assets/main.png" width="100%" alt="Every Embodied 具身智能开源教材封面" />

  # Every Embodied：具身智能开源教材

  从机器人学基础、感知与控制，逐步学习仿真、数据、策略训练、评估与真实系统集成。

  [在线阅读](https://datawhalechina.github.io/every-embodied/zh-cn/) · [教材学习地图](./教材学习地图.md) · [最小运行示例](./examples/README.md) · [English](./README.en.md)
</div>

## 教材定位

Every Embodied 面向希望系统学习具身智能的开发者、学生和研究人员。教材以“概念—实现—验证”为主线：先建立机器人学和感知基础，再进入策略学习、导航、仿真与评估，最后通过完整项目理解前沿方法如何落到可运行系统中。

本仓库包含四类内容：

- **核心教材**：机器人学、视觉、学习算法、操作、导航、数据与评估。
- **实验手册**：硬件接入、仿真环境、数据采集、训练、评估和部署。
- **项目案例**：竞赛复现、专题学习、世界模型、无人机和机器人设计。
- **社区档案**：组队学习路径、活动材料与内容发布记录。

完整的章节依赖关系、推荐顺序和分方向路线见[教材学习地图](./教材学习地图.md)。

## 快速开始

下面的最小示例使用 MuJoCo 创建机械臂场景并执行抓取轨迹，适合检查 Python、图形窗口和物理引擎是否可以正常工作。

```bash
git clone --depth 1 https://github.com/datawhalechina/every-embodied.git
cd every-embodied

conda create -n embodied python=3.10 -y
conda activate embodied
pip install mujoco ruckig

python examples/01_hello_every_embodied_mujoco.py
```

运行后应看到机械臂完成接近、抓取、抬升和放置动作。命令说明与无图形界面运行方式见[最小运行示例](./examples/README.md)。

## 四卷结构

| 卷册 | 学习目标 | 对应专题 |
| --- | --- | --- |
| 第一卷：具身智能基础 | 建立具身智能、坐标变换、运动学、控制、视觉和三维感知基础 | [01](./01-具身智能概述/README.md)、[02](./02-机器人基础和控制、手眼协调/README.md)、[04](./04-具身场景的计算机视觉、3D重建/README.md) |
| 第二卷：学习与决策 | 掌握深度学习、强化学习、机器人操作、VLA、导航、数据与评估 | [05](./05-具身场景的深度和强化学习/README.md)、[06](./06-策略抓取或抓取VLA/README.md)、[07](./07-机器人操作、运动控制/README.md)、[08](./08-具身导航及VLN/README.md)、[09](./09-具身智能数据及评估基准benchmark/README.md) |
| 第三卷：系统与仿真 | 完成硬件接入、仿真配置、数据采集、训练评估与工程工具链 | [03](./03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/README.md)、[10](./10-具身智能其他仿真工具及仿真前沿/README.md)、[11](./11-其他辅助工具/README.md)、[21](./21-机械臂和机器人设计/README.md) |
| 第四卷：前沿与项目 | 通过竞赛、专题、世界模型和无人机项目完成综合实践 | [13](./13-其他前沿项目复现/README.md)、[15](./15-Challenge竞赛/README.md)、[16](./16-专题组队学习/README.md)、[17](./17-具身世界模型/README.md)、[18](./18-无人机专题/README.md) |

面试题与参考资料分别收录在[第 12 专题](./12-具身智能面试问题汇总/README.md)和[第 14 专题](./14-相关参考/README.md)。组队学习与内容发布记录位于[第 19 专题](./19-Datawhale每月组队学习/README.md)和[第 20 专题](./20-公众号短文宣发/README.md)。

## 推荐学习路线

### 零基础路线

`01 具身智能概述 → 02 机器人学基础 → examples 最小示例 → 04 视觉感知 → 07 机器人操作 → 10 仿真工具`

完成后，学习者应能解释机器人观测、状态、动作和控制回路，并在仿真器中运行一个可观察、可调试的操作任务。

### 策略学习路线

`02 机器人学基础 → 05 深度与强化学习 → 06 VLA 策略 → 09 数据与评估 → 16 专题实验`

这条路线覆盖数据格式、模仿学习、动作分块、训练过程、闭环评估和结果分析。

### 导航与移动操作路线

`02 坐标与控制 → 04 感知与建图 → 08 导航与 VLN → 09 评估基准 → 10 仿真环境`

这条路线从定位、地图和路径规划出发，进一步学习语言条件导航与移动操作系统。

### 系统工程路线

`03 硬件与 LeRobot → 10 仿真工具 → 11 工程工具 → 21 机器人设计 → 15/16 综合项目`

这条路线适合需要搭建真实设备、组织数据采集或迁移完整开源项目的读者。

## 代表性实验

| 机器人策略训练 | 家庭长时序操作 |
| --- | --- |
| [SmolVLA-LIBERO 训练与评估](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/01SmolVLA-LIBERO/01SmolVLA-libero.md) | [RoboCasa365 与 DexJoCo 多任务复现](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README_09_AMD_Physical_AI仿真基准与长程视频复现.md) |
| 从数据读取、训练到闭环回放 | 多任务协议、策略推理和多视角视频导出 |

| 感知与建图 | 机器人设计 |
| --- | --- |
| [LingBot-Map 视频流式三维重建](./04-具身场景的计算机视觉、3D重建/03-LingBot-Map视频流式三维重建.md) | [代码驱动机器人与机械结构设计](./21-机械臂和机器人设计/README.md) |
| 连续视频、轨迹、深度与点云 | 参数化建模、格式导出和结构检查 |

## 环境说明

不同章节使用的环境不同，不建议在一个 Python 环境中安装全部依赖。开始实验前先阅读对应章节的环境表，并为仿真器或模型创建独立环境。

常见组合包括：

- Python 3.10 与 MuJoCo：基础机器人学和轻量仿真实验。
- Isaac Sim / Isaac Lab：大规模并行训练与高保真场景。
- LeRobot：数据采集、策略训练和机器人设备接口。
- ROCm 或 CUDA：根据章节标注选择对应的 GPU 运行环境。

模型、数据集、缓存和批量视频应放在仓库之外，通过环境变量传入路径，避免把大文件写入源码目录。

## 教材使用约定

每篇实验章节应明确工作目录、环境、输入、命令、预期输出和常见故障。运行短流程时，先确认数据读取、模型前向、反向传播或环境步进是否正常；正式结果以章节列出的完整训练和评估协议为准。

遇到链接、命令或内容问题，可以提交 [Issue](https://github.com/datawhalechina/every-embodied/issues)。新增章节请遵循[教材编写规范](./.github/TUTORIAL_STYLE_GUIDE.md)，并优先提供可重复运行的最小示例。

## 贡献者

| 姓名 | 主要职责 | GitHub |
| --- | --- | --- |
| Ethan-Chen-plus / 陈可为 | 项目维护，策略、仿真与专题教程 | [@Ethan-Chen-plus](https://github.com/Ethan-Chen-plus) |
| YYPro | 第 1—4 专题 | [@YYpro](https://github.com/Suibian-YY-pro) |
| 李昀迪 | 机器人学、导航与综合项目 | [@muzilyd](https://github.com/muzilyd) |
| 张天一 | 导航专题 | [@GodoneZ](https://github.com/GodoneZ) |
| 霍海杰 | 具身智能概述 | [@howe12](https://github.com/howe12) |

感谢所有参与文档修订、实验复现、英文翻译和问题反馈的社区贡献者。完整贡献记录见 [GitHub Contributors](https://github.com/datawhalechina/every-embodied/graphs/contributors)。

## 许可协议

本项目文档与教程采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可协议。引用、分享或改编时请保留来源署名。
