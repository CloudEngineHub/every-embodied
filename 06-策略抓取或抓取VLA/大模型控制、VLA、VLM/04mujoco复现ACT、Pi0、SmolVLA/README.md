# 在 MuJoCo 中采集数据并训练 ACT、Pi0 与 SmolVLA

本章给出一条可执行的机器人模仿学习流程：在 MuJoCo 中采集示教，检查 LeRobot 数据集，训练 ACT、Pi0 或 SmolVLA，再把策略接回同一环境运行。完成本章后，读者应能辨认观测、状态和动作字段，并能确认训练与部署使用了相同的数据约定。

## 1. 学习路径

| 阶段 | 入口 | 产物 |
| --- | --- | --- |
| OMY 示教采集 | [1.collect_data.ipynb](1.collect_data.ipynb) | `demo_data` 数据集 |
| 其他机械臂采集 | [Nova5](1.collect_data_nova5.ipynb)、[xArm6](1.collect_data_xarm6.ipynb)、[xArm7](1.collect_data_xarm7.ipynb)、[Franka XR](1.collect_data_franka_xr.ipynb) | 对应机械臂的 LeRobot 数据集 |
| 数据回放 | [OMY](2.visualize_data-omy.ipynb)、[Nova5](2.visualize_data-nova5.ipynb)、[xArm7](2.visualize_data-xarm7.ipynb) | 图像、状态和动作的同步检查 |
| ACT 训练 | [3.train.ipynb](3.train.ipynb) | ACT 权重与训练曲线 |
| ACT 部署 | [4.deploy.ipynb](4.deploy.ipynb) | 环境闭环回放 |
| 语言条件数据 | [5.language_env.ipynb](5.language_env.ipynb)、[6.visualize_data.ipynb](6.visualize_data.ipynb) | 含任务文本的数据集 |
| Pi0 | [7.pi0.ipynb](7.pi0.ipynb)、[pi0_omy.yaml](pi0_omy.yaml) | Pi0 微调权重 |
| SmolVLA | [8.smolvla.ipynb](8.smolvla.ipynb)、[smolvla_omy.yaml](smolvla_omy.yaml) | SmolVLA 微调权重 |
| XR 遥操作 | [9.teleop_xr.ipynb](9.teleop_xr.ipynb) | XR 控制与采集流程 |

第一次学习建议按 OMY 采集、OMY 回放、ACT 训练、ACT 部署的顺序完成，再进入语言条件模型和其他机械臂。

## 2. 环境安装

仓库的 [requirements.txt](requirements.txt) 固定了 Python 依赖，其中 MuJoCo 版本为 3.1.6，LeRobot 固定到指定提交。建议创建独立环境：

```bash
conda create -n lerobot-mujoco python=3.10 -y
conda activate lerobot-mujoco
pip install -r requirements.txt
pip install jupyterlab ipywidgets ipykernel
python -m ipykernel install --user --name lerobot-mujoco --display-name "LeRobot MuJoCo"
jupyter lab .
```

`requirements.txt` 中的 PyTorch 安装源面向 CUDA。使用其他计算后端时，应先按对应平台安装 PyTorch，再安装其余依赖，避免安装命令覆盖已有框架。

解压场景资源：

```bash
cd asset/objaverse
unzip plate_11.zip
```

启动 Notebook 前检查：

```bash
python -c "import mujoco; print(mujoco.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. 采集示教数据

打开 [1.collect_data.ipynb](1.collect_data.ipynb)，按顺序运行环境、数据集和控制循环单元。任务要求机械臂抓起杯子并放到盘子上；杯子位置、夹爪释放状态和末端高度共同参与成功判定。

<img src="./media/teleop.gif" width="480" height="360" alt="OMY 键盘示教">

常用键位：

- `W/A/S/D`：在水平面移动；
- `R/F`：沿竖直方向移动；
- `Q/E` 与方向键：调整姿态；
- `Space`：切换夹爪；
- `Z`：重置环境并丢弃当前回合缓存。

示例数据以 20 Hz 保存，包含两个相机、六维状态、七维动作和物体初始状态：

```python
features = {
    "observation.image": {"dtype": "image", "shape": (256, 256, 3)},
    "observation.wrist_image": {"dtype": "image", "shape": (256, 256, 3)},
    "observation.state": {"dtype": "float32", "shape": (6,)},
    "action": {"dtype": "float32", "shape": (7,)},
    "obj_init": {"dtype": "float32", "shape": (6,)},
}
```

采集完成后确认：

1. 每个回合都有连续递增的帧索引；
2. 两路图像与状态、动作使用相同时间步；
3. 动作单位、关节顺序和夹爪符号与环境一致；
4. 只有完成任务的示教才进入成功数据集。

## 4. 回放并检查数据

根据机器人选择对应的 `2.visualize_data-*.ipynb`。回放时同时观察场景运动、相机画面和动作曲线。

<img src="./media/data.gif" width="480" height="360" alt="示教数据回放">

重点检查三类问题：

- 图像是否发生错帧、黑帧或相机顺序交换；
- 状态与动作维度是否与配置一致；
- 回放轨迹是否能够复现采集时的运动方向。

这些检查应在训练前完成。字段错误进入训练后，损失仍可能下降，但策略无法正确控制环境。

## 5. 训练与部署 ACT

打开 [3.train.ipynb](3.train.ipynb) 训练 ACT。示例使用动作分块，将一段未来动作作为一个训练目标。训练前根据显存调整 `batch_size`；在交互环境中遇到数据加载进程错误时，可将 `num_workers` 设为 `0`。

训练完成后，打开 [4.deploy.ipynb](4.deploy.ipynb)，从权重目录加载策略并运行闭环环境。

<img src="./media/rollout.gif" width="480" height="360" alt="ACT 闭环回放">

部署时需要与训练保持一致的内容包括：相机名称与顺序、图像尺寸、状态维度、动作维度、归一化统计、动作块长度和控制频率。

## 6. 语言条件数据

[5.language_env.ipynb](5.language_env.ipynb) 在采集帧中加入任务文本，用于区分不同颜色杯子的操作指令；[6.visualize_data.ipynb](6.visualize_data.ipynb) 用于检查语言字段、图像和动作是否对齐。

<img src="./media/data_v2.gif" width="480" height="360" alt="语言条件数据回放">

语言条件训练前，应统一指令模板。表示同一目标的文本可以有多种自然语言表述，但颜色、对象和空间关系不能相互矛盾。

## 7. 训练 Pi0

训练入口为 [train_model.py](train_model.py)，配置文件为 [pi0_omy.yaml](pi0_omy.yaml)：

```bash
python train_model.py --config_path pi0_omy.yaml
```

[7.pi0.ipynb](7.pi0.ipynb) 展示依赖、配置、训练调用和策略加载过程。运行前需要填写实验记录服务的账号配置，或关闭对应日志功能。

<img src="./media/rollout2.gif" width="480" height="360" alt="Pi0 策略回放">

## 8. 训练 SmolVLA

SmolVLA 使用同一训练入口和 [smolvla_omy.yaml](smolvla_omy.yaml)：

```bash
python train_model.py --config_path smolvla_omy.yaml
```

[8.smolvla.ipynb](8.smolvla.ipynb) 展示模型加载、训练和部署。切换模型时不要复用不匹配的归一化统计或动作配置。

<img src="./media/rollout3.gif" width="480" height="360" alt="SmolVLA 策略回放">

## 9. 闭环评估

模型训练结束后，使用独立种子和固定成功判定运行多回合评估。建议至少记录：

- 总回合数与成功回合数；
- 接近、抓取、抬升、搬运和放置阶段是否完成；
- 每回合种子、终止原因和视频路径；
- 使用的权重、数据配置和归一化统计。

更完整的诊断方法见 [策略诊断与物理成功评估](09策略诊断与物理成功评估.md)。

## 10. 本章检查点

- 能采集至少一个成功示教并在回放中复现动作。
- 能说明图像、状态、动作和语言字段的形状与含义。
- 能从同一数据约定训练并加载 ACT、Pi0 或 SmolVLA。
- 能用固定回合协议报告闭环成功率，而不是只观察训练损失。

## 参考与资源来源

- OMY 机械臂模型：[robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)
- MuJoCo 解析器参考：[yet-another-mujoco-tutorial-v3](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3)
- 训练框架：[LeRobot](https://github.com/huggingface/lerobot)
- 杯子与盘子资源：[Objaverse](https://objaverse.allenai.org/)
