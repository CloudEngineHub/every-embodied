# LightNav-0 多场景导航学习

使用官方预训练 LightNav-0，在 GPU［图形处理器］上运行视觉语言导航。不训练导航模型，不连接开发板。
配套的第 8 本 Notebook［交互式教程］提供默认路径、模型服务、三个场景、真实预测、视频与轨迹对照。
这是 [8 天学习计划](../../学习计划.md)的第 8 天，围绕图像、语言指令、轨迹与行走控制展开实践。

## 准备环境

模型与模拟器使用两个环境，权重和环境放在仓库外。以下以 Ubuntu 为例：

```bash
git clone https://github.com/lightorigins/LightNav-0.git ~/projects/LightNav-0
uv venv --python 3.11 /data/Data14TB/envs/lightnav-0
uv pip install --python /data/Data14TB/envs/lightnav-0/bin/python -e "$HOME/projects/LightNav-0[vllm,video]"
/data/Data14TB/envs/lightnav-0/bin/hf download LightOriginsHQ/LightNav-0 \
  --local-dir /data/Data14TB/checkpoints/LightNav-0

uv venv --python 3.11 /data/Data14TB/envs/lightnav-mujoco
uv pip install --python /data/Data14TB/envs/lightnav-mujoco/bin/python \
  -e "$HOME/projects/LightNav-0/mujoco_demo[microduck]" \
  jupyterlab nbclient nbformat nbconvert ipykernel matplotlib imageio imageio-ffmpeg websocket-client
```

沿用官方 MicroDuck 机器人资源和行走策略：

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git ~/workspaces/microduck_rl
mkdir -p /data/Data14TB/checkpoints/microduck-navigation
curl -fL https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/alpha_walking.onnx \
  -o /data/Data14TB/checkpoints/microduck-navigation/alpha_walking.onnx
```

已存在的环境、代码和权重直接复用。模型推理使用上游固定的 `vllm==0.19.1`，模拟器需要 MuJoCo 和场景资源。
按官方说明使用合适的加速计算环境，不输出具体设备型号。

## 打开与执行

```bash
/data/Data14TB/envs/lightnav-mujoco/bin/python -m ipykernel install --user \
  --name microduck-lightnav --display-name 'MicroDuck LightNav'
cd '<专题路径>'
/data/Data14TB/envs/lightnav-mujoco/bin/jupyter lab --no-browser --ip=127.0.0.1 --port=8890
```

打开 `03-Notebook/08_LightNav0_视觉语言导航_GPU_多场景.ipynb`，选择上述内核，从上到下运行。
默认参数已填好。其他机器可设置 `LIGHTNAV_ROOT`、`LIGHTNAV_MODEL_ENV`、`LIGHTNAV_MODEL_PATH`、
`LIGHTNAV_ROBOT_MODEL`、`LIGHTNAV_WALKING_POLICY` 和 `LIGHTNAV_OUTPUT`。
默认记录在 `/data/Data14TB/lightnav-recordings/teaching/notebook`；小尺寸视频副本写入
`03-Notebook/outputs/lightnav_gpu` 供网页播放。该目录不提交到仓库。
启动时以专题目录作为网页根目录，教程与配套代码的相对链接即可正常打开。

完整批量执行使用：

```bash
cd '<专题路径>/03-Notebook'
/data/Data14TB/envs/lightnav-mujoco/bin/jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=microduck-lightnav \
  '08_LightNav0_视觉语言导航_GPU_多场景.ipynb'
/data/Data14TB/envs/lightnav-mujoco/bin/jupyter trust '08_LightNav0_视觉语言导航_GPU_多场景.ipynb'
```

## 场景与控制

| 场景 | 学习内容 |
| :-- | :-- |
| 官方住宅 | 官方 ProcTHOR `val_2` 多房间场景与模型接口 |
| 自建客厅 | 重新布置房间、扶手椅、沙发和盆栽，观察目标接近 |
| 自建走廊 | 窄空间与侧面家具，观察局部路径和转向 |

导航模型读取第一视角图像与指令，返回 10×3 的平面轨迹。
MPC［模型预测控制］使用模型轨迹输出速度，官方已有行走策略通过 ONNX Runtime［模型推理运行时］
的 CPU［中央处理器］接口输出 14 维关节动作。重力、接触与位置执行器沿用官方实现。
模型推理期间暂停仿真；视频按仿真时间编码，报告单独记录真实推理延迟。
目标坐标仅用于记录距离，未输入模型或用于规划控制。

每次运行覆盖各场景的 `*_latest.mp4`，保留对应预测文本、轨迹、停止标志和实验报告。
不嵌入大视频到交互式教程中；保留小尺寸图像与文本输出方便直接阅读。

## 文件与来源

- `lightnav_lesson.py`：新场景构建、模型协议、物理闭环和视频记录。
- `build_notebook.py`：以结构化格式生成教程；重新生成会清除旧执行输出，随后需完整执行。
- 官方代码：[LightNav-0](https://github.com/lightorigins/LightNav-0)。
- 场景与机器人接口：[官方模拟器](https://github.com/lightorigins/LightNav-0/blob/main/mujoco_demo/README.md)。
- 导航权重：[LightOriginsHQ/LightNav-0](https://huggingface.co/LightOriginsHQ/LightNav-0)。
- 机器人：[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)。

官方场景网格采用 CC BY 4.0；自建场景修改了墙体、地面和家具布局，修改记录随输出保存。
机器人网格遵循上游 CC BY-NC-SA 条款，代码与策略遵循各自上游许可。
