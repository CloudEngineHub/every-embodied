# MicroDuck HelloWorld：策略控制 MuJoCo

先打开 [00_MuJoCo策略控制_HelloWorld.ipynb](../../03-Notebook/00_MuJoCo策略控制_HelloWorld.ipynb)，让小鸭子在同一个场景中站稳、前进、左转和停止，再学习各任务的训练与部署。

## 下载仓库后怎样开始

需要完整仓库中的机器人 XML、网格文件和本目录代码。行走策略约 0.8 MB，Notebook 自动下载并校验，不需要用户填写私有工作站的模型路径。

在本专题目录使用 Python 3.11 或 3.12 建立最小环境：

```bash
python3 -m venv .venv-hello
source .venv-hello/bin/activate
python -m pip install -r 02-可运行代码/hello-world/requirements.txt
python -m jupyter lab 03-Notebook
```

Windows 使用 `python -m venv .venv-hello`，激活命令为 `.venv-hello\Scripts\Activate.ps1`，其余步骤相同。下载不畅时，可设置 `MICRODUCK_DOWNLOAD_PROXY` 为自己的代理地址。模型缓存放在 `~/.cache/microduck-learning/`，不提交到 Git。

这份 HelloWorld 使用 CPU 上的 ONNX Runtime 和原生 MuJoCo，依赖中没有训练框架，也不要求显卡或开发板。

## 亲手执行控制闭环

Notebook 第 4 节执行一个控制步，第 5 节给出持续控制的实际循环：

```python
observation = sim.observation()
actions = sim.session.run([sim.output_name], {sim.input_name: observation})[0]
sim.data.ctrl[:] = sim.default_pose + sim.action_scale * actions[0]
sim.last_action[:] = actions[0]
for _ in range(4):
    mujoco.mj_step(sim.model, sim.data)
```

第 6–9 节通过 Python 修改速度命令，然后继续运行上述循环：

```python
sim.forward(speed=0.30)              # 向前，m/s
run_for_seconds(4.0, "forward")
sim.turn(yaw_rate=0.50)              # 左转，rad/s；负值右转
run_for_seconds(3.0, "turn_left")
sim.stop()                          # 速度命令归零，策略仍持续站稳
run_for_seconds(3.0, "stop")
```

运行段之间沿用同一物理状态，初始化后不改写机器人位置或朝向。`realtime=True` 按墙钟时间运行；默认尽快计算，视频按真实仿真时间播放。命令写入 61 维观测的 `[48:51]`，并不是直接控制 14 个关节。

Notebook 保存实际位移、朝向、速度曲线和逐步控制记录。完整视频与 JSON 写入 `03-Notebook/outputs/control_helloworld/`，仓库只保存执行结果和 GIF 预览。

## 已执行的结果

12 个代码单元已在 Ubuntu 从头执行，使用同一个场景完成 550 个控制步，期间没有重置位置、朝向或姿态。

| 阶段 | 命令与时长 | 实际响应 |
| --- | --- | --- |
| 前进 | 0.30 m/s，4 s | 前进约 0.48 m |
| 左转 | 0.50 rad/s，3 s | 朝向左转约 44° |
| 停止 | 速度命令归零，3 s | 最后 1 s 平均绝对前向速度约 0.00053 m/s |

测量结果见[执行记录](../../01-任务资料/07-RDK端侧与网页部署/assets/hello_world/validation.json)。命令速度与实际速度分别显示在 Notebook 曲线中，便于观察跟踪误差。

## 模型来源与接口

使用 [Pollen Robotics MicroDuck 模拟器的 walking 策略](https://huggingface.co/spaces/pollen-robotics/microduck-simulator/tree/ad9d41a763f30dc770817c33d92dcfec0205db1e/app/public/policies)，固定版本 `ad9d41a763f30dc770817c33d92dcfec0205db1e`。

- 输入：`float32 obs[1,61]`，含角速度、投影重力、关节位置与速度、上一动作和命令。
- 输出：`float32 action[1,14]`，按模型元数据中的关节顺序映射到位置目标。
- SHA-256：`e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c`。
- 物理步长：5 ms；每 4 个物理步执行一次策略推理，控制频率 50 Hz。

模型元数据与场景的关节顺序、默认姿态、命令布局均在启动时核对。这里复用仓库的标准 MicroDuck 模型及其执行器设置；篮球、爬梯和跨缝跳跃使用各自的场景与策略。

## 连接实机时改变什么

MuJoCo 的 `data.ctrl` 接收关节位置目标。实机用机器人 SDK/通信驱动读取 IMU 与关节反馈，并按相同关节顺序发送目标位置；零位、限位、控制周期和急停需先校验。RDK 的 CPU/BPU 可承担策略推理，工作站或机器人端负责观测与执行器接口。

本 HelloWorld 的下发对象是 MuJoCo 位置执行器。Notebook 已实现并展示这条仿真闭环；实机通信驱动不在本例中启动。
