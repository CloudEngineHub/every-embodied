# Genie Sim 环境配置、基准评估与数据回放

Genie Sim Benchmark 是 AgiBot World 面向具身操作提供的仿真评估工具。本章将资产下载、容器启动、任务评估、策略接入、遥操作和轨迹回放整理成一条可复现链路。

- 官方文档：[Genie Sim Benchmark](https://agibot-world.com/sim-evaluation/docs/)
- 资产仓库：[GenieSimAssets](https://huggingface.co/datasets/agibot-world/GenieSimAssets)
- 策略仓库：[AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World)
- 问题跟踪：[AgibotTech/genie_sim issues](https://github.com/AgibotTech/genie_sim/issues)

> 当前上游运行栈基于 Isaac Sim 和 NVIDIA Container Toolkit。安装前应先核对官方硬件与驱动要求。本章的目标是准确复现上游链路，不替换其运行时约束。

## 1. 目录约定

所有命令都使用环境变量表示源码、资产和输出目录：

```bash
export GENIESIM_ROOT="${GENIESIM_ROOT:-/path/to/genie_sim}"
export SIM_ASSETS="${SIM_ASSETS:-$HOME/GenieSimAssets}"
export GENIESIM_OUTPUT="${GENIESIM_OUTPUT:-$GENIESIM_ROOT/output}"

mkdir -p "$SIM_ASSETS" "$GENIESIM_OUTPUT"
```

后续命令默认从源码根目录执行：

```bash
cd "$GENIESIM_ROOT"
```

## 2. 资产下载

Genie Sim 的场景和对象使用 Git LFS 管理。先安装 Git LFS，再完整拉取资产：

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install

git clone https://huggingface.co/datasets/agibot-world/GenieSimAssets "$SIM_ASSETS"
cd "$SIM_ASSETS"
git lfs pull
```

下载完成后检查文件数量和占用空间：

```bash
find "$SIM_ASSETS" -type f | wc -l
du -sh "$SIM_ASSETS"
```

若目录已经存在，应在原目录执行 `git pull` 和 `git lfs pull`，不要重新克隆一份重复资产。

## 3. 容器运行时

### 3.1 安装 NVIDIA Container Toolkit

按照 [Isaac Sim 容器安装文档](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html) 配置 Docker 和 NVIDIA Container Toolkit。安装后先运行最小设备检查：

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

该命令必须能看到显卡型号、驱动版本和显存信息。若设备检查失败，应先修复驱动或容器运行时，再构建 Genie Sim 镜像。

### 3.2 构建镜像

```bash
cd "$GENIESIM_ROOT"
docker build \
  -f ./scripts/dockerfile \
  -t registry.agibot.com/genie-sim/open_source:latest \
  .
```

构建阶段需要代理时，仅把代理传给本次构建，不要永久改写容器配置：

```bash
export DOCKER_HOST_IP="${DOCKER_HOST_IP:-172.17.0.1}"

docker build \
  --add-host="host.docker.internal:$DOCKER_HOST_IP" \
  --build-arg http_proxy="http://host.docker.internal:7890" \
  --build-arg https_proxy="http://host.docker.internal:7890" \
  --build-arg no_proxy="localhost,127.0.0.1" \
  -f ./scripts/dockerfile \
  -t registry.agibot.com/genie-sim/open_source:latest \
  .
```

运行仿真时不应沿用会拦截本机通信的代理变量。

## 4. 启动服务端与最小任务

### 4.1 启动图形容器

```bash
cd "$GENIESIM_ROOT"
SIM_ASSETS="$SIM_ASSETS" ./scripts/start_gui.sh
```

进入容器：

```bash
./scripts/into.sh
```

在容器中启动仿真服务端：

```bash
omni_python server/source/genie.sim.lab/raise_standalone_sim.py \
  --enable_curobo True
```

等服务端显示应用就绪后，再启动客户端。首次加载场景通常会比后续运行更慢。

### 4.2 运行官方示例

在另一终端进入同一容器：

```bash
cd "$GENIESIM_ROOT"
./scripts/into.sh
```

运行补货示例：

```bash
omni_python benchmark/task_benchmark.py \
  --task_name curobo_restock_supermarket_items \
  --env_class DemoEnv
```

开源示例、挑战赛任务和自定义策略的入口不同。出现“只有步数推进、没有任务动作”时，应先确认当前任务是否需要额外的推理策略或遥操作输入。

## 5. 一键任务入口

`autorun.sh` 将场景启动、通信和模式选择封装为统一入口：

```bash
# 只启动任务布局
./scripts/autorun.sh <TASK_NAME>

# 键盘遥操作
./scripts/autorun.sh <TASK_NAME> keyboard

# PICO 遥操作
./scripts/autorun.sh <TASK_NAME> pico <HOST_IP>

# 回放已有状态轨迹
./scripts/autorun.sh <TASK_NAME> replay <STATE_FILE_PATH>

# 运行模型推理
./scripts/autorun.sh <TASK_NAME> infer

# 清理任务进程
./scripts/autorun.sh clean
```

常用任务包括：

| 场景 | 任务名 |
| --- | --- |
| 咖啡区 | `genie_task_cafe_espresso` |
| 咖啡区 | `genie_task_cafe_toast` |
| 家庭 | `genie_task_home_clean_desktop` |
| 家庭 | `genie_task_home_collect_toy` |
| 家庭 | `genie_task_home_microwave_food` |
| 家庭 | `genie_task_home_open_drawer` |
| 家庭 | `genie_task_home_pass_water` |
| 家庭 | `genie_task_home_pour_water` |
| 家庭 | `genie_task_home_wipe_dirt` |
| 超市 | `genie_task_supermarket_cashier_packing` |
| 超市 | `genie_task_supermarket_stock_shelf` |
| 超市 | `genie_task_supermarket_pack_fruit` |

任务结束后按 `q` 或 `Q` 正常退出。输出目录会持续增长，应定期整理 `$GENIESIM_OUTPUT` 中的录像和通信记录。

## 6. 基准评估

### 6.1 运行评估

```bash
cd "$GENIESIM_ROOT"
SIM_ASSETS="$SIM_ASSETS" ./scripts/start_gui.sh
./scripts/autorun.sh genie_task_home_pour_water infer
```

正式评估前先使用单任务验证模型、通信和资产路径，再扩大任务数量与回合数。

### 6.2 ADER 规则

ADER 使用动作和条件组合描述任务阶段。以下示例要求机器人先接近并抓住瓶子，再将瓶子放入手提袋：

```json
{
  "Acts": [
    {
      "ActionList": [
        {
          "ActionSetWaitAny": [
            {"Follow": "beverage_bottle_002|[0.2,0.2,0.2]|right"},
            {"Timeout": 120},
            {"Onfloor": "beverage_bottle_002|0.0"}
          ]
        },
        {
          "ActionSetWaitAny": [
            {"PickUpOnGripper": "beverage_bottle_002|right"},
            {"Timeout": 120}
          ]
        },
        {
          "ActionSetWaitAny": [
            {"Inside": "beverage_bottle_002|handbag_000|1"},
            {"StepOut": 1000}
          ]
        }
      ]
    }
  ],
  "Objects": [
    {
      "bottle": ["beverage_bottle_002"],
      "handbag": ["handbag_000"]
    }
  ],
  "Problem": "pack_in_the_supermarket"
}
```

常用条件如下：

| 条件 | 用途 |
| --- | --- |
| `ActionList` | 按顺序执行子动作 |
| `ActionSetWaitAny` | 任一条件满足时结束当前阶段 |
| `ActionSetWaitAll` | 所有条件满足时结束当前阶段 |
| `Timeout` | 按时间终止阶段 |
| `StepOut` | 按步数终止阶段 |
| `Follow` | 检查夹爪是否进入目标附近的边界框 |
| `PickUpOnGripper` | 检查目标是否被夹爪抓住 |
| `Inside` | 检查物体是否进入容器 |
| `Ontop` | 检查物体是否位于另一物体上方 |
| `PushPull` | 检查抽屉等关节体是否达到目标区间 |
| `Onfloor` | 检查物体是否跌落到参考高度以下 |

逐回合输出至少应包含任务名、唯一标识、开始时间、结束时间、终止码、步数、阶段进度和分数。汇总成功率时应保留逐回合原始结果。

## 7. 接入自定义策略

### 7.1 代码接口

从仓库提供的推理模板开始，保持 `infer` 入口和通信结构不变。推荐目录结构：

```text
main/
├── model/
│   └── demo_infer.py
├── infer.py
├── genie_sim_ros.py
└── requirements.txt
```

模型通过 ROS 2 话题与仿真环境交换数据：

| 话题 | 方向 | 消息类型 | 内容 |
| --- | --- | --- | --- |
| `/joint_command` | 模型 → 仿真 | `JointState` | 目标关节命令 |
| `/joint_states` | 仿真 → 模型 | `JointState` | 当前关节状态 |
| `/sim/head_img` | 仿真 → 模型 | `CompressedImage` | 头部彩色图像 |
| `/sim/left_wrist_img` | 仿真 → 模型 | `CompressedImage` | 左腕彩色图像 |
| `/sim/right_wrist_img` | 仿真 → 模型 | `CompressedImage` | 右腕彩色图像 |
| `/sim/head_depth_img` | 仿真 → 模型 | `CompressedImage` | 头部深度图像 |
| `/sim/left_wrist_depth_img` | 仿真 → 模型 | `CompressedImage` | 左腕深度图像 |
| `/sim/right_wrist_depth_img` | 仿真 → 模型 | `CompressedImage` | 右腕深度图像 |

策略接入时必须核对关节顺序、相机顺序、图像编码、控制频率和夹爪范围。

### 7.2 运行推理

```bash
cd "$GENIESIM_ROOT"
SIM_ASSETS="$SIM_ASSETS" ./scripts/start_gui.sh
./scripts/autorun.sh <TASK_NAME> infer
```

若策略需要额外依赖，将依赖写入随模型提供的 `requirements.txt`，并在隔离环境中先执行导入检查。

## 8. 遥操作

### 8.1 键盘控制

启动服务端：

```bash
cd "$GENIESIM_ROOT"
SIM_ASSETS="$SIM_ASSETS" ./scripts/start_gui.sh
./scripts/into.sh
omni_python server/source/genie.sim.lab/raise_standalone_sim.py
```

在另一容器终端启动键盘客户端：

```bash
omni_python teleop/teleop.py \
  --task_name genie_task_home_microwave_food \
  --mode keyboard
```

| 按键 | 功能 | 按键 | 功能 |
| --- | --- | --- | --- |
| `w` / `s` | 末端向前 / 向后 | `q` / `e` | 末端向上 / 向下 |
| `a` / `d` | 末端向左 / 向右 | `i` / `k` | 滚转正 / 负 |
| `j` / `l` | 俯仰正 / 负 | `u` / `o` | 偏航正 / 负 |
| 方向键 | 基座移动 | `Shift` + 方向键 | 头部姿态 |
| `Ctrl` + 方向键 | 腰部姿态 | `Ctrl` + `Tab` | 切换手臂 |
| `c` | 关闭夹爪 | `Ctrl` + `c` | 打开夹爪 |
| `r` | 重置 |  |  |

### 8.2 PICO 控制

PICO 与主机需要位于同一局域网。在资源库中启动 AIDEA Vision App，选择无线连接并填写主机地址。客户端命令为：

```bash
omni_python teleop/teleop.py \
  --task_name genie_task_home_microwave_food \
  --mode pico \
  --host_ip <HOST_IP>
```

开始采集前应分别验证基座、腰部、头部、双臂和夹爪的方向，避免把控制映射错误写入演示数据。

## 9. 轨迹记录与回放

### 9.1 记录状态轨迹

先启动普通服务端，再用 `--record` 运行遥操作客户端：

```bash
omni_python teleop/teleop.py \
  --task_name genie_task_home_pour_water \
  --mode keyboard \
  --record
```

状态轨迹默认写入：

```text
output/recording_data/<TASK_NAME>/state.json
```

该文件包含机器人关节、物体姿态和相机姿态等场景状态。

### 9.2 回放并导出图像和视频

以禁用物理的记录模式启动服务端：

```bash
omni_python server/source/genie.sim.lab/raise_standalone_sim.py \
  --disable_physics \
  --record_img \
  --record_video
```

在客户端回放：

```bash
export TASK_NAME="genie_task_home_pour_water"

omni_python teleop/replay_state.py \
  --task_file "teleop/tasks/${TASK_NAME}.json" \
  --state_file "output/recording_data/${TASK_NAME}/state.json" \
  --record
```

导出的图像和视频位于：

```text
output/recording_data/<TASK_NAME>/<INDEX>/
```

记录和渲染分成两个阶段，可以降低交互采集时的负载，并允许使用同一条轨迹重复生成不同视角的视频。

## 10. 自定义任务

一个完整任务配置至少包含：

1. `task`：唯一任务名；
2. `objects`：非交互对象、固定交互对象和随机交互对象；
3. `recording_setting`：需要记录的相机；
4. `robot`：机器人配置和初始基座姿态；
5. `scene`：场景标识、功能区域、资产目录和场景文件；
6. `stages`：动作、主动对象和被动对象组成的阶段定义。

任务文件可放在：

```text
benchmark/bddl/eval_tasks/<TASK_NAME>.json
```

任务到场景的映射位于：

```text
benchmark/bddl/task_to_preselected_scenes.json
```

评估规则位于：

```text
benchmark/bddl/task_definitions/<TASK_NAME>/problem0.bddl
```

自定义策略应继承项目提供的基础策略类，并实现初始化、重置和动作推理：

```python
class YourPolicy(BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        # Load configuration, normalization statistics, and model weights.

    def reset(self) -> None:
        # Clear temporal state at the beginning of each episode.
        pass

    def act(self, observations, **kwargs):
        # Return target robot joints in the interface order required by Genie Sim.
        raise NotImplementedError
```

运行自定义策略：

```bash
python3 benchmark/task_benchmark.py \
  --task_name <TASK_NAME> \
  --policy_class <POLICY_NAME> \
  --env_class OmniEnv
```

## 11. 故障排查

### 11.1 服务端已启动，客户端无法连接

检查 50051 端口和服务端进程：

```bash
ss -ltnp | grep 50051
```

若防火墙启用，可只为可信网络开放所需端口：

```bash
sudo ufw allow 50051/tcp
```

不要直接结束所有 Python 进程。先确认占用端口的进程属于旧任务，再按进程号结束。

### 11.2 场景加载卡住

依次检查：

1. `SIM_ASSETS` 是否指向完整资产目录；
2. `git lfs pull` 是否完成；
3. 服务端和客户端代码是否来自兼容提交；
4. 代理变量是否干扰本机通信；
5. 首次加载期间显存和磁盘是否仍在变化。

相关上游记录可参考 [issue 11](https://github.com/AgibotTech/genie_sim/issues/11)、[issue 29](https://github.com/AgibotTech/genie_sim/issues/29) 和 [issue 34](https://github.com/AgibotTech/genie_sim/issues/34)。

### 11.3 模型有动作，但任务没有完成

检查观测字典、关节顺序、动作范围、控制频率和评估阶段。建议保存每步关节目标、夹爪命令、阶段状态和终止原因，再根据 ADER 规则定位失败位置。

### 11.4 碰撞不稳定

操作物体的碰撞网格应尽量简单。优先使用凸包或凸分解，避免在小区域堆叠过密的三角网格；同时检查质量、摩擦和关键抓取区域的碰撞体是否合理。

## 12. 完成标准

完成本章后，应具备以下可验证产物：

- 通过设备检查的容器运行时；
- 完整的 GenieSimAssets 目录；
- 一个可以稳定启动的官方任务；
- 一份逐回合评估结果；
- 一条自定义策略通信链路；
- 一份可回放的 `state.json`；
- 从该轨迹导出的图像或视频。
