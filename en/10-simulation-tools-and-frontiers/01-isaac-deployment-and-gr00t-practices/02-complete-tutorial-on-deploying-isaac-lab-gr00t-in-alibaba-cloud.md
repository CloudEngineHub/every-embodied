# Complete Tutorial on Deploying Isaac Lab + GR00T on Alibaba Cloud

> Guide to deploying an embodied AI development environment for engineers and researchers with a Linux/GPU foundation
>
> Author’s tested environment: Alibaba Cloud A10 GPU preemptive instance

⚠️ **Version Declaration**: This tutorial is written based on the **January 2026** software version, which strongly relies on the following version combinations:
- Isaac Sim 4.2.0 + Isaac Lab v1.4.1
- GR00T N1.6 + transformers==4.51.3
- PyTorch 2.5.1 + CUDA 12.1

**Only guaranteed to be reproduction under the above version**. If using a newer version, you may need to adjust dependencies and configurations manually.

⚠️ **Safety Notice**: This tutorial is designed for quick experimentation. It uses simplified configurations such as the root user, `--network host`, and `xhost +`.**It is not recommended for production environments or long-term running servers**.

## Table of Contents

1. [ Environment Overview ](#1-环境概述)
2. [ Creating Alibaba Cloud Instances ](#2-阿里云实例创建)
3. [ Basic Environment Configuration ](#3-基础环境配置)
4. [ VNC Remote Desktop Configuration ](#4-vnc远程桌面配置)
5. [ Deployment of Isaac Sim + Isaac Lab ](#5-isaac-sim--isaac-lab-部署)
6. [ GR00T Environment Configuration ](#6-groot-环境配置)
7. [ GR00T + MuJoCo Evaluation ](#7-groot--mujoco-评估)
8. [ Common Issues and Troubleshooting Records ](#8-常见问题与踩坑记录)

---

## 1. Environment Overview

### Final Architecture

```
宿主机 (Ubuntu 22.04 + A10 GPU)
├── Docker: isaac-sim-gui
│   ├── Isaac Sim 4.2.0
│   ├── Isaac Lab v1.4.1
│   └── VNC显示 (端口6080)
└── Conda: groot环境
    ├── Isaac-GR00T代码
    ├── GR00T-N1.6-3B模型 (通用)
    └── GR00T-N1.6-G1-PnPAppleToPlate模型 (G1专用)
```

### Why is it designed this way?

- **Isaac Lab within Docker**: Isaac Sim has complex dependencies; the official Docker image is more convenient.
- **GR00T on host Conda**: GR00T requires PyTorch 2.5+, which conflicts with Isaac Sim's PyTorch 2.4.
- **VNC remote desktop**: WebRTC Livestream has compatibility issues on cloud servers; VNC is more stable.

#### Key Design Decisions

| Decision | Why do it this way | What happens if not done |
|--------|-------------|-----------------------|
| GR00T installed via Conda instead of Docker | Conflict with PyTorch 2.5+ and Isaac Sim | Dependency hell, environment crash |
| Skip flash-attn | Compilation is time-consuming and not necessary | OOM/freeze, server restart |
| Lock transformers==4.51.3 | New API changes | Model loading errors |
| Install GR00T with --no-deps | Avoid duplicate compilation of flash-attn | pip gets stuck |
| Manually clone submodules | curl download lacks .git | setup script errors |

### Deployment order (must be executed in sequence)

```
基础配置 (3) → VNC (4) → Isaac Docker (5) → GR00T Conda (6) → MuJoCo 评估 (7)
     │              │              │                │
     └── NVIDIA 驱动 → Docker → Container Toolkit ──┘
```

> ⚠️ **Do not skip steps or disrupt the order**, as each step depends on the previous configuration.

### Hardware Requirements

- GPU: NVIDIA A10 24GB or higher (RTX 3090/4090 also available)
- Memory: 32GB+
- Hard drive: **150GB+ SSD** (actual usage is about 80-100GB)
  - Ubuntu system: ~5GB
  - Isaac Sim Docker image: ~25GB
  - Docker runtime cache: ~10GB
  - Conda environment (PyTorch + CUDA): ~10GB
  - GR00T model (3B + G1): ~12GB
  - Logs and temporary files: ~10GB
  - Reserved space: ~30GB

---

## 2. Creating Alibaba Cloud Instances

### 2.0 Preparations

#### Register a NGC account (required to pull the Isaac Sim image)

1. Access https://ngc.nvidia.com
2. Click "Sign Up" to register with an email address (or log in using Google/GitHub)
3. After logging in, click the avatar in the top-right corner → "Setup" → "Generate API Key"
4. Click "Generate API Key", then copy and save it

> ⚠️ The API key is displayed only once; make sure to save it. It will be needed in subsequent `docker login nvcr.io`s.

### 2.1 Cost Description

- **Actual cost**: Completing the entire tutorial costs approximately **30 RMB** (preemptible instance + data usage fee)
- **Account requirements**: An Alibaba Cloud account must have a balance of **100 RMB or more** to purchase a preemptible instance
- The price of preemptible instances fluctuates, so the actual cost may vary

### 2.2 Select Instance Specifications

1. Log in to the Alibaba Cloud console → Cloud Virtual Machine ECS → Create instance.
2. Select the region: **Southwest 1 (Chengdu)** is recommended (lowest price).
3. Instance specification: Search for `ecs.gn7i` and choose the A10 24GB specification.
   - **ecs.gn7i-c16g1.4xlarge** (recommended): 16 vCPU + 60GB memory
   - ecs.gn7i-c8g1.2xlarge: 8 vCPU + 32GB memory

> 💡 **Cost-saving tips**: Prices vary greatly across regions!
> - Chengdu c16g1.4xlarge:约 **2.4 yuan/hour** (high configuration, low price)
> - Beijing/Hangzhou c8g1.2xlarge: about 3.1 yuan/hour
>
> It is recommended to choose Chengdu as it offers higher configurations at lower prices.
4. **Pay-as-you-go mode: Preemptible instances** (80-90% cheaper than pay-per-use)
   - Set the maximum price to 50-70% of the pay-per-use rate
   - Check "Instance release protection"

### 2.2 Image Selection

- Operating system: **Ubuntu 22.04 64-bit**
- Or choose the NVIDIA GPU cloud acceleration image (with pre-installed drivers)

### 2.3 Network Configuration

- Assign a public IP address (billed by traffic)
- Bandwidth: 5-10 Mbps is sufficient

### 2.4 Security Group Configuration

Create or modify a security group to open the following ports:

| Port | Purpose |
|------|------|
| 22 | SSH |

> ⚠️ **Security advice**: Do not directly open the VNC port (5901/6080). Using an SSH tunnel is safer. See below for details.

---

## 3. Basic Environment Configuration

### 3.1 SSH Connection

```bash
ssh root@<你的公网IP>
```

> ⚠️ **Important**: When connecting using Alibaba Cloud Workbench, do not select "Password-free connection"!
>
> Password-free connection uses a containerized Web terminal with restricted permissions, which prevents Docker from starting properly.
>
> Please choose "Password" or "Key Pair" for connection, or directly use the local SSH client.

### 3.2 Installing NVIDIA drivers (if not pre-installed in the image)

```bash
# 检查驱动
nvidia-smi

# 如果没有，安装驱动
apt update
apt install -y nvidia-driver-535
reboot
```

### 3.3 Installing Docker

```bash
# 检查是否已安装
docker --version

# 如果没有，安装 Docker
apt update
apt install -y docker.io
systemctl enable docker
systemctl start docker

# 验证安装
docker --version
```

### 3.4 Configuring Docker Domestic Images

Docker Hub is unstable to access in China. Configure image acceleration:

```bash
cat > /etc/docker/daemon.json << 'EOF'
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
EOF

systemctl restart docker
```

### 3.5 Installing NVIDIA Container Toolkit

Download the deb package directly for installation (stable reproduction in China):

```bash
cd /tmp

# 下载 4 个必需的包
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-container1_1.17.4-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-container-tools_1.17.4-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-container-toolkit-base_1.17.4-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-container-toolkit_1.17.4-1_amd64.deb

# 按顺序安装
dpkg -i libnvidia-container1_1.17.4-1_amd64.deb
dpkg -i libnvidia-container-tools_1.17.4-1_amd64.deb
dpkg -i nvidia-container-toolkit-base_1.17.4-1_amd64.deb
dpkg -i nvidia-container-toolkit_1.17.4-1_amd64.deb

# 配置 Docker
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# 验证（应该能看到 nvidia-smi 输出）
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

> 💡 **Tip**: The download link will automatically redirect to the domestic mirror (nvidia.cn), which is faster.

---

## 4. VNC Remote Desktop Configuration

### 4.1 Installing the Desktop Environment and VNC

```bash
apt update
apt install -y xfce4 xfce4-goodies tigervnc-standalone-server tigervnc-common novnc websockify
```

> 💡 **Installation instructions**:
> - If the "Daemons using outdated libraries" dialog box appears, press Tab to select `<Ok>` and then enter to continue.
> - This is a normal phenomenon when restarting services after an Ubuntu system update.

### 4.2 Configure VNC

```bash
# 设置 VNC 密码
vncpasswd
# 输入密码（至少6位），view-only 选 n

# 创建 VNC 配置
mkdir -p ~/.vnc
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF
chmod +x ~/.vnc/xstartup

# 启动 VNC 服务
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no

# 验证 VNC 是否正常启动
netstat -tlnp | grep 5901
# 应该看到 Xtigervnc 在监听 5901 端口，类似：
# tcp        0      0 0.0.0.0:5901            0.0.0.0:*               LISTEN      26811/Xtigervnc

# 启动 noVNC（Web 访问）
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5901 > /dev/null 2>&1 &
```

### 4.3 Accessing VNC

#### Method 1: SSH Tunnel (Recommended, safer)

There is no need to open the VNC port in the security group. Execute on the local computer:

```bash
# Windows PowerShell / Mac Terminal / Linux
ssh -L 6080:localhost:6080 root@<你的公网IP>
```

Maintain the SSH connection, and then open the browser: `http://localhost:6080/vnc.html`

#### Method 2: Direct access (port must be opened)

If you choose direct access, you need to open port 6080 in the security group (only for your IP):

Browser open: `http://<你的公网IP>:6080/vnc.html`

Enter the VNC password to enter the desktop.

> ⚠️ **Security Warning**: The VNC protocol has weak security, and directly exposing ports is vulnerable to attacks. It is strongly recommended to use an SSH tunnel.

### 4.4 VNC Self-starting (Optional)

```bash
cat > /etc/systemd/system/vncserver.service << 'EOF'
[Unit]
Description=VNC Server
After=network.target

[Service]
Type=forking
User=root
ExecStart=/usr/bin/vncserver :1 -geometry 1920x1080 -depth 24
ExecStop=/usr/bin/vncserver -kill :1

[Install]
WantedBy=multi-user.target
EOF

systemctl enable vncserver
```

---

## 5. Deployment of Isaac Sim + Isaac Lab

### 5.1 Log in to NGC and pull images

Isaac Sim image is hosted on NVIDIA NGC (GPU Cloud) and requires login first:

```bash
# 登录 NGC
docker login nvcr.io
```

The system will prompt you to enter your username and password:
- **Username**: Enter `$oauthtoken` (a fixed value, copy directly)
- **Password**: Enter the NGC API Key you obtained in Section 2.0

When `Login Succeeded` is seen, it indicates successful login.

```bash
# 拉取 Isaac Sim 4.2.0 镜像（约 15GB，需要一些时间）
docker pull nvcr.io/nvidia/isaac-sim:4.2.0
```

After the download is successful, the following will be displayed:
```
Status: Downloaded newer image for nvcr.io/nvidia/isaac-sim:4.2.0
```

Verified that the image has been downloaded:
```bash
docker images | grep isaac-sim
# 应该看到类似输出：
# nvcr.io/nvidia/isaac-sim   4.2.0   <IMAGE_ID>   <SIZE>
```

### 5.2 Starting Containers

```bash
# 在宿主机设置 X11 权限
export DISPLAY=:1
xhost +local:docker

# 启动容器
# 注意：--network host 和 xhost + 是为了简化配置，仅建议用于短期实验
docker run -it --name isaac-sim-gui --gpus all --network host \
  -e DISPLAY=:1 \
  -e ACCEPT_EULA=Y \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --entrypoint bash \
  nvcr.io/nvidia/isaac-sim:4.2.0
```

### 5.3 Preparation: Download dependencies on the host machine

Due to the network environment limitations within the container, it is recommended to download Isaac Lab and robomimic on the host machine:

```bash
# 退出容器，回到宿主机
exit

# 在宿主机执行（不要进入容器）
cd ~

# 安装下载工具（如果没有）
apt update
apt install -y git wget unzip

# 下载 Isaac Lab（使用国内镜像）
git clone --depth 1 --branch v1.4.1 https://gitclone.com/github.com/isaac-sim/IsaacLab.git

# 下载 robomimic（用于模仿学习任务）
wget https://github.com/ARISE-Initiative/robomimic/archive/refs/heads/master.zip
unzip master.zip
mv robomimic-master robomimic

# 重新启动容器
docker start isaac-sim-gui

# 复制进容器
docker cp IsaacLab isaac-sim-gui:/isaac-sim/
docker cp robomimic isaac-sim-gui:/tmp/
```

### 5.4 Installing dependencies within the container

```bash
# 进入容器
docker exec -it isaac-sim-gui bash

# 安装编译工具和 EGL 库（robomimic 需要）
apt update
apt install -y build-essential cmake pkg-config git
apt install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev

# 安装 robomimic
cd /tmp/robomimic
/isaac-sim/python.sh -m pip install -e .
```

### 5.5 Fixing Isaac Lab configuration files

There are two issues in the dependency configuration of Isaac Lab that need to be fixed:

```bash
cd /isaac-sim/IsaacLab

# 备份原文件
cp source/extensions/omni.isaac.lab_tasks/setup.py source/extensions/omni.isaac.lab_tasks/setup.py.bak

# 修复 1：rsl-rl 名称不匹配
# 原因：setup.py 中引用的是 GitHub 仓库名 "rsl-rl"，但 PyPI 上的包名是 "rsl-rl-lib"
# 这是上游的命名不一致问题，不影响功能
sed -i '46s/"rsl-rl@git+https:\/\/github.com\/leggedrobotics\/rsl_rl.git"/"rsl-rl-lib==2.3.0"/' source/extensions/omni.isaac.lab_tasks/setup.py

# 修复 2：robomimic 已手动安装，删除 git 克隆配置
# 原因：setup.py 试图从 GitHub 克隆 robomimic，但容器内网络不稳定
# 我们已在宿主机下载并复制进来，所以删除这个配置
sed -i '/robomimic@git/d' source/extensions/omni.isaac.lab_tasks/setup.py

# 验证修复
echo "=== 检查第 46 行（rsl-rl）==="
sed -n '46p' source/extensions/omni.isaac.lab_tasks/setup.py
echo "=== 检查第 50-56 行（robomimic 应该已删除）==="
sed -n '50,56p' source/extensions/omni.isaac.lab_tasks/setup.py
```

Expected output:
```
=== 检查第 46 行（rsl-rl）===
"rsl-rl": ["rsl-rl-lib==2.3.0"],
=== 检查第 52-56 行（robomimic）===
# Cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
# Remove duplicates in the all list to avoid double installations
EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))
```

If line 46 is still `rsl-rl@git+https://...`, it means the sed command did not take effect. Check whether the line number is correct.

### 5.6 Installing Isaac Lab

```bash
cd /isaac-sim/IsaacLab

# 创建符号链接
ln -s /isaac-sim _isaac_sim

# 修复 pip（Isaac Sim 容器内 pip 可能损坏）
/isaac-sim/python.sh -m ensurepip --upgrade
/isaac-sim/python.sh -m pip install --upgrade pip setuptools

# 安装 Isaac Lab（约 5-10 分钟）
./isaaclab.sh --install
```

During the installation process, you will see the download and installation of many dependency packages. Finally, it should display:
```
Successfully installed omni-isaac-lab_tasks-0.10.18 ...
```

### 5.7 Testing Isaac Lab

```bash
# 在容器内运行 demo
export DISPLAY=:1
cd /isaac-sim/IsaacLab

# 机械臂 demo
./isaaclab.sh -p source/standalone/demos/arms.py

# 双足机器人 demo
./isaaclab.sh -p source/standalone/demos/bipeds.py

# 人形机器人强化学习训练
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Humanoid-Direct-v0 \
  --num_envs 64 \
  --max_iterations 100
```

The simulation image should be visible on the VNC desktop. Press `Ctrl+C` to stop the program.

---

## 6. GR00T Environment Configuration

### 6.1 Installing Miniconda

```bash
# 退出 Docker 容器，回到宿主机
exit

# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init
source ~/.bashrc

# 接受 Conda 服务条款（新版 Conda 要求）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 6.2 Creating the GR00T Environment

```bash
# 创建 Python 3.10 环境
conda create -n groot python=3.10 -y
conda activate groot

# 安装 PyTorch（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Install flash-attn (optional, recommended to skip)

> ⚠️ **Important note**: The compilation of flash-attn is very time-consuming (15-60 minutes), and it consumes a lot of system resources, which may cause SSH/VNC disconnection.
> GR00T can operate normally without flash-attn, but the inference speed is slightly slower.
> **It is recommended to skip this step** and directly proceed to Section 6.3.

If you really need flash-attn (for ultimate performance):

```bash
# 安装编译依赖
pip install numpy psutil packaging ninja

# 设置 CUDA 环境变量（必须）
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# 限制并行编译数，防止内存不足
# 32GB 内存用 MAX_JOBS=4
# 58GB+ 内存用 MAX_JOBS=6
MAX_JOBS=6 pip install flash-attn --no-build-isolation -v
```

> ⏳ **Compilation process description**:
> - The `running bdist_wheel` stage will hang for 1–3 minutes, which is normal (due to cmake configuration)
> - Later, `[1/73] ...` will indicate the start of actual compilation (73 files in total)
> - The entire compilation process takes 15–60 minutes, depending on the number of CPU cores and MAX_JOBS settings

> ⚠️ **Abnormal phenomena during compilation (normal!)**:
> - **New SSH connections may fail**: Compilation consumes a lot of resources, causing slow sshd responses
> - **VNC connection may drop**: Same reason as above
> - **Existing terminals may become unresponsive**: High system load
> - **As long as the CPU usage remains above 50%, it means compilation is in progress. Do not force a restart!**
> - The compilation is complete when CPU usage drops below 5%

> 💡 You can monitor CPU usage on the "Instance Monitoring" page of the Alibaba Cloud console without needing an SSH connection

> ⚠️ **可能需要 restart after compilation**:
> - SSH may still not connect after compilation
> - Restart the instance in the Alibaba Cloud console
> - After restart, run `MAX_JOBS=6 pip install flash-attn --no-build-isolation -v` again
> - This time, the cached wheel will be used directly, and installation will complete in just a few seconds

**Verify installation**:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
# 应该输出 2.8.3 或类似版本
```

### 6.3 Installing Isaac-GR00T

```bash
cd ~

# 下载 Isaac-GR00T（注意：仓库在 NVIDIA 组织下，不是 NVIDIA-Omniverse）
# 方法1：git clone（国内可能较慢）
git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git

# 方法2：使用 gitclone 镜像
git clone --depth 1 https://gitclone.com/github.com/NVIDIA/Isaac-GR00T.git

# 方法3（推荐）：用 ghproxy 加速下载
curl -L -o isaac-groot.tar.gz https://ghproxy.cn/https://github.com/NVIDIA/Isaac-GR00T/archive/refs/heads/main.tar.gz
tar -xzf isaac-groot.tar.gz
mv Isaac-GR00T-main Isaac-GR00T

cd Isaac-GR00T
```

#### Install GR00T

> ⚠️ **Important**: Do not use `pip install -e .` directly, as it causes the flash-attn to be recompiled repeatedly, leading to freezing.
>
> **Reason**: The setup.py of GR00T lists flash-attn as a dependency. Even if it is already installed, pip will attempt to rebuild it.
> Using `--no-deps` can skip the automatic dependency resolution, and manual installation of the core package can avoid this issue.

```bash
# 使用 --no-deps 跳过依赖自动安装
pip install -e . --no-build-isolation --no-deps

# 手动安装核心依赖
# transformers 必须锁定 4.51.3，更高版本会导致模型加载报错
pip install transformers==4.51.3 safetensors einops peft diffusers tyro omegaconf pandas dm-tree termcolor av albumentations huggingface_hub deepspeed accelerate
pip install click datasets gymnasium lmdb matplotlib msgpack-numpy pyzmq wandb torchcodec

# 验证安装
python -c 'from gr00t.policy.gr00t_policy import Gr00tPolicy; print("GR00T loaded!")'
```

> 💡 **Regarding dependency version warnings**: After installation, you may see a bunch of `pip's dependency resolver` warnings indicating incompatible versions.
> This is because GR00T's setup.py locks very strict version numbers, but actually slightly newer versions can also work properly.
> **As long as the verification command output `GR00T loaded!` indicates success**, these warnings can be ignored.

> 💡 This approach avoids the issue of pip recompiling flash-attn. If flash-attn has been installed before, GR00T will automatically use it; if not, it will fall back to the standard attention.

### 6.4 Download GR00T Model

**Note**: For use in China, HuggingFace requires the use of an image.

```bash
conda activate groot
cd ~/Isaac-GR00T
export GROOT_MODEL_ROOT="${GROOT_MODEL_ROOT:-$HOME/models/groot}"
export GROOT_N16_MODEL="$GROOT_MODEL_ROOT/groot-n1.6-3b"
export GROOT_G1_MODEL="$GROOT_MODEL_ROOT/groot-n1.6-g1-pnp-apple-to-plate"
mkdir -p "$GROOT_MODEL_ROOT"

# Download the general GR00T-N1.6-3B model.
HF_ENDPOINT=https://hf-mirror.com python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('nvidia/GR00T-N1.6-3B', local_dir=os.environ['GROOT_N16_MODEL'])"

# Download the G1-specific evaluation model.
HF_ENDPOINT=https://hf-mirror.com python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('nvidia/GR00T-N1.6-G1-PnPAppleToPlate', local_dir=os.environ['GROOT_G1_MODEL'])"
```

### 6.5 Testing GR00T Inference

```bash
cd ~/Isaac-GR00T

# 创建测试脚本
cat > test_groot.py << 'EOF'
import os
import numpy as np
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

print('Loading GR00T N1.6...')
policy = Gr00tPolicy(
    model_path=os.environ['GROOT_N16_MODEL'],
    embodiment_tag=EmbodimentTag('gr1'),
    device='cuda',
)

obs = {
    'video': {
        'ego_view_bg_crop_pad_res256_freq20': np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8),
    },
    'state': {
        'left_arm': np.random.rand(1, 1, 7).astype(np.float32),
        'right_arm': np.random.rand(1, 1, 7).astype(np.float32),
        'left_hand': np.random.rand(1, 1, 6).astype(np.float32),
        'right_hand': np.random.rand(1, 1, 6).astype(np.float32),
        'waist': np.random.rand(1, 1, 3).astype(np.float32),
    },
    'language': {
        'task': [['pick up the red apple']],
    },
}

action = policy.get_action(obs)
print('Action output:')
for k, v in action[0].items():
    print(f'  {k}: shape={v.shape}')
print('GR00T Inference Success!')
EOF

# 运行测试
python test_groot.py
```

---

## 7. GR00T + MuJoCo Evaluation

### 7.1 Preparation Work

```bash
conda activate groot
cd ~/Isaac-GR00T

# 安装依赖
pip install uv
apt-get update && apt-get install -y libegl1-mesa-dev libglu1-mesa git-lfs
git lfs install

# 配置 git 使用 ghproxy 加速（关键！）
git config --global http.version HTTP/1.1
git config --global url."https://ghproxy.cn/https://github.com/".insteadOf "https://github.com/"

# 克隆 GR00T-WholeBodyControl 子模块（必须用 git clone，不能用 tar.gz）
mkdir -p external_dependencies
cd external_dependencies
rm -rf GR00T-WholeBodyControl
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git

cd ~/Isaac-GR00T

# 验证 LFS 文件已下载（应该是 100KB+ 而不是 131 字节）
ls -la ~/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/robot_model/model_data/g1/meshes/left_hip_pitch_link.STL
```

> ⚠️ **Important**:
> - You must use `git clone` instead of `curl` to download the tar.gz file, as the repository contains large Git LFS files (robot STL models)
> - The tar.gz download only provides a LFS pointer file (131 bytes), not the actual model file
> - If the STL file is only 131 bytes, MuJoCo simulation will report an error `Failed to determine STL storage representation`

### 7.2 Modify the setup script and run it

```bash
cd ~/Isaac-GR00T

# 注释掉 git submodule 和 git lfs pull 命令（我们已经手动处理了）
sed -i 's/^git submodule update/#git submodule update/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
sed -i 's/^git -C/#git -C/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh

# 注释掉 robosuite 的删除和 git clone（我们手动下载）
sed -i '26s/^rm/#rm/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
sed -i '27s/^git clone/#git clone/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh

# 手动下载 robosuite（setup 脚本里的 git clone 经常失败）
cd ~/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg
curl -L -o robosuite.tar.gz https://ghproxy.cn/https://github.com/xieleo5/robosuite/archive/refs/heads/leo/support_g1_locomanip.tar.gz
tar -xzf robosuite.tar.gz
mv robosuite-leo-support_g1_locomanip gr00trobosuite
rm robosuite.tar.gz

# 运行 setup 脚本
cd ~/Isaac-GR00T
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

The script will:
1. Create a uv virtual environment
2. Install dependencies such as robosuite, robocasa, and lerobot (downloading large packages like torch takes several minutes)
3. Verify the environment (`Imports OK` and `Env OK` should be visible)

> 💡 **Network issue troubleshooting**:
> - If the error occurs with `HTTP2 framing layer`: Ensure `git config --global http.version HTTP/1.1` has been executed.
> - If the error occurs with `Connection timed out`: Verify that ghproxy acceleration is configured.
> - If SSH connection fails: Reconnect and run the setup script again; downloaded packages will be cached.

### 7.3 Modify the code to enable real-time visualization

```bash
# 备份原文件
cp ~/Isaac-GR00T/gr00t/eval/rollout_policy.py ~/Isaac-GR00T/gr00t/eval/rollout_policy.py.bak

# 修改渲染模式
sed -i 's/os.environ\["MUJOCO_GL"\] = "egl"/os.environ["MUJOCO_GL"] = "glx"/' ~/Isaac-GR00T/gr00t/eval/rollout_policy.py
sed -i 's/onscreen=False/onscreen=True/' ~/Isaac-GR00T/gr00t/eval/rollout_policy.py
```

### 7.4 Run Evaluation

Two terminals are required:

**Terminal 1 - Start GR00T Server:**
```bash
cd ~/Isaac-GR00T
conda activate groot
export GROOT_G1_MODEL="${GROOT_G1_MODEL:-${GROOT_MODEL_ROOT:-$HOME/models/groot}/groot-n1.6-g1-pnp-apple-to-plate}"
python gr00t/eval/run_gr00t_server.py \
  --model-path "$GROOT_G1_MODEL" \
  --embodiment-tag UNITREE_G1 \
  --use-sim-policy-wrapper
```

Waiting to display `Server is ready and listening on tcp://127.0.0.1:5555`

**Terminal 2 - Start the evaluation Client (executed in VNC):**

> 💡 **VNC Copy-Paste Tips**:
> 1. Click the expand menu on the left side of the noVNC web page to find "Clipboard".
> 2. Paste the following command into the clipboard text box.
> 3. Open the terminal in the VNC desktop (right-click the desktop → Terminal).
> 4. Use `Ctrl+Shift+V` to paste the command in the terminal.
> 5. Press Enter to execute it.
>
> ⚠️ **Note**: The command has been combined into a single line to avoid hidden characters during multi-line copying that could cause errors (such as `invalid int value: '5555~'`).

```bash
cd ~/Isaac-GR00T && export DISPLAY=:1 && gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python gr00t/eval/rollout_policy.py --n_episodes 3 --max_episode_steps 500 --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc --n_action_steps 20 --n_envs 1 --policy_client_host 127.0.0.1 --policy_client_port 5555
```

A MuJoCo window will appear on the VNC desktop, displaying in real time how the G1 robot performs the "pick apple and place on plate" task.

---

## 8. Common Issues and Pitfalls Records

### Q1: How to install NVIDIA Container Toolkit?

**Recommended method**: Directly download the deb package. See Section 3.4. This method is the most stable and reproducible.

**Alternative method** (if using the apt source):
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
```

Note: nvidia.github.io may be blocked in China.

### Q2: WebRTC Livestream Black Screen

**Solution**: Switch to VNC remote desktop, which is more stable.

### Q3: Isaac Lab cloning failed

**Symptoms**: `fatal: unable to access 'https://github.com/isaac-sim/IsaacLab.git/'`

**Solution**: Copy the downloaded file from the domestic mirror into the container after using it on the host machine.
```bash
# 在宿主机
cd ~
git clone --depth 1 --branch v1.4.1 https://gitclone.com/github.com/isaac-sim/IsaacLab.git
docker cp IsaacLab isaac-sim-gui:/isaac-sim/
```

### Q4: Robomimic installation failed

**Symptoms**: `Failed to build 'robomimic'` or `egl_probe` compilation errors

**Solution**:
```bash
# 在宿主机下载
cd ~
wget https://github.com/ARISE-Initiative/robomimic/archive/refs/heads/master.zip
unzip master.zip
mv robomimic-master robomimic
docker cp robomimic isaac-sim-gui:/tmp/

# 在容器内安装 EGL 库和编译工具
docker exec -it isaac-sim-gui bash
apt update
apt install -y build-essential cmake pkg-config
apt install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev

# 安装 robomimic
cd /tmp/robomimic
/isaac-sim/python.sh -m pip install -e .
```

### Q5: Setup.py configuration error

**Symptoms**: `IndentationError` or `Could not find a version that satisfies the requirement rsl-rl`

**Solution**: Fix the configuration file according to Section 5.5 of the tutorial
```bash
cd /isaac-sim/IsaacLab
sed -i '46s/"rsl-rl@git+https:\/\/github.com\/leggedrobotics\/rsl_rl.git"/"rsl-rl-lib==2.3.0"/' source/extensions/omni.isaac.lab_tasks/setup.py
sed -i '53,54d' source/extensions/omni.isaac.lab_tasks/setup.py
```

### Q6: Error in the latest version of Isaac Lab

**Reason**: The latest version requires Isaac Sim 4.5+.
**Solution**: Use `git clone --depth 1 --branch v1.4.1` to clone the specified version directly.

### Q7: rsl-rl version conflict

**Solution**: Fixed in Section 5.5 of the tutorial, use `rsl-rl-lib==2.3.0`

### Q8: HuggingFace cannot be accessed

**Solution**: Use the mirror `HF_ENDPOINT=https://hf-mirror.com`

### Q9: GR00T model loading error - shape mismatch

**Reason**: The old version of the N1-2B model was downloaded, but the code is from the N1.6 version.
**Solution**: Download the `nvidia/GR00T-N1.6-3B` model.

### Q10: Flash-Attn compilation issues

#### Symptom 1: SSH/VNC connection drops during compilation, but CPU usage is high.

**This is normal!** Compilation consumes a lot of resources, causing the system to respond slowly.
- Monitor CPU usage in the Alibaba Cloud console.
- Wait until CPU drops below 5% to consider compilation complete.
- **Do not force a server restart!** Otherwise, the compilation will be futile.

#### Symptom 2: CPU utilization is close to 0%, compilation stalls

Indicated that the compilation process exited abnormally and needs to be retried:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
MAX_JOBS=4 pip install flash-attn --no-build-isolation -v
```

#### Symptom 3: `flash-attn` disappears after restarting

Compilation is interrupted and needs to be recompiled. If you don't want to wait, you can skip flash-attn:
```bash
cd ~/Isaac-GR00T
pip install -e . --no-deps
pip install transformers==4.51.3 safetensors einops peft diffusers tyro omegaconf pandas dm-tree termcolor av albumentations huggingface_hub deepspeed accelerate
pip install click datasets gymnasium lmdb matplotlib msgpack-numpy pyzmq wandb torchcodec
```

#### Symptom 4: Installation succeeds, but there are many version warnings.

```
gr00t 0.1.0 requires flash-attn==2.7.4.post1, but you have flash-attn 2.8.3 which is incompatible.
...
```

**Can be ignored!** This is because GR00T's setup.py locks very strict version numbers, but actually a slightly newer version can also work properly. As long as `python -c "import flash_attn"` doesn't report any errors, it's fine.

> 💡 flash-attn is a performance optimization component and is not required. Skipping it does not affect the functionality of GR00T.

### Q11: MuJoCo cannot display the window

**Solution**:
1. Ensure VNC is running: `netstat -tlnp | grep 5901`
2. Set `export DISPLAY=:1`
3. Replace `MUJOCO_GL` in the code with `glx`, and `onscreen` with `True`

### Q12: Unable to connect after VNC starts

**Symptoms**: websockify logs show `Connection refused`

**Solution**:
```bash
# 杀掉旧进程
vncserver -kill :1
pkill -9 websockify

# 修复 xstartup 配置
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF
chmod +x ~/.vnc/xstartup

# 重启 VNC（注意 -localhost no 参数）
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5901 > /dev/null 2>&1 &
```

### Q13: Preemptible instance released

**Suggestions**:
- Set a reasonable maximum price
- Back up important data promptly
- Use a data disk to store models and code

### Q14: Insufficient disk space

**Solutions**:
- Clean up Docker cache: `docker system prune -a`
- Clean up pip cache: `pip cache purge`
- Clean up conda cache: `conda clean -a`
- It is recommended to use a system disk of 150GB+ or mount a separate data drive

### Q15: VNC screen lagging

**Note**: VNC transmission of 3D simulation images will experience delays and frame loss, which is normal. VNC is mainly used for "verifying operational status," not for smooth manipulation.

**Advanced Solutions**:
- Use NoMachine (better performance)
- Record the video and download it locally for viewing

### Q16: GR00T-WholeBodyControl clone failed

**Symptoms**:
- `fatal: repository 'https://github.com/NVIDIA/GR00T-WholeBodyControl.git/' not found`
- `error: RPC failed; curl 16 Error in the HTTP2 framing layer`

**Reason**: The sub-module address is under `NVlabs`, not `NVIDIA`; or there are network issues.

**Review**: Section 7.1

**Solution**:
```bash
cd ~/Isaac-GR00T/external_dependencies

# 方法1：正确地址
git clone --depth 1 https://github.com/NVlabs/GR00T-WholeBodyControl.git

# 方法2（推荐）：用 ghproxy 加速
curl -L -o wbc.tar.gz https://ghproxy.cn/https://github.com/NVlabs/GR00T-WholeBodyControl/archive/refs/heads/main.tar.gz
tar -xzf wbc.tar.gz
mv GR00T-WholeBodyControl-main GR00T-WholeBodyControl
rm wbc.tar.gz
```

### Q17: Transformers version is incompatible

**Symptoms**: `AttributeError: 'Eagle3_VLConfig' object has no attribute '_attn_implementation_autoset'`

**Reason**: The transformers version is too new.

**Review**: Section 6.3, confirm that `transformers==4.51.3` was used during installation

**Solution**:
```bash
pip install transformers==4.51.3
```

---

## Appendix: Quick Reference of Common Commands

```bash
# 启动 VNC
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5901 > /dev/null 2>&1 &

# 进入 Isaac Sim 容器
docker start isaac-sim-gui
docker exec -it isaac-sim-gui bash

# 在容器内运行 Isaac Lab demo
export DISPLAY=:1
cd /isaac-sim/IsaacLab
./isaaclab.sh -p source/standalone/demos/arms.py

# 激活 GR00T 环境
conda activate groot

# 启动 GR00T Server
export GROOT_G1_MODEL="${GROOT_G1_MODEL:-${GROOT_MODEL_ROOT:-$HOME/models/groot}/groot-n1.6-g1-pnp-apple-to-plate}"
python gr00t/eval/run_gr00t_server.py \
  --model-path "$GROOT_G1_MODEL" \
  --embodiment-tag UNITREE_G1 \
  --use-sim-policy-wrapper
```

---

## Reference Links

- [ Isaac Sim official documentation ](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [ Isaac Lab GitHub ](https://github.com/isaac-sim/IsaacLab)
- [ Isaac-GR00T GitHub ](https://github.com/NVIDIA/Isaac-GR00T)
- [ GR00T-WholeBodyControl GitHub ](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [ GR00T model on HuggingFace ](https://huggingface.co/nvidia/GR00T-N1.6-3B)

---

* Tutorial completion time: January 2026 *
* Test environment: Alibaba Cloud ecs.gn7i-c16g1.4xlarge (A10 24GB) *
