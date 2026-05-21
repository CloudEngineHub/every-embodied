# Isaac Sim 本地与云端配置教程

学完这一章后，大家可以判断自己的电脑或云服务器是否适合运行 Isaac Sim，并完成三类常见配置：Windows 本地工作站安装、Linux 工作站或云服务器安装、以及 micromamba/venv 方式的 Python 环境安装。本文还会说明 Isaac Lab 应该接在哪一步，以及什么时候才需要进入 GR00T 复现。

这里先给出核心判断：如果只是学习 Isaac Sim 的 GUI、机器人导入、传感器和基础场景，Windows 本地工作站完全可以作为入门环境；如果要跑 Isaac Lab 强化学习训练、Docker、远程桌面或多人复现，Linux 工作站和云 GPU 服务器更稳。云服务器和本地工作站的命令差别不大，本质区别是云服务器需要额外处理远程显示、端口、安全组和数据持久化。

## 1. 先判断该选哪条安装路线

| 场景 | 推荐路线 | 说明 |
| --- | --- | --- |
| Windows 电脑，只想先打开 Isaac Sim GUI 学习 | Workstation 安装包 | 最容易给初学者解释，直接启动 App Selector。 |
| Windows 电脑，希望用 Python 环境管理 Isaac Sim | micromamba/venv + pip | 适合后续接 Isaac Lab，但安装包较大，首次拉取扩展较慢。 |
| Ubuntu 工作站，有显示器 | Workstation 安装包或 pip | Workstation 更直观，pip 更利于和 Isaac Lab 组合。 |
| Ubuntu 云服务器，没有显示器 | Docker 或 pip + 远程桌面/直播流 | 要额外处理 VNC、WebRTC、NICE DCV 或 X11。 |
| 目标是 Isaac Lab 训练 | pip Isaac Sim + Isaac Lab 源码 | Isaac Lab 官方推荐先装 Isaac Sim pip 包，再装 Isaac Lab。 |
| 目标是 GR00T 复现 | 先读版本声明，再进入 GR00T 教程 | GR00T 与 Isaac 的依赖很重，建议单独环境，不要和本地入门环境混用。 |

Isaac Sim 是一个高保真仿真平台，不是普通 Python 小库。它依赖 NVIDIA RTX GPU、驱动、Omniverse Kit、PhysX、USD 资产、图形渲染和大量扩展。大家安装时不要只盯着 `pip install` 是否成功，更重要的是确认 GPU 驱动、显存、图形界面和扩展缓存都能正常工作。

## 2. 下载入口总表

为了让大家能一步一步复现，这里把本章会用到的下载入口集中列出来。Isaac Sim 版本更新较快，如果大家后续使用 5.2 或更高版本，应优先进入官方下载页确认文件名和版本号，再替换命令里的 URL。

| 内容 | 下载或获取方式 | 用在什么场景 |
| --- | --- | --- |
| Isaac Sim 5.1 Windows Workstation | [isaac-sim-standalone-5.1.0-windows-x86_64.zip](https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-windows-x86_64.zip) | Windows 本地 GUI 入门 |
| Isaac Sim 5.1 Linux Workstation | [isaac-sim-standalone-5.1.0-linux-x86_64.zip](https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip) | Ubuntu 工作站 GUI 入门 |
| Isaac Sim 5.1 Python 包 | `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com` | micromamba / venv / Isaac Lab |
| Isaac Sim 5.1 Docker 镜像 | `docker pull nvcr.io/nvidia/isaac-sim:5.1.0` | Linux 云服务器、容器隔离 |
| WebRTC Streaming Client Windows | [isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe](https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe) | 远程连接云端 Isaac Sim |
| WebRTC Streaming Client Linux | [isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage](https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage) | Linux 客户端连接远程 Isaac Sim |
| Isaac Sim 完整资产包 | [complete 001](https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.001.zip)、[complete 002](https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.002.zip)、[complete 003](https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.003.zip) | 离线资产、本地缓存、内网环境 |
| NVIDIA Container Toolkit | [官方安装文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | Docker 访问 GPU |
| Isaac Lab 源码 | `git clone https://github.com/isaac-sim/IsaacLab.git` | Isaac Lab 训练和任务构建 |

最稳的初学者路线是先下载 Workstation 包，把 Isaac Sim GUI 跑起来；如果后续要做 Isaac Lab，再改用 pip 环境。云服务器路线则先配置 NVIDIA 驱动和 Docker GPU runtime，再拉取 `nvcr.io/nvidia/isaac-sim:5.1.0`。

## 3. 硬件和系统要求

根据 NVIDIA Isaac Sim 5.1 官方要求，x86_64 平台支持 Ubuntu 22.04/24.04 和 Windows 10/11。官方最低配置表中给出的内存是 32GB RAM，存储至少 50GB SSD，显卡侧重点是带 RT Core 的 NVIDIA RTX GPU。官方还提醒，复杂场景、传感器和大规模训练会继续增加 RAM 与 VRAM 需求，低于 16GB 显存的 GPU 可能无法运行复杂渲染场景。

这意味着，RTX 3060 Laptop 6GB 这类机器可以用于轻量学习和兼容性验证，但不适合作为复杂 Isaac Lab 训练、高清传感器、多机器人场景或 GR00T 大模型复现的主力环境。如果学习者不想租服务器，可以先在本机完成 GUI 和基础场景；等到需要训练、合成数据或大模型推理时，再切到云服务器。

## 4. 安装前检查

在 Windows PowerShell 中，先确认 NVIDIA 驱动、Python 和磁盘空间。Isaac Sim 5.x 的 pip 安装需要 Python 3.11；如果本机默认是 Python 3.12 或 3.10，请单独建环境，不要直接装进系统 Python。

```powershell
nvidia-smi
python --version
python -m pip --version
Get-PSDrive C
```

如果准备走 pip 路线，还可以先检查 PyPI 和 NVIDIA PyPI 源是否能看到 Isaac Sim 包。这个命令只查询包索引，不会下载完整安装包。

```powershell
python -m pip index versions isaacsim --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com
```

在 Linux 上，额外检查 GLIBC。Isaac Sim 5.x pip 包要求 GLIBC 2.35 或更高，Ubuntu 22.04 通常满足，Ubuntu 20.04 默认 GLIBC 2.31，不建议直接走 pip 路线。

```bash
nvidia-smi
python3 --version
ldd --version
df -h
```

Checkpoint 1：如果 `nvidia-smi` 看不到 NVIDIA GPU，先不要继续安装 Isaac Sim。此时问题通常是显卡驱动、云服务器 GPU 实例规格、Docker GPU runtime 或远程桌面环境，而不是 Isaac Sim 本身。

## 5. Windows 本地 Workstation 安装

Windows 初学者优先推荐 Workstation 安装包，因为它最接近普通桌面软件的使用方式。大家可以从浏览器下载，也可以在 PowerShell 中直接下载。安装包大约几十 GB，建议先确认磁盘空间，再解压到 `C:\isaacsim` 这类短路径，减少 Windows 长路径导致的安装和缓存问题。

第一步，下载 Windows standalone 包：

```powershell
New-Item -ItemType Directory -Force C:\isaacsim-downloads
cd C:\isaacsim-downloads
Invoke-WebRequest `
  -Uri "https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-windows-x86_64.zip" `
  -OutFile "isaac-sim-standalone-5.1.0-windows-x86_64.zip"
```

如果 `Invoke-WebRequest` 下载很慢，可以把上面的 URL 复制到浏览器、IDM、aria2 或学校网络下载器中下载，文件名保持不变即可。

第二步，解压并执行安装后处理：

```powershell
mkdir C:\isaacsim
cd C:\isaacsim-downloads
tar -xvzf "isaac-sim-standalone-5.1.0-windows-x86_64.zip" -C C:\isaacsim
cd C:\isaacsim
.\post_install.bat
.\isaac-sim.selector.bat
```

启动 App Selector 后，选择 Isaac Sim Full 并点击 START。第一次启动会预热 shader cache 并拉取扩展，空白窗口持续几分钟是正常现象。进入主界面后，建议先做最小图形验证：选择 `Create > Environment > Simple Room`，再选择 `Create > Robots > Franka Emika Panda Arm`，最后点击左侧播放按钮。如果场景能显示并运行，说明本地 GUI、渲染、基础资产和物理仿真链路已经打通。

Checkpoint 2：这个验证只证明 Isaac Sim 可以在本机打开并运行基础场景，不代表显存足够做大规模训练，也不代表 ROS2、Isaac Lab 或 GR00T 已经配置完成。

## 6. Windows micromamba / venv 安装

如果后续要接 Isaac Lab，或者希望把 Isaac Sim 当成 Python 包管理，可以使用 micromamba 或 venv。这里推荐单独创建 `env_isaacsim`，不要装进已经用于 LeRobot、OpenVLA、PyTorch 实验的环境。

```powershell
micromamba create -n env_isaacsim python=3.11 -y
micromamba activate env_isaacsim
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

Isaac Sim 首次运行需要接受 NVIDIA Omniverse EULA。PowerShell 中可以临时设置环境变量：

```powershell
$env:OMNI_KIT_ACCEPT_EULA="YES"
isaacsim isaacsim.exp.compatibility_check
isaacsim
```

如果只是想先检查机器是否满足要求，可以先安装更小的兼容性检查 bundle：

```powershell
python -m pip install "isaacsim[compatibility-check]==5.1.0" --extra-index-url https://pypi.nvidia.com
$env:OMNI_KIT_ACCEPT_EULA="YES"
isaacsim isaacsim.exp.compatibility_check
```

Checkpoint 3：`isaacsim isaacsim.exp.compatibility_check` 会启动 Isaac Sim 兼容性检查应用。它比完整 GUI 更适合作为安装前体检，但它不能替代后续的完整场景验证。

## 7. Linux 工作站和云服务器安装

Linux 本地工作站可以直接使用 Workstation 安装包。第一步，下载安装包：

```bash
mkdir -p ~/isaacsim-downloads
cd ~/isaacsim-downloads
wget -c https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
```

第二步，解压并启动 App Selector：

```bash
mkdir -p ~/isaacsim
cd ~/isaacsim-downloads
unzip "isaac-sim-standalone-5.1.0-linux-x86_64.zip" -d ~/isaacsim
cd ~/isaacsim
./post_install.sh
./isaac-sim.selector.sh
```

如果是云服务器，通常没有物理显示器，Workstation GUI 需要配合 VNC、NICE DCV、X11 转发或 Isaac Sim Livestream。此时建议大家把“云服务器”和“本地 Linux 工作站”理解成同一套 Isaac 安装，只是云服务器多了远程显示和端口暴露问题。端口不要直接对公网开放，优先使用 SSH 隧道或云厂商安全组限制来源 IP。

如果走 pip 路线，Linux 命令如下：

```bash
micromamba create -n env_isaacsim python=3.11 -y
micromamba activate env_isaacsim
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES
isaacsim isaacsim.exp.compatibility_check
isaacsim
```

如果 `ldd --version` 显示 GLIBC 低于 2.35，不要在旧系统上硬装 pip 版本。更稳妥的做法是换 Ubuntu 22.04/24.04，或者使用官方二进制安装包和容器路线。

## 8. Linux Docker 安装

Docker 更适合 Linux 云服务器、多人复现和环境隔离。NVIDIA 官方明确说明 Isaac Sim container 只支持 Linux。Windows 本地如果想用 Isaac Sim GUI，优先用 Workstation 或 pip，不建议把 Windows Docker 当作首选教学路线。

第一步，安装 Docker。如果云服务器镜像已经预装 Docker，可以跳过这一段：

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
docker --version
```

第二步，安装 NVIDIA Container Toolkit。下面是 Ubuntu 上的官方安装方式，其他发行版以 NVIDIA Container Toolkit 官方文档为准：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

第三步，验证容器内能看到 GPU：

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

第四步，拉取 Isaac Sim 5.1 镜像。镜像来自 NVIDIA NGC，首次拉取可能需要登录 NGC；如果提示认证失败，先在 [NGC](https://ngc.nvidia.com/) 注册并生成 API Key，再执行 `docker login nvcr.io`。

```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/isaac-sim:5.1.0
```

建议给 Isaac Sim 缓存、日志和用户数据挂载独立目录，避免容器删除后缓存丢失：

```bash
mkdir -p ~/docker/isaac-sim/cache/main
mkdir -p ~/docker/isaac-sim/cache/computecache
mkdir -p ~/docker/isaac-sim/config
mkdir -p ~/docker/isaac-sim/data
mkdir -p ~/docker/isaac-sim/logs
mkdir -p ~/docker/isaac-sim/pkg
sudo chown -R 1234:1234 ~/docker/isaac-sim
```

启动交互容器：

```bash
docker run --name isaac-sim --entrypoint bash -it --gpus all --rm --network=host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  -u 1234:1234 \
  nvcr.io/nvidia/isaac-sim:5.1.0
```

进入容器后先跑兼容性检查：

```bash
./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```

Checkpoint 4：如果这里看到系统检查通过，说明 Docker、NVIDIA Container Toolkit、GPU 驱动和 Isaac Sim 容器至少能在 headless 模式下工作。它仍然不等价于远程 GUI 或 Livestream 已经配置成功。

## 9. 远程云服务器怎么打开画面

云服务器和本地工作站的 Isaac Sim 本体是一套东西，但云端多了“如何看到画面”的问题。最简单的教学方案是 VNC 或 NICE DCV；如果使用 Isaac Sim 自带 livestream，可以下载 WebRTC Streaming Client。

Windows 本地电脑连接云端 Isaac Sim 时，下载这个客户端：

```powershell
New-Item -ItemType Directory -Force C:\isaacsim-downloads
Invoke-WebRequest `
  -Uri "https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe" `
  -OutFile "C:\isaacsim-downloads\isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe"
```

Linux 客户端下载这个 AppImage：

```bash
mkdir -p ~/isaacsim-downloads
cd ~/isaacsim-downloads
wget -c https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage
chmod +x isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage
```

远程显示的配置和云厂商安全组强相关。原则上不要把 VNC 或 livestream 端口直接暴露给公网，优先使用 SSH 隧道或只允许自己的固定 IP 访问。

## 10. Isaac Lab 配置

Isaac Lab 是建立在 Isaac Sim 之上的机器人学习框架，适合强化学习、模仿学习、任务构建和并行仿真。推荐流程是先让 Isaac Sim 可以启动，再安装 Isaac Lab。不要在 Isaac Sim 还打不开时直接装 Isaac Lab，否则后续报错会很难定位。

以下命令以 Isaac Lab 2.3.x 和 Isaac Sim 5.1 为主线。Isaac Lab 本体从 GitHub 获取，大家可以先下载源码，再切到对应 release。建议进入 Isaac Lab 官方 release 页面确认当前版本对应的 Isaac Sim 版本，再决定是否升级。

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.0
```

Linux 上安装基础编译依赖，然后安装 Isaac Lab 扩展：

```bash
sudo apt update
sudo apt install -y cmake build-essential
./isaaclab.sh --install
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

Windows 上在已激活的 Isaac Python 环境中执行：

```powershell
isaaclab.bat --install
isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
```

官方验证脚本会启动一个黑色 viewport 的空场景。这个结果看起来很简单，但它证明 Isaac Lab 可以调用当前 Python 环境、找到 Isaac Sim、加载基础扩展并打开仿真应用。

## 11. ROS2 集成怎么选

Isaac Sim 5.1 支持 ROS2 Humble 和 Jazzy。Ubuntu 22.04 推荐 Humble，Ubuntu 24.04 推荐 Jazzy；Windows 10/11 场景下，官方文档建议 ROS2 Humble 通过 WSL 使用。

初学者不建议把 ROS2 和 Isaac Sim 安装放在第一天一起做。更稳的顺序是先打开 Isaac Sim，再安装 ROS2，再启用 ROS2 Bridge。默认消息类型如 `std_msgs`、`geometry_msgs`、`nav_msgs` 通常比较容易跑通；如果需要自定义 ROS2 interface，需要同时考虑 Isaac Sim Python 3.11 和系统 ROS Python 版本之间的兼容关系。

## 12. 什么时候进入 GR00T 教程

GR00T 不是 Isaac Sim 的入门配置，而是更靠后的机器人基础模型实践。仓库中的 [阿里云部署 Isaac Lab + GR00T 完整教程](02阿里云部署Isaac-Lab-GR00T完整教程.md) 已经锁定了一套历史可复现版本：Isaac Sim 4.2.0、Isaac Lab v1.4.1、GR00T N1.6、PyTorch 2.5.1 和 CUDA 12.1。这个版本组合与本文的 Isaac Sim 5.1 主线不同。

如果大家只是本地学习 Isaac Sim，不需要立刻装 GR00T。如果目标是复现 GR00T，建议单独开一个云服务器或单独环境，按照原教程版本执行，不要把 GR00T 依赖安装进 `env_isaacsim`。这样做可以避免 PyTorch、CUDA、transformers、flash-attn 和 Isaac 扩展互相污染。

## 13. 常见问题

Isaac Sim 安装失败时，先按层排查。第一层是硬件和驱动：`nvidia-smi` 是否正常，驱动是否足够新，GPU 是否有 RT Core，显存是否过低。第二层是系统：Windows 是否开启长路径支持，Linux GLIBC 是否满足 pip 包要求，磁盘是否至少预留几十 GB。第三层才是 Python：是否使用 Python 3.11，是否把 Isaac 装进了干净环境，是否使用了 NVIDIA PyPI 源。

如果首次启动很慢，不要马上关闭窗口。Isaac Sim 第一次启动会拉取扩展、编译 shader cache 和初始化用户配置，5 到 10 分钟都可能是正常现象。只有当日志明确显示驱动错误、扩展下载失败、EULA 未接受或进程崩溃时，才按错误信息处理。

如果 Windows 上 pip 安装报路径过长，先开启长路径支持，或者改用更短的环境路径。Workstation 安装时也建议使用 `C:\isaacsim` 这种短路径，而不是层级很深的个人文档目录。

如果云服务器上 GUI 打不开，先区分 Isaac 是否能 headless 启动。如果兼容性检查能通过，问题多半在 VNC、WebRTC、DISPLAY、Xauthority 或安全组端口，而不是 Isaac Sim 本体。此时可以先用容器 headless 命令验证，再处理远程显示。

## 14. 本教程的轻量验证记录

本文没有在本仓库机器上完整下载 40GB 级 Isaac Sim 安装包，因为这会产生较大的时间和磁盘成本。已完成的轻量验证包括：本机 `nvidia-smi` 可识别 NVIDIA GeForce RTX 3060 Laptop GPU，驱动版本 581.95，显存 6144 MiB；当前 Python 为 3.11.15；micromamba 版本为 2.3.0；通过 PyPI 与 NVIDIA PyPI 索引可查询到 `isaacsim` 的 5.1.0.0 和 5.0.0.0 包版本。

这组验证证明本机具备最基本的 NVIDIA 驱动、Python 3.11 和包索引访问条件，但不证明本机 6GB 显存足够运行复杂 Isaac 场景。真正交付给学习者时，仍建议按本文 Checkpoint 逐步确认。

## 参考资料

- [NVIDIA Isaac Sim 5.1 Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html)
- [NVIDIA Isaac Sim 5.1 Downloads](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
- [NVIDIA Isaac Sim 5.1 Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [NVIDIA Isaac Sim 5.1 Workstation Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)
- [NVIDIA Isaac Sim 5.1 Python Environment Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [NVIDIA Isaac Sim 5.1 Container Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html)
- [NVIDIA Isaac Sim 5.1 ROS 2 Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html)
- [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Isaac Lab 2.3 Installation using Isaac Sim Pip Package](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
