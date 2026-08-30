# Isaac Sim Local and Cloud Configuration Tutorial

After completing this chapter, you can determine whether your computer or cloud server is suitable for running Isaac Sim, and set up three common configurations: installation on a Windows local workstation, installation on a Linux workstation or cloud server, and installation of a Python environment using micromamba/venv. This article will also explain where Isaac Lab should be used and when it is necessary to enter GR00T for reproduction.

Here is the key judgment: If you only want to learn about Isaac Sim’s GUI, robot import, sensors, and basic scenarios, a Windows local workstation can serve as an excellent starting environment. However, if you plan to conduct reinforcement learning training with Isaac Lab, use Docker, enable remote desktop, or perform reproduction by multiple people, a Linux workstation and a cloud GPU server are more suitable. The commands for cloud servers and local workstations are not very different; the main difference is that cloud servers require additional handling for remote display, ports, security groups, and data persistence.

## 1. First, determine which installation path to choose

| Scenario | Recommended Approach | Explanation |
| --- | --- | --- |
| Windows computer, just want to open the Isaac Sim GUI to learn | Workstation installation package | Most easy for beginners; directly launch App Selector. |
| Windows computer, want to manage Isaac Sim using a Python environment | micromamba/venv + pip | Suitable for later integration with Isaac Lab, but the installation package is large and initial download of extensions is slow. |
| Ubuntu workstation with monitor | Workstation installation package or pip | Workstation is more intuitive; pip is better for integration with Isaac Lab. |
| Ubuntu cloud server without monitor | Docker or pip + remote desktop/live stream | Need to handle VNC, WebRTC, NICE DCV, or X11 separately. |
| Goal is to train with Isaac Lab | pip Isaac Sim + Isaac Lab source code | The official recommendation for Isaac Lab is to install the Isaac Sim pip package first, then Isaac Lab. |
| Goal is reproduction of GR00T | Read the version statement first, then follow the GR00T tutorial | GR00T has heavy dependencies on Isaac; it is recommended to use a separate environment and not mix with the local beginner environment. |

Isaac Sim is a high-fidelity simulation platform, not an ordinary Python library. It relies on NVIDIA RTX GPU, drivers, Omniverse Kit, PhysX, USD assets, graphics rendering, and numerous extensions. When installing, don’t just focus on whether `pip install` is installed successfully; more importantly, ensure that the GPU driver, memory, graphical interface, and extension cache all function properly.

## 2. Download Entry Summary Table

To enable everyone to reproduce the process step by step, we have listed here all the download links that will be used in this chapter. The Isaac Sim version is updated frequently. If you use version 5.2 or later in the future, please go to the official download page first to confirm the file name and version number, and then replace the URL in the command.

| Content | Download or acquisition method | Usage scenario |
| --- | --- | --- |
| Isaac Sim 5.1 Windows Workstation | [isaac-sim-standalone-5.1.0-windows-x86_64.zip](https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-windows-x86_64.zip) | Introduction to Windows local GUI |
| Isaac Sim 5.1 Linux Workstation | [isaac-sim-standalone-5.1.0-linux-x86_64.zip](https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip) | Introduction to Ubuntu workstation GUI |
| Isaac Sim 5.1 Python package | `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com` | micromamba / venv / Isaac Lab |
| Isaac Sim 5.1 Docker image | `docker pull nvcr.io/nvidia/isaac-sim:5.1.0` | Linux cloud server, container isolation |
| WebRTC Streaming Client Windows | [isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe](https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe) | Remote connection to cloud Isaac Sim |
| WebRTC Streaming Client Linux | [isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage](https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage) | Linux client connecting to remote Isaac Sim |
| Complete Isaac Sim asset package | [complete 001](https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.001.zip), [complete 002](https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.002.zip), [complete 003](https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.003.zip) | Offline assets, local caching, intranet environment |
| NVIDIA Container Toolkit | [Official installation documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | Docker access to GPU |
| Isaac Lab source code | `git clone https://github.com/isaac-sim/IsaacLab.git` | Isaac Lab training and task building |

The most stable starting path is to download the Workstation package first and run the Isaac Sim GUI; if you later want to use Isaac Lab, switch to the pip environment. For the cloud server approach, configure the NVIDIA driver and Docker GPU runtime first, and then pull `nvcr.io/nvidia/isaac-sim:5.1.0`.

## 3. Hardware and System Requirements

According to the official requirements of NVIDIA Isaac Sim 5.1, the x86_64 platform supports Ubuntu 22.04/24.04 and Windows 10/11. The minimum configuration listed in the official table indicates 32GB RAM and at least 50GB SSD storage. The graphics card should be a NVIDIA RTX GPU with RT Core. The official also notes that complex scenarios, sensors, and large-scale training will further increase the demand for RAM and VRAM. GPUs with less than 16GB of video memory may not be able to handle complex rendering scenarios.

This means that machines like the RTX 3060 Laptop 6GB can be used for light learning and compatibility testing, but they are not suitable as primary environments for complex Isaac Lab training, high-definition sensors, multi-robot scenarios, or reproduction of GR00T large models. If learners do not want to rent servers, they can first complete the GUI and basic scenarios on their own machine; when training, synthetic data generation, or large model inference is required, they can switch to a cloud server.

## 4. Pre-installation Checks

In Windows PowerShell, first verify the NVIDIA driver, Python, and disk space. The pip installation for Isaac Sim 5.x requires Python 3.11; if the default Python on your machine is 3.12 or 3.10, create a separate environment instead of installing it directly into the system Python.

```powershell
nvidia-smi
python --version
python -m pip --version
Get-PSDrive C
```

If you plan to use the pip method, you can first check whether the Isaac Sim package is available in the PyPI and NVIDIA PyPI repositories. This command only queries the package index and does not download the complete installation package.

```powershell
python -m pip index versions isaacsim --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com
```

On Linux, check GLIBC additionally. The Isaac Sim 5.x pip package requires GLIBC 2.35 or higher. Ubuntu 22.04 usually meets this requirement, while Ubuntu 20.04 defaults to GLIBC 2.31. It is not recommended to use pip directly.

```bash
nvidia-smi
python3 --version
ldd --version
df -h
```

Checkpoint 1: If the `nvidia-smi` does not detect the NVIDIA GPU, do not proceed with installing Isaac Sim. Usually, the issue is related to the graphics card driver, cloud server GPU instance specifications, Docker GPU runtime, or remote desktop environment, rather than Isaac Sim itself.

## 5. Installation on Windows Local Workstation

For beginners using Windows, the Workstation installation package is highly recommended because it behaves like ordinary desktop software. You can download it in a browser or directly in PowerShell. The package is several dozen GB, so check disk space first and use a short directory under your user profile to avoid long-path and cache issues.

Step 1: Download the Windows standalone package:

```powershell
$ISAAC_SIM_DOWNLOADS = Join-Path $env:USERPROFILE "Downloads\isaacsim"
$ISAAC_SIM_ROOT = Join-Path $env:USERPROFILE "isaacsim"
New-Item -ItemType Directory -Force $ISAAC_SIM_DOWNLOADS
Set-Location $ISAAC_SIM_DOWNLOADS
Invoke-WebRequest `
  -Uri "https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-windows-x86_64.zip" `
  -OutFile "isaac-sim-standalone-5.1.0-windows-x86_64.zip"
```

If `Invoke-WebRequest` downloads slowly, you can copy the above URL into a browser, IDM, aria2, or school network downloader to download it, and keep the file name unchanged.

Step 2: Unzip and execute the post-installation processing:

```powershell
New-Item -ItemType Directory -Force $ISAAC_SIM_ROOT
Set-Location $ISAAC_SIM_DOWNLOADS
tar -xvzf "isaac-sim-standalone-5.1.0-windows-x86_64.zip" -C $ISAAC_SIM_ROOT
Set-Location $ISAAC_SIM_ROOT
.\post_install.bat
.\isaac-sim.selector.bat
```

After starting App Selector, select Isaac Sim Full and click START. The first launch will preheat the shader cache and load extensions. It is normal for a blank window to persist for a few minutes. After entering the main interface, it is recommended to perform a minimum graphics verification: select `Create > Environment > Simple Room`, then `Create > Robots > Franka Emika Panda Arm`, and finally click the play button on the left. If the scene appears and runs, it indicates that the local GUI, rendering, basic assets, and physical simulation chain are connected.

Checkpoint 2: This verification only proves that Isaac Sim can be launched and run with basic scenarios on the local machine. It does not indicate that there is sufficient VRAM for large-scale training, nor does it mean that ROS2, Isaac Lab, or GR00T have been properly configured.

## 6. Installation via Windows micromamba / venv

If you want to integrate Isaac Lab later, or prefer to use Isaac Sim as a Python package manager, you can use micromamba or venv. It is recommended to create a separate `env_isaacsim` instead of installing it into the environment already used for LeRobot, OpenVLA, and PyTorch experiments.

```powershell
micromamba create -n env_isaacsim python=3.11 -y
micromamba activate env_isaacsim
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

Isaac Sim requires acceptance of the NVIDIA Omniverse EULA for the first run. Environment variables can be temporarily set in PowerShell:

```powershell
$env:OMNI_KIT_ACCEPT_EULA="YES"
isaacsim isaacsim.exp.compatibility_check
isaacsim
```

If you only want to check whether the machine meets the requirements, you can first install a smaller compatibility check bundle:

```powershell
python -m pip install "isaacsim[compatibility-check]==5.1.0" --extra-index-url https://pypi.nvidia.com
$env:OMNI_KIT_ACCEPT_EULA="YES"
isaacsim isaacsim.exp.compatibility_check
```

Checkpoint 3: `isaacsim isaacsim.exp.compatibility_check` will launch the Isaac Sim compatibility check application. It is more suitable as a pre-installation inspection than the full GUI, but it cannot replace the subsequent full scenario verification.

## 7. Installation of Linux workstations and cloud servers

The Linux local workstation can directly use the Workstation installation package. Step 1: Download the installation package:

```bash
mkdir -p ~/isaacsim-downloads
cd ~/isaacsim-downloads
wget -c https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
```

Step 2: Unzip and start App Selector:

```bash
mkdir -p ~/isaacsim
cd ~/isaacsim-downloads
unzip "isaac-sim-standalone-5.1.0-linux-x86_64.zip" -d ~/isaacsim
cd ~/isaacsim
./post_install.sh
./isaac-sim.selector.sh
```

If it is a cloud server, there is usually no physical monitor. The Workstation GUI requires VNC, NICE DCV, X11 forwarding, or Isaac Sim Livestream for support. In this case, it is recommended to consider "cloud server" and "local Linux workstation" as the same Isaac installation setup, with only additional remote display and port exposure issues for cloud servers. Do not open ports directly to the public network; instead, use an SSH tunnel or the security group settings of the cloud provider to restrict source IPs.

If using the pip method, the Linux command is as follows:

```bash
micromamba create -n env_isaacsim python=3.11 -y
micromamba activate env_isaacsim
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES
isaacsim isaacsim.exp.compatibility_check
isaacsim
```

If `ldd --version` indicates that GLIBC is below 2.35, do not install the pip version rigidly on old systems. A safer approach is to switch to Ubuntu 22.04/24.04, or use the official binary installation package and container method.

## 8. Linux Docker Installation

Docker is more suitable for Linux cloud servers, multi-person reproduction, and environment isolation. NVIDIA officially states that the Isaac Sim container only supports Linux. If you want to use the Isaac Sim GUI on Windows local, prefer using Workstation or pip; it is not recommended to use Windows Docker as the preferred teaching method.

Step 1: Install Docker. If the cloud server image already includes Docker, you can skip this step:

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
docker --version
```

Step 2: Install the NVIDIA Container Toolkit. The following is the official installation method for Ubuntu. For other distributions, refer to the official documentation of the NVIDIA Container Toolkit:

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

Step 3: Verify that a GPU can be seen inside the container:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

Step 4: Pull the Isaac Sim 5.1 image. The image comes from NVIDIA NGC. Logging in to NGC may be required for the first pull. If authentication fails, register and generate an API Key in [NGC](https://ngc.nvidia.com/), and then execute `docker login nvcr.io`.

```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/isaac-sim:5.1.0
```

It is recommended to create separate directories for Isaac Sim cache, logs, and user data, so that the cache is not lost after the container is deleted:

```bash
mkdir -p ~/docker/isaac-sim/cache/main
mkdir -p ~/docker/isaac-sim/cache/computecache
mkdir -p ~/docker/isaac-sim/config
mkdir -p ~/docker/isaac-sim/data
mkdir -p ~/docker/isaac-sim/logs
mkdir -p ~/docker/isaac-sim/pkg
sudo chown -R 1234:1234 ~/docker/isaac-sim
```

Start the interactive container:

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

Run the compatibility check first after entering the container:

```bash
./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```

Checkpoint 4: If the system check passes here, it means that Docker, NVIDIA Container Toolkit, GPU driver, and Isaac Sim container can at least work in headless mode. However, it is not equivalent to a successfully configured remote GUI or Livestream.

## 9. How to open the screen on a remote cloud server

The Isaac Sim instance for cloud servers and local workstations is a single entity, but there is the issue of "how to view the screen" in the cloud environment. The simplest teaching solution is VNC or NICE DCV; if Isaac Sim provides a live stream, you can download the WebRTC Streaming Client.

Download this client when connecting Isaac Sim to the cloud from a Windows local computer:

```powershell
$ISAAC_SIM_DOWNLOADS = Join-Path $env:USERPROFILE "Downloads\isaacsim"
New-Item -ItemType Directory -Force $ISAAC_SIM_DOWNLOADS
Invoke-WebRequest `
  -Uri "https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe" `
  -OutFile (Join-Path $ISAAC_SIM_DOWNLOADS "isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe")
```

Download this AppImage from the Linux client:

```bash
mkdir -p ~/isaacsim-downloads
cd ~/isaacsim-downloads
wget -c https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage
chmod +x isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage
```

The configuration for remote display is closely related to the security groups of cloud providers. In principle, do not expose VNC or livestream ports directly to the public network. Instead, use an SSH tunnel or allow only your own fixed IP address to access them.

## 10. Isaac Lab Configuration

Isaac Lab is a robot learning framework built on top of Isaac Sim, suitable for reinforcement learning, imitation learning, task construction, and parallel simulation. The recommended process is to ensure Isaac Sim can start before installing Isaac Lab. Do not install Isaac Lab directly when Isaac Sim cannot be opened, as this will make it difficult to identify subsequent errors.

The following commands are based on Isaac Lab 2.3.x and Isaac Sim 5.1. The Isaac Lab source code can be downloaded from GitHub, and you can switch to the corresponding release version afterward. It is recommended to visit the Isaac Lab official release page to confirm the Isaac Sim version corresponding to the current version, before deciding whether to upgrade.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.0
```

Install the basic compilation dependencies on Linux, and then install the Isaac Lab extension:

```bash
sudo apt update
sudo apt install -y cmake build-essential
./isaaclab.sh --install
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

Execute on Windows in an activated Isaac Python environment:

```powershell
isaaclab.bat --install
isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
```

The official verification script launches a blank scene with a black viewport. This result seems simple, but it proves that Isaac Lab can invoke the current Python environment, locate Isaac Sim, load basic extensions, and open the simulation application.

## 11. How to Choose ROS2 Integration

Isaac Sim 5.1 supports ROS2 Humble and Jazzy. For Ubuntu 22.04, Humble is recommended; for Ubuntu 24.04, Jazzy is recommended. In Windows 10/11 scenarios, the official documentation suggests using ROS2 Humble via WSL.

Beginners are advised not to install ROS2 and Isaac Sim together on the first day. A more stable order is to start with opening Isaac Sim, then installing ROS2, and finally enabling the ROS2 Bridge. Default message types such as `std_msgs`, `geometry_msgs`, and `nav_msgs` usually run smoothly; if you need to customize the ROS2 interface, you need to consider the compatibility between Isaac Sim Python 3.11 and the system ROS Python version.

## 12. When to start the GR00T tutorial

GR00T is not an introductory configuration for Isaac Sim, but a later-stage practice for robot foundation-model development. The repository's [Alibaba Cloud tutorial for deploying Isaac Lab and GR00T](02-complete-tutorial-on-deploying-isaac-lab-gr00t-in-alibaba-cloud.md) uses a reproducible version set: Isaac Sim 4.2.0, Isaac Lab v1.4.1, GR00T N1.6, PyTorch 2.5.1, and CUDA 12.1. This combination differs from the Isaac Sim 5.1 route used in this article.

If you only want to study Isaac Sim locally, there is no need to install GR00T immediately. If your goal is to reproduce GR00T, it is recommended to use a separate cloud server or isolated environment and follow the original tutorial version. Do not install GR00T as a dependency in `env_isaacsim`. This approach can prevent PyTorch, CUDA, transformers, flash-attn, and the Isaac extension from interfering with each other.

## 13. Common Questions

When the Isaac Sim installation fails, first troubleshoot by layer. The first layer is hardware and drivers: whether `nvidia-smi` is functioning properly, whether the drivers are sufficiently updated, whether the GPU has an RT Core, and whether the memory is too low. The second layer is the system: whether Windows supports long path support, whether Linux GLIBC meets pip package requirements, and whether at least several tens of GB are reserved on the disk. The third layer is Python: whether Python 3.11 is used, whether Isaac is installed in a clean environment, and whether the NVIDIA PyPI source is used.

If the first startup is slow, do not close the window immediately. The first launch of Isaac Sim will download extensions, compile the shader cache, and initialize user settings. It is normal for it to take 5 to 10 minutes. Only when the log clearly indicates a driver error, extension download failure, unacceptance of the EULA, or process crash, should you address it according to the error message.

If the pip installation on Windows reports a path-length error, enable long-path support or move the environment to a shorter directory such as `$env:USERPROFILE\isaacsim` instead of a deeply nested personal document directory.

If the GUI on the cloud server cannot be opened, first determine whether Isaac can start headless. If the compatibility check passes, the issue is usually in VNC, WebRTC, DISPLAY, Xauthority, or security-group ports rather than Isaac Sim itself. Verify the headless route first, then address remote-display issues.

## 14. Light verification records of this tutorial

This article did not fully download the 40GB Isaac Sim installation package on this repository machine, as it would incur significant time and disk costs. The completed lightweight verification includes: the local `nvidia-smi` can recognize the NVIDIA GeForce RTX 3060 Laptop GPU, with a driver version of 581.95 and 6144 MiB of video memory; the current Python is 3.11.15; the micromamba version is 2.3.0; and through PyPI and NVIDIA PyPI indexes, the `isaacsim` package versions 5.1.0.0 and 5.0.0.0 can be queried.

This set of verifications proves that the local machine has the basic requirements for NVIDIA drivers, Python 3.11, and package index access. However, it does not prove that a 6GB GPU memory is sufficient to run complex Isaac scenarios. When actually delivered to learners, it is still recommended to confirm step by step using the Checkpoint in this article.

## References

- [NVIDIA Isaac Sim 5.1 Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html)
- [NVIDIA Isaac Sim 5.1 Downloads](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
- [NVIDIA Isaac Sim 5.1 Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [NVIDIA Isaac Sim 5.1 Workstation Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)
- [NVIDIA Isaac Sim 5.1 Python Environment Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [NVIDIA Isaac Sim 5.1 Container Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html)
- [NVIDIA Isaac Sim 5.1 ROS 2 Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html)
- [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Isaac Lab 2.3 Installation using Isaac Sim Pip Package](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
