# Configure APT Mirrors and XRDP in WSL

The first section replaces Ubuntu package sources with a regional mirror. The remaining sections configure an XRDP desktop session for WSL.

```
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak && sudo tee /etc/apt/sources.list > /dev/null <<'EOF'
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse

deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
# deb-src http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
EOF
```

```
sudo apt update

sudo apt install -y xfce4 xfce4-goodies
sudo apt install -y xrdp


echo "xfce4-session" > ~/.xsession

# 编辑 xrdp 启动脚本
sudo sed -i 's/^test -x \/etc\/X11\/Xsession && exec \/etc\/X11\/Xsession$/startxfce4/' /etc/xrdp/startwm.sh

# 或者手动编辑 /etc/xrdp/startwm.sh，在文件末尾添加：
# startxfce4

# 默认端口是 3389，如需修改可编辑配置文件
sudo nano /etc/xrdp/xrdp.ini
# 找到 port=3389 这行，可以修改为其他端口

sudo service xrdp start

# 检查服务状态
sudo service xrdp status

# 在 WSL2 中，可以在 ~/.bashrc 或 ~/.zshrc 末尾添加：
# sudo service xrdp start
```





Detailed steps are as follows

## Start a Graphical Interface with XRDP in WSL



## 1. Install the graphical interface environment

First, you need to select and install a desktop environment. Here are two common options:

### Option A: Install Xfce (lightweight, recommended)

```bash
sudo apt update
sudo apt install -y xfce4 xfce4-goodies
```

### Option B: Install Ubuntu Desktop (Full Desktop)

```bash
sudo apt update
sudo apt install -y ubuntu-desktop
```

## 2. Installing xrdp

```bash
sudo apt install -y xrdp
```

## 3. Configure xrdp

```bash
# 配置 xrdp 使用 Xfce
echo "xfce4-session" > ~/.xsession

# 编辑 xrdp 启动脚本
sudo sed -i 's/^test -x \/etc\/X11\/Xsession && exec \/etc\/X11\/Xsession$/startxfce4/' /etc/xrdp/startwm.sh

# 或者手动编辑 /etc/xrdp/startwm.sh，在文件末尾添加：
# startxfce4
```

## 4. Configure xrdp port (optional)

```bash
# 默认端口是 3389，如需修改可编辑配置文件
sudo nano /etc/xrdp/xrdp.ini
# 找到 port=3389 这行，可以修改为其他端口
```

## 5. Start the xrdp service

```bash
sudo service xrdp start

# 检查服务状态
sudo service xrdp status
```

## 6. Set to start automatically on boot (optional)

```bash
# 在 WSL2 中，可以在 ~/.bashrc 或 ~/.zshrc 末尾添加：
# sudo service xrdp start
```

## 7. Connect to the graphical interface

**On Windows:**

1. Open the Run window by entering `Win + R`.
2. Enter `mstsc` and press Enter to activate the remote desktop connection.
3. Enter `localhost:3389` or `127.0.0.1:3389` in the computer name field.
4. Click Connect.
5. Enter the WSL username and password.

## 8. Troubleshooting

### Black screen or unable to connect

```bash
# 重启 xrdp 服务
sudo service xrdp restart

# 检查端口是否被占用
netstat -an | grep 3389

# 查看 xrdp 日志
cat /var/log/xrdp.log
cat /var/log/xrdp-sesman.log
```

### Optimal Configuration

The following can be adjusted in `/etc/xrdp/xrdp.ini`:

```ini
# 颜色深度（提高性能）
max_bpp=16

# 压缩设置
compression_level=9
```

## 9. Additional Tips

**Performance Optimization:**
- If you encounter performance issues, consider using Xfce instead of GNOME or KDE
- Reducing the color depth can improve response speed

**Security Recommendations:**
```bash
# 只允许本地连接
sudo ufw allow from 127.0.0.1 to any port 3389
```

**Start script (in Windows PowerShell):**
```powershell
# 创建一个启动脚本
wsl -d Ubuntu sudo service xrdp start
```

## 10. Alternative Solutions

If xrdp encounters issues, you can also consider:

- **VcXsrv** or **X410**: Use X Server
- **WSLg**: WSL2 built-in graphics support (requires Windows 11 or the latest version of Windows 10)
