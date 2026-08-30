# WSL Graphical Interface Configuration

This guide installs a desktop environment and exposes it through XRDP for local use.

```bash
wsl
```

![image.png](../../../11-其他辅助工具/wsl/assets/a85b9693-6964-4d81-9cee-e2a1e5d7313c.png)

After installation is complete

```bash
# 需要重启电脑

wsl --install -d Ubuntu-22.04

# sudo apt update && sudo apt upgrade
# 可以先不更新
```



This method is an alternative option.

Headers often need to be recompiled:

If not, this method can be used as a temporary alternative.

```
ls /usr/src/linux-headers-5.15.0-156
sudo mkdir -p /lib/modules/5.15.0-156
cd /lib/modules/5.15.0-156
sudo ln -fs /usr/src/linux-headers-5.15.0-156 build
sudo ln -fs /usr/src/linux-headers-5.15.0-156 source
```

If the system breaks down:

```
export WSL_BACKUP_ROOT="${WSL_BACKUP_ROOT:-/mnt/c/WSL_Backup}"
mkdir -p "$WSL_BACKUP_ROOT/home"
rsync -ah --info=progress2 "$HOME/" "$WSL_BACKUP_ROOT/home/"
wsl --list --verbose
wsl --shutdown
wsl --unregister Ubuntu-22.04

# 重新构建

cd ~
rsync -ah --info=progress2 "$WSL_BACKUP_ROOT/home/" "$HOME/"
```



Now let's get to the point. The more reliable connection method is xrdp.

wsl: A local localhost proxy configuration is detected, but it has not been mirrored to WSL. WSL in NAT mode does not support local localhost proxy.

This setting for wsl setting is to set the network to mirror.

![image-20251004125329464](../../../11-其他辅助工具/wsl/assets/image-20251004125329464.png)



```
sudo apt update
sudo apt install -y xfce4 xfce4-goodies
```

## Start the Graphical Interface with XRDP

I will provide a detailed tutorial on configuring the graphical interface using xrdp in WSL:

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
2. Enter `mstsc` and press Enter to open the remote desktop connection.
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
- **WSLg**: Built-in graphics support for WSL2 (requires Windows 11 or the latest version of Windows 10)





Then

```
sudo bash -c 'cat > /etc/xrdp/startwm.sh << "EOF"
#!/bin/sh
# Unset session-breaking variables
unset DBUS_SESSION_BUS_ADDRESS
unset XDG_RUNTIME_DIR

# Source profile to get path and other env vars
. /etc/profile

# Start the Xubuntu/Xfce desktop session
exec startxfce4
EOF'
```

```
sudo bash -c 'cat > /etc/xrdp/startwm.sh << "EOF"
#!/bin/sh
# Unset session-breaking variables
unset DBUS_SESSION_BUS_ADDRESS
unset XDG_RUNTIME_DIR

# Source profile to get path and other env vars
. /etc/profile

# Start the MATE desktop session
exec mate-session
EOF'
```

Packages that need to be installed additionally for the mate desktop



```

```
