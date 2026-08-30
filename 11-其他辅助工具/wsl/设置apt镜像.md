# 在 WSL Ubuntu 中配置 APT 镜像

当默认软件源访问缓慢时，可以把 Ubuntu 软件源切换到就近镜像。修改前先确认发行版代号，并备份现有配置。

## 检查系统版本

```bash
. /etc/os-release
echo "$PRETTY_NAME"
echo "$VERSION_CODENAME"
```

下面以 Ubuntu 22.04 的 `jammy` 为例。其他版本应把代号替换为命令实际输出。

## 备份并写入镜像

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo tee /etc/apt/sources.list > /dev/null <<'EOF'
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
EOF
sudo apt-get update
```

Ubuntu 24.04 及更新版本可能使用 `/etc/apt/sources.list.d/ubuntu.sources`。这种情况下应先备份该文件，再按同样的镜像地址修改其中的 `URIs` 字段。

## 验证

```bash
apt-cache policy | sed -n '1,30p'
sudo apt-get install --download-only -y curl
```

输出中应出现新镜像地址，并能正常下载软件包。

## 排错与恢复

- 提示发行版不存在：检查 `VERSION_CODENAME`，不要把 `jammy` 用于其他 Ubuntu 版本。
- 证书错误：先检查系统时间，再确认 `ca-certificates` 已安装。
- 镜像暂时不可用：恢复备份并更新索引。

```bash
sudo cp /etc/apt/sources.list.bak /etc/apt/sources.list
sudo apt-get update
```
