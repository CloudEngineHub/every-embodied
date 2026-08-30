# 在 Linux 中安装 micromamba

`micromamba` 是单文件环境管理器，适合服务器和没有系统级 Conda 的环境。下面把可执行文件安装到 `$HOME/.local/bin`，并把环境根目录放在用户目录下。

## 安装

```bash
sudo apt-get update
sudo apt-get install -y bzip2 curl
mkdir -p "$HOME/.local/bin"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C "$HOME/.local/bin" --strip-components=1 bin/micromamba
export PATH="$HOME/.local/bin:$PATH"
```

## 初始化 Shell

```bash
micromamba shell init -s bash --root-prefix "$HOME/micromamba"
source "$HOME/.bashrc"
```

创建测试环境：

```bash
micromamba create -n test-python python=3.11 -y
micromamba run -n test-python python --version
```

## 验证与排错

`micromamba info` 应显示根目录与软件源配置。若终端找不到命令，确认 `$HOME/.local/bin` 已加入 `PATH`；若下载时出现证书错误，先检查系统时间和证书包，不要关闭证书校验。
