# Ubuntu 安装 Docker

以下命令使用 Docker 官方软件源安装 Docker Engine。执行前确认当前用户具有 `sudo` 权限。

```bash
# 1. 更新包索引
sudo apt-get update

# 2. 安装必要的依赖
sudo apt-get install -y ca-certificates curl gnupg

# 3. 添加 Docker 的官方 GPG 密钥
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 4. 设置仓库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. 再次更新并安装 Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

## 验证

```bash
sudo docker run --rm hello-world
docker compose version
```

若第一条命令提示无法连接服务，运行 `sudo systemctl enable --now docker` 后重试。若当前用户无权访问套接字，可以继续使用 `sudo docker`，或按团队的权限策略加入 `docker` 用户组。
