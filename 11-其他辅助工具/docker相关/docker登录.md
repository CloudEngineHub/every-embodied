# Docker 镜像仓库登录

`docker login` 用于将本机 Docker 客户端连接到镜像仓库。登录成功后，`docker pull` 和 `docker push` 会使用已保存的凭据访问私有镜像或受限资源。

## 登录 Docker Hub

直接执行：

```bash
docker login
```

新版 Docker 客户端在 Docker Hub 上默认可使用设备代码登录。终端会显示验证地址和一次性代码，在浏览器中完成授权即可。如果需要显式指定账号，可使用：

```bash
docker login --username <docker-id>
```

## 登录自建仓库

命令中只填写仓库主机名和可选端口，不要附加镜像路径：

```bash
docker login registry.example.com
docker login registry.example.com:5000
```

## 在脚本中使用令牌

自动化脚本不应把密码直接写在命令行中。使用 `--password-stdin` 可避免密码出现在 shell 历史和进程列表中：

```bash
printf '%s' "$REGISTRY_TOKEN" | \
  docker login registry.example.com \
  --username "$REGISTRY_USER" \
  --password-stdin
```

`REGISTRY_TOKEN` 应由交互式终端、密钥管理服务或持续集成平台注入，不写入教程、源码仓库或镜像层。

## 凭据存储与退出

Docker 通常将登录配置写入 `~/.docker/config.json`。如果系统配置了 credential store，实际凭据会交由操作系统密钥链管理。公用设备或短期服务器使用完成后应退出：

```bash
docker logout
docker logout registry.example.com
```

## 验证与排错

```bash
docker info
docker pull hello-world:latest
```

- 出现 `unauthorized` 时，先检查仓库地址、用户名和令牌权限。
- 已登录但仍无法推送时，检查镜像名是否包含正确的仓库前缀和命名空间。
- 无图形界面的服务器上无法调用密钥链时，可为该主机配置可用的 credential helper，或使用有限权限的短期令牌。

官方参考：[Docker CLI `login`](https://docs.docker.com/reference/cli/docker/login/)。
