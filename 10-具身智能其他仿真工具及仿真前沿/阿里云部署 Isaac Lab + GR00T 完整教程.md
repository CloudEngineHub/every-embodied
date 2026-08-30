# 阿里云 Isaac Lab 与 GR00T 教程导航

阿里云部署教程已经移入 Isaac 专题目录。本页保留稳定入口，并给出开始实验前的检查顺序。

## 阅读顺序

1. 先阅读 [Isaac Sim、Isaac Lab 与 GR00T 部署导览](01Isaac部署与GR00T实践/00Isaac部署导览.md)，确定容器、远程桌面和策略服务的关系。
2. 再执行 [阿里云部署 Isaac Lab + GR00T 完整教程](01Isaac部署与GR00T实践/02阿里云部署Isaac-Lab-GR00T完整教程.md)。
3. 完成部署后，分别验证显卡、Isaac Lab 示例和 GR00T 模型加载。

## 最小验证

```bash
nvidia-smi
docker ps
```

进入教程创建的环境后，再运行对应的 Isaac Lab 示例与 GR00T 导入命令。三个检查点应分别证明加速设备可见、容器正常运行和 Python 依赖可导入。

## 常见问题

- 容器看不到显卡：检查 NVIDIA Container Toolkit 与容器启动参数。
- 远程桌面可连接但仿真黑屏：改用无图形界面模式验证，再排查显示转发。
- 模型下载中断：保留已下载缓存并使用支持续传的下载方式。
