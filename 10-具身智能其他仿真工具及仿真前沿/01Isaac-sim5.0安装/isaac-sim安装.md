# Isaac Sim 5 安装导航与环境验证

Isaac Sim 5 的完整安装步骤已经统一到专题教程。本页用于帮助从旧链接进入的读者选择正确路线，并在安装后完成最小验证。

## 选择安装路线

- 本地工作站或常规云服务器：[Isaac Sim 本地与云端配置教程](../01Isaac部署与GR00T实践/01Isaac-Sim本地与云端配置.md)
- Isaac Lab 与 GR00T 云端实验：[阿里云部署 Isaac Lab + GR00T 完整教程](../01Isaac部署与GR00T实践/02阿里云部署Isaac-Lab-GR00T完整教程.md)

## 验证安装

在 Isaac Sim 的 Python 环境中执行：

```bash
python -c "from isaacsim import SimulationApp; print('Isaac Sim import passed')"
```

随后启动一个官方最小示例，确认窗口或无图形界面日志能够完成场景加载与至少一个仿真步。导入成功只说明 Python 包可见，场景加载成功才说明扩展、资产和渲染链路可用。

## 常见问题

- 找不到 `isaacsim`：确认使用的是 Isaac Sim 自带的 Python，或已激活文档指定环境。
- 启动后黑屏：检查显卡驱动、显示转发和渲染模式；服务器环境优先使用无图形界面模式。
- 资产加载缓慢：检查网络与缓存目录的可写权限，并为缓存预留足够磁盘空间。
