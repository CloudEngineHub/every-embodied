# MicroDuck 跨缝跳跃配套代码

打开 [第 9 本 Notebook](../../03-Notebook/09_跨缝跳跃_PPO_GPU_ONNX.ipynb)，按顺序运行：模型下载与校验、GPU 仿真、10 轮 PPO 续训、ONNX 导出对齐和完整视频。

## 环境

Ubuntu、可用的 CUDA、Python 3.12、Git、EGL、ffmpeg 与 uv。Jupyter 内核需要 `nbformat`、`numpy`、`matplotlib`、`IPython`；任务依赖由独立目录中的 `uv sync --locked` 安装。

```bash
python -m pip install jupyterlab nbformat nbclient numpy matplotlib
cd ../../03-Notebook
jupyter lab
```

Notebook 不依赖固定工作站 IP，不连接 RDK。模型、源码和环境默认在 `~/projects/robotics_shared/downloads/microduck-long-jump/`，可用 `MICRODUCK_JUMP_HOME` 改位置。网络需要代理时，仅给本次下载设置 `MICRODUCK_DOWNLOAD_PROXY`。

## 来源与接口

- [任务源码](https://github.com/Vottivott/microduck-playground/tree/b6aed572587b52e41a452a9b2be517c95d2b3d96/experiments/parkour/long-jump)：固定源码版本，Apache-2.0。
- [模型包](https://huggingface.co/HannesVonEssen/microduck-long-jump)：原始 PPO 检查点、ONNX、命令配置和核验清单。
- [续训方法](https://github.com/Vottivott/microduck-playground/blob/b6aed572587b52e41a452a9b2be517c95d2b3d96/experiments/parkour/TRAINING.md)：恢复完整 PPO 状态，按保存的训练计数应用课程。

任务为 `Mjlab-PlatformJump-MicroDuck`，61 维观测、14 维有界关节动作、50 Hz 控制。命令前两项是间隙和高差除以 0.3；ONNX 已包含归一化和动作边界。不能按普通行走模型的默认命令解释它。

## 文件

- `build_notebook.py`：生成第 9 本，不修改已有 Notebook。
- `jump_lesson.py`：核验模型、准备环境、调用训练和导出入口。
- `jump_runtime.py`：记录完整物理回合、真实轨迹和终止项，生成 MP4。
- `test_jump_lesson.py`：不联网的辅助代码测试。

训练候选模型在 `03-Notebook/outputs/long_jump/training/`，每次运行覆盖最新候选。原始模型始终保留。检查点与环境不提交到资料仓库；本轮三段完整视频与报告收录在 `03-Notebook/recordings/long_jump/`，GitHub 使用压缩 GIF 预览。

部署前分别检查原始模型和续训候选的跨缝、落地支撑与 12 秒回合。候选通过才采用，否则保留通过评测的原始模型；两者都失败会阻止部署。加载模型后按固定种子重置环境，让成对评测使用同一初始状态。
