# BeyondRetarget 实测记录

日期：2026-10-18。结论：**官方在线服务的视频到八机器人动作生成已跑通；本地 GPU 推理、训练、SONIC 动力学跟踪和真机控制未验证。**

[新增方法对比页](../comparison.md)使用作者论文数据与官方演示，并给出本地基线复跑流程；四条 GVHMR/WHAM → GMR/NMR 基线尚未实跑。已核对 GMR 的 GVHMR 输入入口、NMR 输入输出文档及其在线接口，不把这些接口核查当成基线实验完成。

## 输入及来源

- 输入来源：[用户提供的视频号演示](https://weixin.qq.com/sph/A9gBA1DBZ6)，研究工作为南京大学等团队的 [BeyondRetarget](https://bear-ty.github.io/Beyondretarget_page/)。
- 原视频经本地下载确认：720×1280、78.75 秒、H.264 / AAC。
- 取第 2–8 秒，只裁出左侧真人全身画面，裁剪尺寸 178×306、左上角 `(0,534)`，重采样为 30 FPS、去音轨、H.264 CRF 18。原视频与输入裁剪不随教程发布。
- 输入 SHA256：`4f425867086a0eede218a671748ce14666a1fbf53b21e767f7e3be6fa0bb2d69`。

本次输入准备命令如下，`source.mp4` 为读者自行取得的原视频。若平台后续替换视频画幅，必须重新确认裁剪区域，不应照抄坐标：

```bash
ffmpeg -ss 2 -i source.mp4 -t 6 -vf "crop=178:306:0:534,fps=30" -an -c:v libx264 -crf 18 person_6s.mp4
```

这段素材人物高度有限，主要是站立和手部动作。该样例验证服务调用与数据导出，不是独立测试集、定量复现实验或对论文泛化能力的验证。

## 运行环境与版本

- 客户端：Windows、Python 3.12、`gradio_client==1.13.3`、`numpy==1.26.4`。
- 推理设备：官方 Hugging Face ZeroGPU 服务；未使用本地 GPU。
- Space：`bear-ty/BeyondRetarget`；核查公开外壳 SHA：`13eef4dde553ff96a39c7129291ef70c9a31fcef`。私有运行包版本未公开，不能据此外推其内部模型与 GitHub 的对应版本。
- 两个阶段按同一会话顺序调用：`/zero_gpu_inference` → `/finish_cpu_stage`。
- 运行开始：北京时间 2026-10-18 20:24:47；总耗时约 179.78 秒，包含网络、队列、GPU 推理、CPU 渲染及下载，不是纯模型延迟。
- 本地步骤仅核对上游提交 `e0331d0faf9091095bae9100e03655c944e2b64a` 的入口与资源定义。环境探测时可用实验室 GPU 只剩约 2 GB 显存，未启动推理或改动其他任务。

## 实际产物

| 文件 | 实测结果 |
| --- | --- |
| [eight_robots.mp4](./eight_robots.mp4) | H.264、1920×1080、30 FPS、180 帧、6.0 秒；3,276,479 B |
| [motions.zip](./motions.zip) | 212,005 B；8 个 NPZ，均为 `*_raw_pred.npz` |
| [run.json](./run.json) | 完成状态、输入及输出 SHA256、耗时 |
| [motion_report.json](./motion_report.json) | 每个数组形状、FPS、有效性检查结果 |
| [preview.jpg](./preview.jpg) | 回放的第 29、89、149 帧，缩放拼图 |

| NPZ 中的机器人 | 帧数 | FPS | 关节维度 | 数值与单位四元数检查 |
| --- | ---: | ---: | ---: | --- |
| G1 | 180 | 30 | 29 | 通过 |
| R1 | 180 | 30 | 26 | 通过 |
| GR1-T1 | 180 | 30 | 32 | 通过 |
| GR2 | 180 | 30 | 29 | 通过 |
| H1 | 180 | 30 | 21 | 通过 |
| Tienkung | 180 | 30 | 20 | 通过 |
| T1 Serial | 180 | 30 | 23 | 通过 |
| Atlas v4 | 180 | 30 | 30 | 通过 |

上述维度是在线服务产物的实际观测，不替代本地各机器人配置。GR2/H1 文件名前缀也与本地 CLI 名称不同。ZIP 中没有接触预测或独立后处理数据，因此本记录不声称验证了服务内部的后处理过程。

关键帧中可见八个机器人与姿势变化。机器人被放置在同一场景回放，不能由此判定重力、支撑接触或闭环跟踪成立。没有真实机器人/人体真值、足部接触标注和跟踪控制器日志，故不报告 MAMPJPE、脚滑指标、延迟基准或执行成功率。

## 教程代码验证

从本章目录执行：

```bash
python -m unittest discover -s scripts -p 'test_*.py'
python scripts/inspect_motion.py results/motions.zip
```

- 六组单元测试通过：有效原始动作、异常预测、有效 body 导出、接触数据误输入、对象数组及损坏文件。
- 随章八份真实推理数据均通过检查。
- 客户端实际完成两阶段调用和文件保存；对 78.75 秒原视频的输入检查在上传前拒绝，未创建输出目录。
- 教程目录被电子书目录收录，正文经现有 Markdown 渲染器生成 HTML；本章相对链接检查通过。

## 署名与资源边界

研究与模型归 Tianyu Xiong、Yi Lu 等 BeyondRetarget 作者及所属团队；视频和动作文件由其[官方在线服务](https://huggingface.co/spaces/bear-ty/BeyondRetarget)根据上述裁剪输入新生成。`preview.jpg` 由生成视频抽帧、缩放与拼接，教程未重新训练模型。

上游原始代码和公开权重为 [CC BY-NC 4.0](https://github.com/bear-ty/BeyondRetarget/blob/main/LICENSE)，机器人外观资产等保留[各自来源和许可](https://github.com/bear-ty/BeyondRetarget/blob/main/THIRD_PARTY_NOTICES.md)。本仓库通用许可不替代上游资源及第三方资产的许可；本章不重新分发权重、机器人网格、私有运行包或 MotionPRO 原始数据。
