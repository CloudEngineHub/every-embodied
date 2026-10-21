# 05-LTX Alpha Gen：视频抠像、半透明边缘与透明合成

给一段普通视频，保留人物的发丝、烟雾的薄边和火焰的透明过渡，再把主体放到新的背景上。**LTX-2.5 Alpha Gen** 把这个任务表示成“根据 RGB 视频生成一段透明度视频”，为视频素材分层和离线视觉数据处理提供了一条开放权重路线。

本章接在 [SAM 分割与单目深度估计](./01-sam和深度估计.md)和 [Locate Anything 视觉语言定位](./04-Locate-Anything视觉语言定位.md)之后，进一步理解分割与抠像的区别，并学习如何用官方 ComfyUI 工作流或 Python 管线生成 alpha matte、合成透明视频。

本章依据 2026-10-21 的官方发布材料与源码编写，已完成代码阅读、尺寸计算和电子书构建检查；使用合成素材在 CPU 上验证了预处理和 ProRes 4444 透明导出，记录见[命令检查](./assets/ltx-alpha-gen/command-check.json)。GPU 推理步骤供读者执行，未列为本地实测结果。

## 一、先看它输出什么

Alpha Gen 的输入是一段普通 RGB 视频，输出是与输入逐帧对齐的灰度 **alpha matte（透明度图）**：

| 透明度图上的颜色 | 数值 | 合成时的作用 |
| --- | ---: | --- |
| 黑色 | 0 | 使用新背景 |
| 白色 | 1 | 完全保留前景 |
| 灰色 | 0～1 之间 | 按比例混合前景与背景 |

毛发边缘、烟雾、火焰、薄纱和运动模糊包含大量中间值，连续透明度可以让这些区域自然融入新背景。相比只回答“这个像素属于哪个物体”，抠像还要回答“这个像素有多少前景成分”。

![Alpha Gen 数据流：RGB 视频作为持续参考，视频模型生成透明度，再与原视频和新背景合成](./assets/ltx-alpha-gen/pipeline.svg)

图 1：本教程绘制的机制示意图，展示数据流，不是模型实测输出。实际示例可在[官方模型卡](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen)中查看。

理想的图像合成关系为：

$$
C = \alpha F + (1-\alpha)B
$$

其中 $C$ 是观察到的像素，$F$ 是前景颜色，$B$ 是背景颜色，$\alpha$ 是透明度。官方推荐把预测 matte 作为原 RGB 视频的 alpha 使用：

```python
output = source_rgb * alpha + new_background * (1 - alpha)
```

原 RGB 视频负责颜色，灰度 matte 负责透明程度。生成的黑白 MP4 本身没有封装透明通道，需要在合成软件中使用它，或进一步输出 ProRes 4444 等带 alpha 的格式。

## 二、它在具身视觉链路中的位置

| 模块 | 典型输入 | 典型输出 | 适合做什么 |
| --- | --- | --- | --- |
| Locate Anything | 图像、自然语言 | box / point | 定位指定目标 |
| SAM 等分割模型 | 图像、点或框等条件 | 目标区域 mask | 实例选择、区域提取 |
| Alpha Gen | RGB 视频 | 连续透明度视频 | 细边保留、素材分层、背景合成 |
| 深度估计 / RGB-D | 图像或传感器数据 | 深度 | 恢复空间几何信息 |

Alpha Gen 对具身方向的价值主要在**离线视觉素材处理**：例如提取机器人或物体的视觉图层，制作教学演示，尝试背景变化的数据增强。将合成素材用于策略训练时，应同步处理标注，并观察透明边缘和新背景是否改变任务线索。

当前公开适配器自动判断主体，不能通过文字、点击或输入 mask 选择某个指定物体。如果画面中有多个人或多件物体，需要先确认选中的前景是否符合任务。它输出透明度，不提供深度、抓取位姿或机器人动作。

## 三、项目、论文和代码是什么关系

| 材料 | 入口 | 本章使用方式 |
| --- | --- | --- |
| Alpha Gen 0.9 模型 | [Hugging Face](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen) | 核对权重、推荐设置与输入范围 |
| Alpha Gen 官方文档 | [Alpha Gen Beta](https://docs.ltx.io/open-source-model/vfx-post-production/alpha-gen) | 对照 Python 与 ComfyUI 发布路线 |
| 基础模型 | [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | 提供 Transformer、VAE、文本编码器等组件 |
| 官方 Python 实现 | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) | 阅读参考条件、推理与通用训练框架 |
| ComfyUI 节点 | [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) | 导入官方单阶段工作流 |
| 基础论文 | [LTX-2: Efficient Joint Audio-Visual Foundation Model](https://arxiv.org/abs/2601.03233) / [PDF](https://arxiv.org/pdf/2601.03233) | 理解音视频生成骨干 |

截至核验日期，未在官方引用和 arXiv 检索中找到 Alpha Gen 的独立论文。关联的 LTX-2 论文发表于 2026-01-06，共 14 页，研究联合音视频生成；当前 Alpha Gen 则基于后来发布的 LTX-2.5 22B，通过 IC-LoRA 完成视频抠像。阅读时应把基础架构与具体抠像适配器区分开。

LTX-2 论文的主要设计包括：视频和音频使用独立 VAE、不对称双流 Transformer、双向音视频注意力、文本多层特征与 thinking tokens，以及可分别调节文本和跨模态影响的 guidance。Alpha Gen 使用空提示词，并通过参考视频约束目标视频生成。

## 四、为什么视频生成模型可以做抠像

### 4.1 把透明度也表示为视频

RGB 输入先经视频 VAE 编码成 latent tokens。待生成的灰度 matte 也用视频 latent 表示，从噪声开始逐步生成，再经过 VAE 解码得到黑白视频。

这样，原本用于视频生成的时空表达能力可以服务于透明度估计。输入参考视频贯穿去噪过程，为输出提供每一帧的内容和位置条件。

### 4.2 IC-LoRA：参考 token 与目标 token 一起处理

IC 是 In-Context。公开代码把参考视频 token 和目标 token 拼入同一序列，通过注意力交换信息：

- **参考部分保持干净。** 参考 token 的去噪 mask 为 0，作为持续条件使用。
- **目标部分逐步去噪。** 模型在参考条件下生成 matte 的 latent。
- **坐标保持对应。** 时间和空间位置编码把参考放到目标坐标系；Alpha Gen 模型卡给出的参考空间缩放因子为 1。

LoRA 用低秩矩阵适配基础模型，可写为 $W' = W + sBA$。它降低任务适配的训练参数量；推理仍需运行大型基础 Transformer，不能按“小适配器文件”估计整条流程的资源需求。

### 4.3 从通用训练代码读目标函数

当前 V2V 配置采用 `FlexibleStrategy`：对目标 latent 加噪，保持参考 latent 不变，损失只计算生成目标部分。

$$
z_\sigma=(1-\sigma)z_{target}+\sigma\epsilon,\qquad
v_{target}=\epsilon-z_{target}
$$

训练使用目标 token 上的 masked MSE。把 RGB 看作参考、matte 看作目标，就能理解这个任务的基本组织方式。公开材料没有给出 Alpha Gen 专用训练集构成、完整训练配置或专用损失；通用模板中的 rank 等值不应直接当成其实际训练参数。模型卡明确发布检查点为第 25,000 训练步。

## 五、先选计算资源与体验路线

**这是一条偏离线的视频处理路线。** 高清片段需要较多显存和计算，实际耗时随分辨率、帧数、量化方式与卸载策略变化。

| 需求 | 可参考的资源信息 | 选择建议 |
| --- | --- | --- |
| 先理解功能 | 阅读官方示例与本章工作流 | 无需下载权重 |
| 本地小片段体验 | 降低分辨率、缩短片段；显存紧张时考虑量化和 CPU 卸载 | 从 960×544、49 或 121 帧开始 |
| 高清完整规格 | 官方模型卡指出，1080p、145 帧需要 H100/B200 级 GPU | 先确认显存，再尝试最大配置 |
| 摄像头、直播或机器人在线处理 | 当前发布路线对视频片段执行多步扩散 | 按离线处理规划 |

官方基础推理示例所列组件下载量约 66GiB，这是存储量，不是显存实测。仅按 22B 参数、每个 BF16 参数 2 字节估算，参数存储约 44GB；运行还涉及其他组件、中间激活和工作区。量化、按需加载可以降低同时驻留显存的权重，但会改变部署方式和耗时。

作为速度量级参考，第三方服务 [TryH3](https://tryh3.com/generator/ltx-alpha-gen)报告，其 2026-10-06 样例中，5 秒素材的 matte 在 720p 约需 3 分钟、1080p 约需 8 分钟，启动、排队和透明导出另计。该服务未注明 GPU，这些数字是其服务样例，不是官方基准或本章本机成绩。

基础论文中的 1.22 秒/扩散步是在 H100、121 帧、720p 等指定条件下对 LTX-2 的测量；不能将它直接当成 Alpha Gen 完整处理耗时。做自己的测速时，应分别记录模型加载、视频预处理、推理、解码和合成导出时间。

## 六、路线 A：使用官方 ComfyUI 工作流

已使用 ComfyUI 的读者可以从这条路线开始：

1. 按[官方仓库说明](https://github.com/Lightricks/ComfyUI-LTXVideo)安装或更新 LTX 节点。
2. 在 Hugging Face 上按页面提示完成 LTX-2.5 和 Alpha Gen 的访问条件，下载对应组件。
3. 导入本章保留的[官方单阶段 IC-LoRA 工作流](./assets/ltx-alpha-gen/LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json)，也可从[上游工作流](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/2.5/LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json)获取更新。
4. 使用 LTX-2.5 distilled 基础模型，把 `ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors` 放到 `ComfyUI/models/loras/`，在 IC-LoRA 节点中选中它，强度设为 `1.0`。
5. 连接 RGB 参考视频，**清空提示词**，保留单阶段生成设置。
6. 运行并保存 matte，再在合成软件中作为原视频的 alpha 使用。

输入宽高为 32 的倍数，帧数为 `8n+1`。优先使用不超过 121 帧的片段；145 帧是模型卡上界，超过 121 帧可能降低稳定性。超过 145 帧可能出现 RGB 内容泄漏到 matte，需要拆分素材。

本章保存的是通用官方工作流原文件，导入后要选择 Alpha Gen 适配器并清空原有提示词。工作流输出普通视频对象，需要进一步合成或导出带透明通道的文件。

## 七、路线 B：固定版本的 Python 推理

下面采用模型卡推荐的 `ltx_pipelines.ic_lora` 蒸馏版路线。命令使用 Linux / Bash；Windows 读者可优先选择 ComfyUI，或把路径和多行命令改为自己环境的写法。

### 7.1 准备环境和权重

把源码、模型和输出放在教程仓库之外。需要 NVIDIA GPU、与驱动匹配的 PyTorch/CUDA 环境，以及可用的 FFmpeg。

```bash
export LTX_ROOT=/path/to/large_disk/ltx-alpha-gen
mkdir -p "$LTX_ROOT"
git clone https://github.com/Lightricks/LTX-2.git "$LTX_ROOT/LTX-2"
cd "$LTX_ROOT/LTX-2"
git checkout e98f6d927d83c92ef2be6aec76c8440b19306e39
uv sync --extra natten
source .venv/bin/activate

hf auth login
hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --local-dir "$LTX_ROOT/models/ltx-2.5"

hf download Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen \
  ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors \
  --local-dir "$LTX_ROOT/models/alpha-gen"
```

如所在实验室使用专门的资源下载入口，让 `uv` 依赖准备和 `hf download` 通过该入口执行。上面给出通用命令，不修改系统代理。`natten` 加速后端依赖平台与 CUDA；其他平台的回退行为见[官方安装与优化说明](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-pipelines/docs)。

### 7.2 预处理一段小视频

选择主体清楚、无镜头切换、至少约 5.1 秒的源视频，设置 `INPUT_VIDEO`。下面缩放时保持宽高比，再补边到 960×544，取 24fps 的前 121 帧：

```bash
export INPUT_VIDEO=/path/to/input.mp4
mkdir -p "$LTX_ROOT/runs"
ffmpeg -i "$INPUT_VIDEO" \
  -vf "fps=24,scale=960:540:force_original_aspect_ratio=decrease,pad=960:544:(ow-iw)/2:(oh-ih)/2" \
  -frames:v 121 -an -c:v libx264 -crf 18 -pix_fmt yuv420p \
  "$LTX_ROOT/runs/input_960x544.mp4"

ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_read_frames \
  -of default=noprint_wrappers=1 "$LTX_ROOT/runs/input_960x544.mp4"
```

检查结果应为宽 960、高 544、帧率 24/1、121 帧。如果源片段不足 121 帧，先选择足够长的素材，或把帧数调整为不超过实际长度的 `8n+1` 值，并同步修改后面的推理参数。

### 7.3 生成透明度视频

```bash
cd "$LTX_ROOT/LTX-2"
uv run python -m ltx_pipelines.ic_lora \
  --transformer-path "$LTX_ROOT/models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
  --text-encoder-path "$LTX_ROOT/models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path "$LTX_ROOT/models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --audio-vae-path "$LTX_ROOT/models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --spatial-upsampler-path "$LTX_ROOT/models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --lora "$LTX_ROOT/models/alpha-gen/ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors" 1.0 \
  --video-conditioning "$LTX_ROOT/runs/input_960x544.mp4" 1.0 \
  --prompt "" --width 1920 --height 1088 \
  --num-frames 121 --frame-rate 24 --seed 1234 \
  --skip-stage-2 --output-path "$LTX_ROOT/runs/matte_960x544.mp4"
```

**为什么输入是 960×544，命令却填 1920×1088？** 这个通用管线的第一阶段按请求尺寸的一半生成；`--skip-stage-2` 跳过后续放大，所以实际输出回到 960×544。本章已用下载源码中的尺寸表达式检查这组关系。若直接把输入尺寸填进这两个 CLI 参数，得到的 matte 会比输入更小。

即使提示词为空，发布接口仍要求文本编码器；该通用接口也要求音频 VAE 与空间放大器路径。更换分辨率时，保持预处理视频、实际 matte 和合成时的 RGB 帧逐一对应。

### 7.4 导出带透明通道的视频

以下将预处理后的 RGB 与 matte 合并成 ProRes 4444 MOV。先确认两个视频同尺寸、同帧数、同帧率：

```bash
ffmpeg -i "$LTX_ROOT/runs/input_960x544.mp4" \
  -i "$LTX_ROOT/runs/matte_960x544.mp4" \
  -filter_complex "[0:v]format=rgb24[rgb];[1:v]format=gray[a];[rgb][a]alphamerge[v]" \
  -map "[v]" -an -c:v prores_ks -profile:v 4 -pix_fmt yuva444p10le \
  "$LTX_ROOT/runs/foreground_alpha.mov"
```

导入支持 alpha 的剪辑或合成软件，分别换浅色和深色背景观察边缘。半透明处的原始 RGB 已经混入旧背景，直接合成有时会留下背景残色，可按素材增加去污染或前景颜色恢复。普通视频播放器显示黑底，不代表透明通道一定丢失，应在支持 alpha 的工具中检查。

## 八、沿着哪些源码继续读

下面链接固定到本章阅读的提交。通用 V2V 配置可从 `packages/ltx-trainer/configs/v2v_ic_lora.yaml` 开始。

| 文件 | 关键位置 | 重点理解 |
| --- | --- | --- |
| [ic_lora.py](https://github.com/Lightricks/LTX-2/blob/e98f6d927d83c92ef2be6aec76c8440b19306e39/packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py) | `DEFAULT_IC_LORA_STAGES`、`_resolve_active_stages` | 第一阶段带参考条件；跳过第二阶段后的尺寸关系 |
| [iclora_utils.py](https://github.com/Lightricks/LTX-2/blob/e98f6d927d83c92ef2be6aec76c8440b19306e39/packages/ltx-pipelines/src/ltx_pipelines/iclora_utils.py) | `append_ic_lora_reference_video_conditionings` | 读取参考视频、VAE 编码、构造参考条件 |
| [reference_video_cond.py](https://github.com/Lightricks/LTX-2/blob/e98f6d927d83c92ef2be6aec76c8440b19306e39/packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py) | `VideoConditionByReferenceLatent.apply_to` | 参考 token 的拼接、位置编码和去噪 mask |
| [flexible.py](https://github.com/Lightricks/LTX-2/blob/e98f6d927d83c92ef2be6aec76c8440b19306e39/packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py) | `_initialize_noisy_target`、`_apply_reference_condition`、`_compute_modality_loss` | 目标加噪、参考保持干净、损失只计算目标 |
| [alpha_gen.py](https://github.com/Lightricks/LTX-2/blob/e98f6d927d83c92ef2be6aec76c8440b19306e39/packages/ltx-pipelines/src/ltx_pipelines/alpha_gen.py) | `AlphaGenPipeline` | 新增的 full checkpoint 单阶段入口，直接接受目标尺寸 |
| [encode.py](https://github.com/Lightricks/LTX-2/blob/e98f6d927d83c92ef2be6aec76c8440b19306e39/packages/ltx-pipelines/src/ltx_pipelines/utils/media_io/encode.py) | SDR 视频保存逻辑 | 默认保存 H.264/yuv420p 视频，透明打包是后续合成步骤 |

注意两个 Python 入口的区别：**本章命令沿用模型卡推荐的 `ic_lora + distilled + skip-stage-2`**；源码中的 `alpha_gen.py` 则说明采用 full model，单阶段直接按请求宽高生成。两条路径的模型和尺寸约定不同，不要把参数混用。

默认蒸馏第一阶段有 8 个去噪区间。最新通用管线另有 `stream_chunks` 和 `chunk_config` 等分段能力，默认仍是一整个窗口。处理长视频时按 Alpha Gen 文档拆分短段，再检查相邻段的主体选择和边界连续性。

## 九、完成这个小练习

1. 找一段单主体素材，生成 matte，与原视频逐帧对齐播放。
2. 用同一 matte 换浅、深两种背景，观察毛发、运动模糊和半透明区域。
3. 记录 GPU 型号、分辨率、帧数、量化与卸载设置，以及完整耗时。改变一个设置，再比较画面和速度。

如果要比较不同方法，统一输入片段、输出尺寸和计时范围。有透明度真值时可计算 SAD、MSE、Gradient、Connectivity 等误差；没有真值时，先保留原图、matte、合成预览和时间记录，清楚展示实际效果。

继续学习：[SAM 与深度融合](./01-sam和深度估计.md) · [视觉语言目标定位](./04-Locate-Anything视觉语言定位.md) · [LingBot-Map 视频建图](./03-LingBot-Map视频流式三维重建.md)。

## 参考与文件来源

- [Alpha Gen 官方模型卡](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen)，0.9 检查点；[官方功能文档](https://docs.ltx.io/open-source-model/vfx-post-production/alpha-gen)。
- Yoav HaCohen 等，[*LTX-2: Efficient Joint Audio-Visual Foundation Model*](https://arxiv.org/abs/2601.03233)，arXiv:2601.03233，2026。
- [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)，本章阅读版本 `e98f6d927d83c92ef2be6aec76c8440b19306e39`，v1.5.0。
- 本章 ComfyUI JSON 为上游工作流原文件，来源提交和 SHA-256 见 [SOURCE.json](./assets/ltx-alpha-gen/SOURCE.json)；使用时遵循上游项目与模型许可。模型权重及第三方源码不随本教程分发。
- [TryH3 样例耗时](https://tryh3.com/generator/ltx-alpha-gen)，仅用于说明离线处理的时间量级。
