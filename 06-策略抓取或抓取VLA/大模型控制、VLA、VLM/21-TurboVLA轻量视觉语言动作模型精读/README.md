# TurboVLA 论文精读：轻量视觉—语言交互如何实现实时操作

TurboVLA 用约 0.2B 参数，在 LIBERO 上取得 **97.6% 平均成功率、31.2 ms 策略推理延迟和 0.9 GiB 推理显存占用**；采用更大视觉主干的 0.4B 版本，在 RoboTwin 2.0 的 50 个双臂任务上取得 **88.06% 平均成功率**。它提供了一条很实用的设计路线：任务目标已经明确时，用轻量语言编码与视觉交互支撑高频动作生成。

本章沿着“问题 → 网络与张量 → 实验 → 训练资源 → 代码与复现”的顺序精读论文。结果采用 **arXiv v3，2026-09-27**；代码核对版本为 [`a3365e8`](https://github.com/H-EmbodVis/TurboVLA/tree/a3365e8f1787d30bafd99d2806edbf4dc82b5370)，查阅日期为 2026-10-23。文中实验数字来自作者报告，复现命令按公开接口整理。

## 1. 论文与官方资源

| 资源 | 入口 | 阅读用途 |
| --- | --- | --- |
| 论文 | [TurboVLA v3 正文与附录](https://arxiv.org/html/2607.27205v3)、[PDF](https://arxiv.org/pdf/2607.27205v3) | 架构、实验、训练与计时协议 |
| 源码 | [H-EmbodVis/TurboVLA](https://github.com/H-EmbodVis/TurboVLA) | 训练、模型、数据和评测 |
| 权重 | [Hugging Face 模型仓库](https://huggingface.co/H-EmbodVis/TurboVLA) | LIBERO/RoboTwin checkpoint 和归一化元数据 |
| 项目页 | [TurboVLA 官方主页](https://h-embodvis.github.io/TurboVLA/) | 真机视频、模型与场景示意 |
| 复现指南 | [LIBERO](https://github.com/H-EmbodVis/TurboVLA/blob/a3365e8f1787d30bafd99d2806edbf4dc82b5370/experiments/libero/README.md)、[RoboTwin 2.0](https://github.com/H-EmbodVis/TurboVLA/blob/a3365e8f1787d30bafd99d2806edbf4dc82b5370/experiments/robotwin/README.md) | 环境、数据格式、启动与评测参数 |

论文题目为 *TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM*，arXiv 编号为 `2607.27205`。此处按公开预印本介绍。

**版本提示：**项目页和部分模型卡仍保留 LIBERO 97.7%、RoboTwin Clean50 60.2% 等旧版结果。本章统一使用 v3 的 97.6% / 88.06%，各子任务数字也采用同一版本。

## 2. 核心问题：执行操作时，大模型应放在哪里

很多 VLA 把图像和语言送入大型多模态骨干，再生成离散动作 token，或接入连续动作专家。TurboVLA 重新选择了执行策略的信息路径：

```text
多视角 RGB → DINOv3 → 视觉投影 ─┐
                               ├→ 6 层双向视觉—语言交互 ─┐
任务指令   → 冻结 BERT → 文本投影 ┘                       │
                                                        ├→ ACT 风格解码器 → H 步连续动作
当前机器人状态 → 状态投影 → 2 个状态 token ────────────────┘
```

例如，指令已给定“把碗放到盘子上”，执行策略首先需要找到碗和盘子、判断当前机械臂状态并预测动作。TurboVLA 让这些信息直接交互，以紧凑的动作头完成控制。论文将其概括为从 `V → L → A` 转向 `V + L → A`。

可以结合[官方范式与性能对照图](https://github.com/H-EmbodVis/TurboVLA/blob/a3365e8f1787d30bafd99d2806edbf4dc82b5370/assets/figures/paradigm-and-performance.png)阅读两种路径的区别。图内结果须结合上文版本提示阅读。

它保留了语言、预训练表征和 Transformer；节省的主要是执行链路中大型生成式语言骨干的开销，以及自回归或多轮采样式动作生成的开销。对教程读者，值得学习的是**依据执行任务重新分配模型容量**。

## 3. 模型结构与张量流

![TurboVLA 多视角、语言与状态到动作块的教学结构图](./assets/turbovla-architecture.svg)

图 1：本教程根据论文第 3 节与公开模型代码绘制的教学图。视觉与指令分别编码，经双向交互后，与机器人状态一起送入动作解码器。对照查看[官方架构原图](https://github.com/H-EmbodVis/TurboVLA/blob/a3365e8f1787d30bafd99d2806edbf4dc82b5370/assets/figures/architecture-overview.png)。

### 3.1 视觉编码：保留物体与空间信息

DINOv3 从每个相机画面提取 patch token，再投影到共享的 `d=256` 维特征空间，并加入位置、视角信息。保留空间 token，使后续模块能够选择与目标物体、属性和位置关系相关的局部信息。

| 设置 | LIBERO | RoboTwin 2.0 |
| --- | --- | --- |
| 视觉骨干 | DINOv3 ViT-B/16 | DINOv3 ViT-L/16 |
| RGB 视角 | 主视角、腕部，共 2 路 | 头部、左右腕部，共 3 路 |
| 输入分辨率 | 每路 256×256 | 每路 224×224 |
| 状态维度 | 8 | 14 |
| 动作维度 | 7 | 14 |
| 预测动作长度 H | 12 | 50 |

RoboTwin 的动作是双臂绝对关节位置和夹爪指令；LIBERO 的 7 维动作契约与它不同。迁移时应先对齐动作含义、归一化和机器人接口。

### 3.2 语言编码：冻结参数，仍参与在线推理

语言分支采用冻结的 `BERT-base-uncased`，保留 token 级特征并投影到 256 维。当前实现在线完成 tokenizer 和 BERT 前向；“冻结”指参数不更新。发布的 `online_text_layout.json` 保存文本 padding 元数据，没有缓存语言特征。

这可以理解为：BERT 提供指令表征，后续交互模块负责把指令中的目标与当前画面对应起来。

### 3.3 六层双向交互：让两个模态互相查询

设初始视觉、语言特征分别为 $V^0$ 和 $L^0$，用六层交互更新：

$$
V^{k+1}=V^k+\operatorname{Attn}_{v\leftarrow l}(V^k,L^k),
\qquad
L^{k+1}=L^k+\operatorname{Attn}_{l\leftarrow v}(L^k,V^k).
$$

前者让画面特征受到任务指令调制，后者让语言特征融入当前场景信息。上式概括信息流；实现还包含归一化、残差、缩放与文本 Transformer 层。交互模块沿用 GroundingDINO 风格的结构，并使用其 feature-enhancement 预训练权重初始化。

阅读源码时，重点观察每个分支的 query/key/value 来自哪里，以及交互后哪些 token 交给动作头。

### 3.4 动作头：一次前向产生一个动作块

当前状态单独投影为两个状态 token，与交互后的视觉和文本 token 拼接。三层 ACT 风格 Transformer Decoder 接收 $H$ 个可学习 action query，并行输出：

$$
\hat A_t=D_\theta\left(Q_a,[V^6;L^6;Z_t^s]\right)
\in\mathbb R^{H\times d_a}.
$$

输出经过 MLP 和 `tanh`，训练采用带有效步 mask 的 L1 行为克隆损失。轨迹末尾不足一个 chunk 时，补齐位置从损失中排除。这里的 ACT 风格主要指动作块解码结构，训练目标按 TurboVLA 的公开实现读取。

## 4. 实验结果：小模型获得了怎样的性能

### 4.1 LIBERO：四套任务统一训练

论文用一个混合策略覆盖 Spatial、Object、Goal、Long 四套任务。每套 10 个任务，每任务 50 次测试，共 2,000 次。

| Spatial | Object | Goal | Long | 平均 |
| ---: | ---: | ---: | ---: | ---: |
| 97.0% | 100.0% | 98.2% | 95.2% | **97.6%** |

下面摘录 v3 表 1 的部分对照。延迟与显存采用论文的 RTX 4090 测量口径。

| 模型 | 参数量 | 推理显存 | 延迟 | 平均成功率 |
| --- | ---: | ---: | ---: | ---: |
| TurboVLA | 0.2B | 0.9 GiB | 31.2 ms | **97.6%** |
| π0.5 | 3.4B | 12.8 GiB | 93.6 ms | 96.9% |
| VLA-Adapter | 1.5B | 4.3 GiB | 87.3 ms | 97.3% |
| DDVLA | 7.5B | 14.5 GiB | 60.8 ms | 96.4% |
| SmolVLA | 0.5B | 1.4 GiB | 212.7 ms | 87.3% |

TurboVLA 在这组测试中取得了很强的性能与效率组合。以表中 π0.5 为例，策略延迟比为 `93.6 / 31.2 = 3.0×`。各方法的预训练数据与动作机制不同，这张表适合比较部署选择；架构机制则结合下一节的受控消融理解。

### 4.2 RoboTwin 2.0：50 个双臂任务

训练包括 2,500 条 Clean 和 25,000 条 Randomized 演示，总计 27,500 条；任务均匀采样，每任务内 Randomized:Clean 为 10:1。评测使用未见过的任务指令表述，在两种设置下各进行 `50 任务 × 100 次` 测试。

| 模型 | 参数量 | 延迟 | Clean | Randomized | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboVLA | 0.4B | 43.4 ms | 88.02% | 88.10% | **88.06%** |
| π0.5 | 3.4B | 246.5 ms | 82.74% | 76.76% | 79.75% |
| LingBot-VLA | 4.2B | 164.4 ms | 88.56% | 86.68% | 87.62% |

双臂结果说明这条紧凑执行路线能扩展到更复杂的多任务操作。论文中“没有额外具身预训练”指未使用目标基准之外的额外机器人数据预训练；模型仍使用通用视觉、文本与 grounding 预训练，并在上述机器人演示上训练。

### 4.3 真机：Piper 上的四类操作

作者使用 AgileX Piper 单臂与外部、腕部 RealSense D435 相机，测试抓滚筒、移开扑克牌、按订书机、叠三个碗。每任务采集 65 条遥操作演示，共 260 条，从 LIBERO 策略初始化后微调 12,500 步。

每任务测试 40 次，共 160 次，平均成功率 **87.5%**。这组实验展示了从仿真策略到真机任务适配的实用性。操作视频可在[官方项目页](https://h-embodvis.github.io/TurboVLA/)查看。

来源：论文 v3 第 4 节、附录 A/B/C 与表 1、表 2、真机实验。

## 5. 消融实验：哪些设计最值得借鉴

### 5.1 任务条件与语言语义

| LIBERO 条件输入 | 平均成功率 |
| --- | ---: |
| 无语言条件 | 68.5% |
| 任务 ID embedding | 96.6% |
| 自然语言指令 | **97.6%** |

条件输入让多任务策略知道“这次要做什么”，收益十分明显。任务 ID 本身也很强，说明在已知任务集合中，紧凑的任务条件已经很有效；自然语言进一步提供语义接口。若关注新物体组合、复杂新指令，可以沿此设置增加组合泛化测试。

### 5.2 双向交互的收益

| 融合方式 | LIBERO | RoboTwin 2.0 |
| --- | ---: | ---: |
| 无视觉—语言交互 | 96.3% | 82.58% |
| 语言查询视觉 | 96.8% | 86.62% |
| 视觉查询语言 | 97.2% | 87.08% |
| 完整双向交互 | **97.6%** | **88.06%** |

相对无交互版本，双向交互分别提高 **1.3 和 5.48 个百分点**。更复杂的双臂多任务场景收益更明显，是这套结构很值得复用的部分。

附录还有两组有用的设计实验：交互层数从 2/4/6 层增加时，LIBERO 成功率为 95.4%/96.9%/97.6%；chunk 长度为 8/10/12/15 时，对应 96.8%/97.3%/97.6%/96.7%。动作块长度需要在预测跨度与新观测反馈之间选择。

来源：论文 v3 第 4.3 节和附录 C.1。

## 6. 31.2 ms、43.4 ms 与 7.42 ms 怎么读

| LIBERO 运行设置 | 无 CUDA Graph | 使用 CUDA Graph |
| --- | ---: | ---: |
| 基础 TurboVLA | 31.2 ms | 11.3 ms |
| 加入数据转换、内存等工程优化 | 30.3 ms | **7.42 ms** |

43.4 ms 对应 RoboTwin 的 ViT-L、三视角设置。7.42 ms 来自 LIBERO 模型的运行时优化实验；它说明架构轻量化之后，仍可以通过减少调度和内存开销继续加速。

论文附录 B 的计时使用已在 GPU 上的输入，热身 10 次后进行 50 次同步推理，报告中位数。0.9 GiB 按 `nvidia-smi` 峰值占用减去空闲占用统计。相机采集、输入准备与传输、通信和机器人执行延迟需另测。

`1000 / 31.2 ≈ 32` 表示每秒可进行约 32 次策略调用。LIBERO 每次预测并执行 12 个动作后重新规划，因此**策略调用频率、动作执行频率和使用新视觉信息的反馈频率需要分别记录**。评估自己的系统时，建议同时记录模型延迟、完整请求延迟和实际重规划间隔。

来源：论文 v3 表 6、附录 B.2；若要复现 7.42 ms，还应补齐对应 CUDA Graph 优化实现与软件环境，常规评测命令本身不包含这整套优化。

## 7. 为什么模型这么小，训练还使用多张卡

**论文的 4×RTX 4090 / 8×H100 是作者使用的训练配方，并非模型结构规定的最低卡数。** 多卡分摊批量与训练计算，能提高吞吐量、缩短实验周期；单卡是否装得下，主要取决于每次实际处理的样本数、视觉激活与训练精度。

| 训练项目 | LIBERO | RoboTwin 2.0 |
| --- | --- | --- |
| 硬件 | 4×RTX 4090 | 8×H100 |
| 优化器步数 | 40,000 | 150,000 |
| 有效 batch | 128 | 512 |
| 每卡 microbatch | 8 | 64 |
| 梯度累积 | 4 | 1 |
| 学习率 | 5e-5 | 5e-5 |
| warmup 步数 | 10,000 | 1,000 |
| EMA decay | 0.999 | 0.999 |
| 冻结模块 | BERT 主干 | BERT 主干 |

推理只保存当前前向所需的状态；训练还需要梯度、优化器状态、反向传播激活及 EMA 副本。TurboVLA 的 DINOv3 主干参与更新，多视角图像的激活开销也要计入。以“全部可训练、FP32 参数与梯度、Adam 两份 FP32 状态”为简化例子，常驻参数相关存储约为 `16 × 参数个数` 字节，尚未计激活与 EMA；这是解释开销来源的估算，不能直接当作设备实测显存。

### 7.1 用梯度累积减少卡数

有效 batch 的关系为：

$$
B_{\mathrm{effective}}=N_{\mathrm{GPU}}\times B_{\mathrm{micro}}\times N_{\mathrm{accum}}.
$$

LIBERO 原配方为 `4 × 8 × 4 = 128`。单卡可以从 `1 × 8 × 16 = 128` 开始尝试；若单次显存不足，再降低 microbatch 并提高累积次数。RoboTwin 的同类算术示例是 `1 × 8 × 64 = 512`，需要同步修改训练器与 DeepSpeed 的累积配置。

梯度累积让多个小批次共同形成一次参数更新，通常以更多顺序计算换取较小的单次激活占用。它保留有效 batch 的大小，仍需核对采样、随机性、学习率调度和更新步数；不能按卡数直接推算准确训练时间。

RoboTwin 公开配置还使用 DeepSpeed ZeRO-2，它分摊优化器与梯度状态。减少卡数后，这部分分摊也会改变。对于 96 GB 单卡，小 microbatch 加梯度累积是合理的起步路线；先跑少量完整前后向与优化器更新，记录显存峰值和每步时间，再决定训练预算。

### 7.2 优先做哪一级复现

1. **理解与评测：**加载作者 checkpoint，先检查权重兼容与归一化，再跑一个任务或一个 suite。
2. **训练链路：**少量数据、小 microbatch 验证损失、梯度和 checkpoint 保存，然后扩展到完整配方。
3. **完整训练：**保持任务采样、有效 batch、优化器步数与评测协议，报告真实硬件和耗时。

若冻结视觉骨干、仅训练动作头或使用 LoRA，应作为新的资源节省实验记录，与作者的视觉主干全量更新设置分开比较。

来源：论文附录 A.3；官方 [LIBERO recipe](https://github.com/H-EmbodVis/TurboVLA/blob/a3365e8f1787d30bafd99d2806edbf4dc82b5370/experiments/libero/README.md)、[RoboTwin 配置](https://github.com/H-EmbodVis/TurboVLA/blob/a3365e8f1787d30bafd99d2806edbf4dc82b5370/experiments/robotwin/configs/taskbalanced_all50.yaml)及 [DeepSpeed 配置](https://github.com/H-EmbodVis/TurboVLA/blob/a3365e8f1787d30bafd99d2806edbf4dc82b5370/experiments/robotwin/configs/deepspeed_zero2.yaml)。单卡调整是依据这些配置给出的教学建议。

## 8. 代码阅读顺序

以下路径均相对[官方固定版本仓库](https://github.com/H-EmbodVis/TurboVLA/tree/a3365e8f1787d30bafd99d2806edbf4dc82b5370)。

| 路径 | 阅读重点 |
| --- | --- |
| `turbovla/models/turbovla.py` | 总模型、视觉投影、视觉—语言交互与条件编码 |
| `turbovla/models/vision_encoder.py` | DINOv3、patch token、多视角组织 |
| `turbovla/models/text_encoder.py` | 在线 BERT、冻结参数、padding mask 和投影 |
| `turbovla/models/components/fusion.py` | 双向 attention、残差及 SDPA 路径 |
| `turbovla/models/action_head.py` | 状态 token、动作 query、解码与输出范围 |
| `turbovla/training/trainer.py` | masked L1、初始化权重映射、梯度累积与调度 |
| `experiments/libero/train.py`、`evaluate.py` | 对外训练与单 suite 评测入口 |
| `experiments/robotwin/configs/taskbalanced_all50.yaml` | 任务采样、三相机、H=50 与训练参数 |
| `scripts/robotwin/train.sh`、`evaluate.sh` | 分布式训练和模拟器/策略环境衔接 |

一个容易踩到的细节：`trainer.py` 的 `--max_steps` 默认值是 **80,000**，论文 LIBERO 配方是 **40,000**。复现时显式传入步数、warmup 与累积参数，避免把命令行默认值当成论文设置。

## 9. 从官方 checkpoint 开始复现

下面命令面向 **Linux + NVIDIA CUDA**，在 TurboVLA 仓库根目录执行。LIBERO 与 RoboTwin 使用不同数据和模拟器栈，按官方指南准备独立 Python 3.10 环境。权重、数据集和训练输出保存在教程仓库之外。

### 9.1 固定源码与准备资源

```bash
git clone https://github.com/H-EmbodVis/TurboVLA.git
cd TurboVLA
git checkout a3365e8f1787d30bafd99d2806edbf4dc82b5370

# 在已激活的 LIBERO 环境中，先安装匹配设备的 CUDA PyTorch。
python -m pip install -e ".[libero]"
python -m pip install -U huggingface_hub
hf download H-EmbodVis/TurboVLA --local-dir /path/to/models/TurboVLA
```

还需按官方指南安装 LIBERO，准备本地 DINOv3 ViT-B、BERT 与相应配置；训练时另需 GroundingDINO 初始化权重和四套 no-noops RLDS 数据。完整 checkpoint 已包含模型各模块参数，仍应保留模型配置、tokenizer 和归一化元数据。DINOv3 资源按其访问与许可要求获取。

实验室本机、moon、Ubuntu 的资源与依赖下载使用既定 `LabDownload.cmd` / `lab-download` 专线入口，具体环境的 `python -m pip` 也经入口执行。仅预取资源，不把训练或评测进程整体放入下载线路。

### 9.2 先检查加载，再评测一个 suite

在下面命令后加 `--dry_run_model_load true`，先做官方提供的 checkpoint 兼容检查；检查通过后去掉该选项，运行完整 suite。

```bash
python experiments/libero/evaluate.py \
  --ckpt_path /path/to/models/TurboVLA/checkpoints/libero/turbovla_libero.pth \
  --dinov3_path /path/to/models/dinov3-vitb \
  --bert_path /path/to/models/bert-base-uncased \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --task_suite_name libero_object \
  --num_trials_per_task 50 \
  --chunk_size 12 \
  --num_open_loop_steps 12 \
  --seed 7 \
  --precision bf16 \
  --result_json_path /path/to/runs/turbovla/object.json
```

其余三套分别为 `libero_spatial`、`libero_goal`、`libero_10`。四套均完成后再汇总平均值；记录源码版本、权重版本、suite、成功次数和总次数。

### 9.3 单卡 LIBERO 训练参数示例

以下以 microbatch=8、累积=16 保持有效 batch=128。路径需替换为已准备的本地资源；这是单卡配方调整示例，耗时与显存需要实测。

```bash
torchrun --nproc_per_node=1 experiments/libero/train.py \
  --dataset_dir /path/to/data/libero/libero_10_no_noops/1.0.0 \
  --dataset_dirs "/path/to/data/libero/libero_10_no_noops/1.0.0,/path/to/data/libero/libero_goal_no_noops/1.0.0,/path/to/data/libero/libero_object_no_noops/1.0.0,/path/to/data/libero/libero_spatial_no_noops/1.0.0" \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --dinov3_path /path/to/models/dinov3-vitb \
  --bert_path /path/to/models/bert-base-uncased \
  --pretrained_init_ckpt /path/to/models/groundingdino_swint_ogc.pth \
  --batch_size 8 --grad_accum_steps 16 \
  --max_steps 40000 --warmup_steps 10000 \
  --lr 5e-5 --min_lr_ratio 1.0 \
  --checkpoint_dir /path/to/runs/turbovla/checkpoints
```

### 9.4 RoboTwin 先对齐模拟器与元数据

RoboTwin 评测指南指定模拟器 commit [`bf44be5`](https://github.com/RoboTwin-Platform/RoboTwin/commit/bf44be5)，并要求策略与模拟器各自的 Python 环境路径。checkpoint 应与对应 `config.yaml`、`dataset_statistics.json` 一起保存。

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
export STARVLA_PYTHON=/path/to/policy-env/bin/python
export ROBOTWIN_PYTHON=/path/to/robotwin-env/bin/python

# 首次先选一个任务验证接口；完整命令评测 50 个任务、两种环境设置。
bash scripts/robotwin/evaluate.sh \
  /path/to/models/TurboVLA/checkpoints/robotwin/steps_150000_ema_pytorch_model.pt
```

完整参数与任务子集语法见[官方 RoboTwin 指南](https://github.com/H-EmbodVis/TurboVLA/blob/a3365e8f1787d30bafd99d2806edbf4dc82b5370/experiments/robotwin/README.md)。先验证单任务的数据、动作与环境接口，再扩展到 10,000 次完整评测。

## 10. 如何把这篇论文用到自己的项目

TurboVLA 的价值在于，轻量的任务条件编码、明确的跨模态交互和直接动作块预测，可以形成成功率很高、推理成本很低的执行策略。它很适合作为低延迟操作系统的基线，也适合用来学习视觉、语言和动作各模块的职责。

接下来可以根据项目的实际困难选择一项扩展：

- **需要更复杂的任务理解：**将低频任务规划与高频 TurboVLA 执行结合，分别测规划与执行开销。
- **需要遮挡恢复或历史信息：**加入短时状态/视觉记忆，比较成功率与新增延迟。
- **需要动态纠错：**缩短开环执行段或研究异步调度，记录反馈间隔、动作连续性和任务成功率。

相关章节：[SmolVLA 与 LIBERO](../01SmolVLA-LIBERO/01SmolVLA-libero.md)、[EventVLA 视觉证据记忆](../06EventVLA视觉证据记忆导读/README.md)、[具身推理与部署框架](../09具身智能推理与部署框架/README.md)、[EVA-Client 真机部署与评测](../19-EVA-Client真机部署与评测工程导航/README.md)。
