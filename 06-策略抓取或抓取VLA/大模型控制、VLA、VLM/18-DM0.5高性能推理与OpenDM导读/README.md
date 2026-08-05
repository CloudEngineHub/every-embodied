# DM0.5：高性能具身 VLA 与 OpenDM 开源导读

> 本文定位：读懂 DM0.5 的公开模型结构、推理路径和高性能优化公告，并给出当前 OpenDM 能够落地的安装、推理、微调与评测入口。
>
> 重要边界：DM0.5 是 VLA，不是世界模型。本篇放在现有 VLA 章节中；文中会单独讨论它的推理优化，但不会把“低延迟 VLA 推理”误写成世界模型 rollout。

## 1. 先给结论：这次更新到底是什么

DM0.5 是原力灵机 Dexmal 面向开放世界的具身视觉语言动作模型。官方公开的 `OpenDM` 仓库已经包含模型权重、训练脚本、推理服务、数据注册范例和 LIBERO/RoboTwin2.0 评测指南；当前公开模型卡里可以看到 `Dexmal/DM05`、`Dexmal/DM05-libero`、`Dexmal/DM05-robotwin2` 和 `Dexmal/DM05-SO101-Pick-Cube` 等模型入口。

这次更新的重点不是“又换了一个更大的 VLM”，而是把在线执行链路逐段压缩：

```text
图像/状态/指令
    -> Vision TensorRT
    -> Gemma3 VLM prefix + KV cache
    -> Action Expert suffix / Flow Matching
    -> action chunk
    -> 机器人控制接口
```

官网性能公告给出的核心数据是：在 RTX 4090D 上，模型核心延迟从 `534.04 ms` 降至 `57.49 ms`，累计加速 `9.29×`；API 延迟为 `p50 67.75 ms`、`p90 70.01 ms`。公告还称，在相同 checkpoint 和 LIBERO 输入条件下，2000 个 episode 的成功率从 `98.45%` 变为 `98.40%`。这些是官方公告披露的数字，不是本教程在本机重新测出的结果。

真正需要学习的地方是：**如何在模型行为基本不变的前提下，把 VLA 的固定形状计算、低精度矩阵乘、融合算子和 GPU 图执行组合成一个低延迟在线服务。**

## 2. 官方资源与当前开源状态

| 资源 | 作用 | 当前状态 |
| :--- | :--- | :--- |
| [OpenDM 官方仓库](https://github.com/dexmal/opendm) | DM0.5 权重、训练、推理、数据注册和评测入口 | 已公开，Apache-2.0 |
| [OpenDM 中文 README](https://github.com/dexmal/opendm/blob/main/README.zh-CN.md) | 中文安装、模型和命令说明 | 已公开 |
| [Dexmal/DM05 模型卡](https://huggingface.co/Dexmal/DM05) | 6B 级 DM0.5 权重与模型文件 | 已公开 |
| [DM0.5 官方技术博客](https://www.dexmal.com/blog/dm0.5/index.html) | 官方模型介绍和性能公告入口 | 官方项目页 |
| [Dexbotic 官方仓库](https://github.com/dexmal/Dexbotic) | Dexmal VLA 工具箱、DM0 文档和父代路线参考 | 已公开 |
| [DM0 实时推理文档](https://github.com/dexmal/Dexbotic/blob/main/docs/DM0RealtimeInference.md) | 了解 DM0 的实时后端和 CUDA Graph 约束 | 已公开 |
| [realtime-vla](https://github.com/dexmal/realtime-vla) | DM0/Pi0/Pi05 的 Triton 推理参考代码 | 已公开，但主要对应 DM0/Pi0/Pi05 |
| [DM0.5 架构源码](https://github.com/dexmal/opendm/blob/main/opendm/model/dm05/dm05_arch.py) | 公开模型类、action expert、KV cache、Flow Matching 推理 | 已公开 |
| [DM0.5 实验脚本](https://github.com/dexmal/opendm/blob/main/opendm/exp/dm05_exp.py) | 训练和评测实验配置 | 已公开 |

### 2.1 “官网有代码吗”的准确回答

有，但需要分成两层理解：

1. **DM0.5 模型和训练/推理代码已经公开。** OpenDM 可以用于下载 checkpoint、启动 inference server、注册自定义数据集、做 SFT/LoRA，并按照官方指南评测 LIBERO 和 RoboTwin2.0。
2. **公告里的 9.29× 专用优化栈没有在当前 OpenDM 中完整出现。** 我们核对了当前公开源码：可以看到 `FlexAttention`、fused attention、固定 action chunk 等模型级路径，但没有看到把公告中 `Vision TensorRT + FP8 E4M3 Prefix MLP + Triton Suffix fused kernels + startup CUDA Graph` 完整串起来的 DM0.5 部署实现，也没有看到 `57.49`、`534.04` 或 `9.29` 这些性能基准对应的脚本。

所以本教程不会把 `realtime-vla` 的 DM0 Triton 代码冒充成 DM0.5 9.29× 代码。`realtime-vla` 很有参考价值，它公开了 DM0 的 Triton kernel、权重转换和 CUDA Graph 推理入口；但它只能证明“相关低延迟技术路径有公开参考”，不能证明“DM0.5 公告全部优化已经开源”。

## 3. 先看图：DM0.5 的产品信号与父代路线

![DM0.5 官方项目页头图](assets/dm05_header_zh.png)

**图 1：DM0.5 官方项目页头图。** 图中用擦拭、抓取、缠绕、推动、滚动、提取等动作，传达 DM0.5 面向开放世界、多任务操作和泛化能力的产品定位。它是宣传/项目页视觉，不是网络结构图。

来源：[OpenDM 官方仓库 `docs/image/header-zh.png`](https://github.com/dexmal/opendm/blob/main/docs/image/header-zh.png)。

![DM0 父代架构参考](assets/dm0_parent_architecture.png)

**图 2：Dexbotic 仓库中的 DM0 父代架构图。** 这张图展示的是 DM0 路线的公开父代参考：视觉编码器把观测变成视觉 token，再与机器人状态和指令一起进入 LLM；ActionExpert 负责把多模态上下文转成 Subtask、目标框、末端执行器轨迹以及离散/连续动作。它帮助我们理解 Dexmal 的“VLM + Action Expert”路线，但不能当作 DM0.5 的官方 exact diagram。

来源：[Dexbotic 官方仓库 `resources/dm0_arch.png`](https://github.com/dexmal/Dexbotic/blob/main/resources/dm0_arch.png)。

## 4. DM0.5 到底是不是 pi0.5

不是。这里有两个容易混淆的名字：

- `pi0.5` 是 Physical Intelligence 的另一条 VLA 模型路线。
- `DM0.5` 是 Dexmal 的 DM 系列模型，OpenDM 公开源码显示它基于 Gemma3 VLM 和自定义 Action Expert 构建。

因此不能因为名称里都有 `0.5`，就把两者当成同一个基座，也不能把 `realtime-vla` 中的 Pi0/Pi05 代码直接套到 DM0.5 上。对于 DM0.5，应该先读 OpenDM 的 `DM05Config`、`DM05ActionExpert`、`DM05Model.forward()` 和 `inference_action()`。

## 5. 公开源码对应的架构拆解

下面这张图是根据 OpenDM 当前公开源码整理的教学图。它不是官方论文图，目的是把代码中的 prefix、suffix、KV cache 和 Flow Matching 过程放在一张图里。

![DM0.5 公开源码架构图](assets/dm05_public_architecture.svg)

**图 3：根据 OpenDM `dm05_arch.py` 整理的公开架构链路。** 关键是 prefix 条件编码和 suffix 动作生成之间的复用关系：prefix 先把图像、指令、状态编码成条件上下文并缓存，动作 diffusion 的每个时间步主要运行 action suffix。

### 5.1 Prefix：把观测变成可复用的条件上下文

DM0.5 的输入不是单一图像，而是一个机器人决策样本：

```text
多视角图像 + 任务文本 + proprioception/state + robot type
```

视觉输入先经过视觉编码器，再进入 Gemma3 VLM 的多模态处理路径。文本指令和机器人状态按照 OpenDM 的 processor、dataset transform 和 prompt 约定组织，最终形成 prefix token 序列。

prefix 的输出有两类用途：

1. 产生当前观测的多模态 hidden states，用作动作专家的条件信息。
2. 产生并保存 VLM 的 KV cache，让后面的动作生成步骤不需要重复把相同图像和文本再过一遍完整 VLM。

这一步是理解低延迟优化的钥匙。假设一个 action chunk 需要 10 个 Flow Matching steps，如果每个 step 都重复做视觉编码和语言 prefix，计算量会非常浪费；如果 prefix 只做一次，后续只处理 action suffix，才能把动作采样循环压到几十毫秒量级。

### 5.2 Suffix：Action Expert 不是简单的线性动作头

OpenDM 源码中的 `DM05ActionExpert` 继承自 Gemma3 的文本模型结构，但它承担的是动作 suffix 的工作。它不是只把最后一个 hidden state 过一层 MLP，而是让动作 token/动作向量在一个 transformer action expert 中与 prefix 条件交互。

公开代码能确认的模块包括：

- `action_in_proj`：把连续动作向量投影到语言模型隐藏维度。
- `time_mlp_in` 与 `time_mlp_out`：把 Flow Matching 时间步编码成调制信号。
- 每层的时间调制器：让 action expert 知道当前在噪声轨迹的哪个时间位置。
- prefix KV cache：让 suffix 访问图像、文本、状态形成的条件上下文。
- `action_out_proj`：把 action expert 的 hidden states 投影回动作维度，得到速度场预测。

因此 DM0.5 的动作生成可以理解成：

```text
连续动作 chunk x_t
    -> action_in_proj
    -> 时间调制的 Action Expert
    -> action_out_proj
    -> 速度场 v_theta(x_t, t | c)
```

其中 `c` 就是 prefix 编码得到的图像/语言/状态条件。

### 5.3 Flow Matching：训练和推理分别在做什么

令真实动作 chunk 为 `a`，随机噪声为 `epsilon`，公开实现采用线性插值构造中间状态：

```text
x_t = t * epsilon + (1 - t) * a
u_t = epsilon - a
```

模型接收 `x_t`、时间步 `t` 和观测条件 `c`，预测速度场 `v_theta(x_t, t | c)`。训练目标可以概括为带 action mask 的速度场回归：

```text
L = MSE(v_theta(x_t, t | c), u_t)
```

这里的 `action mask` 很重要：训练损失只对有效动作维度和有效时间片计算，避免 padding 或无效维度干扰动作专家学习。

推理时没有真实动作 `a`，所以从随机 `x_1` 出发，反复调用 action suffix 估计速度，再用 Euler 更新把噪声逐渐积分到动作轨迹：

```text
x_{t-dt} = x_t + v_theta(x_t, t | c) * dt
```

公开实现默认 `diffusion_steps=10`，最终返回一段 action chunk。也就是说，VLA 一次推理输出的是一段动作序列，而不是单个关节动作；机器人执行其中一部分后，再读取新观测重新推理，这就是常见的 receding-horizon / action chunking 闭环。

## 6. 9.29× 加速到底优化了哪些环节

公告给出的思路可以按在线推理链路分成四层。要注意：下面的“技术解释”对应公告的描述；其中只有公开 OpenDM 源码已经能直接验证的部分，才标记为“公开源码可见”。

| 层级 | 公告中的技术 | 它解决什么问题 | 当前公开状态 |
| :--- | :--- | :--- | :--- |
| Vision | Checkpoint-specific TensorRT engine、FP16、static I/O、fixed image-count contract | 把视觉编码器变成固定输入形状的高效 engine，减少动态图和 Python 调度 | 当前 OpenDM 未公开完整 DM0.5 TensorRT engine 构建代码 |
| Prefix attention | Tuned FlexAttention | 针对固定的 prefix mask、序列长度和 attention 访问模式减少通用实现开销 | OpenDM 模型配置可见 FlexAttention 入口，但不是完整部署优化包 |
| Prefix MLP | FP8 E4M3 Tensor Core gate/up/down GEMM | 用 FP8 Tensor Core 加速 MLP 中 gate/up/down 三个大矩阵乘 | 当前 OpenDM 未检索到对应 DM0.5 FP8 kernel/校准脚本 |
| Suffix | Triton fused kernels | 把多个逐元素、投影、归一化或小算子融合，减少 kernel launch 和中间张量读写 | `realtime-vla` 有 DM0 Triton 参考；当前 OpenDM 未出现公告完整 suffix 实现 |
| Runtime | Startup-captured CUDA Graph、static-address buffers、one replay | 把固定形状在线执行从多次 kernel launch 压缩成一次 graph replay | Dexbotic 的 DM0 realtime 文档有类似固定步数 CUDA Graph 约束；DM0.5 9.29×专用代码未在 OpenDM 完整公开 |

### 6.1 Vision TensorRT：为什么要 checkpoint-specific

TensorRT engine 不是“装上 TensorRT 就自动变快”。它往往需要根据 checkpoint、输入尺寸、图像数量和数据类型构建或选择 engine。

公告强调的四个词有实际含义：

- **checkpoint-specific**：权重、层结构、量化/精度策略和 engine 绑定，不能随便拿另一模型的 engine。
- **FP16 execution**：视觉编码阶段使用半精度执行，降低显存带宽和 Tensor Core 计算成本。
- **static I/O**：输入输出 buffer 形状固定，便于内存复用、地址稳定和 CUDA Graph 捕获。
- **fixed image-count contract**：部署端必须固定每次输入几路图像；如果有时 2 路、有时 3 路，静态 engine 和 graph 都会失效或需要多份 profile。

因此它对在线机器人系统的启发不是“都换 TensorRT”，而是：**先把观测协议固定下来，再为固定协议做 engine。**

### 6.2 FlexAttention：优化的不是模型语义，而是 attention 执行形状

VLA 的 prefix 和 suffix 具有不同的 attention 结构。prefix 需要处理图像 token、文本 token 和状态 token；suffix 需要访问 prefix cache，同时处理当前 action chunk 的 token/向量。如果每次调用通用 attention，可能会为 mask、shape、KV layout 和不同阶段付出额外开销。

FlexAttention 允许实现者描述更贴合模型结构的 attention 访问模式，再由 PyTorch 编译/生成高效执行路径。OpenDM 当前源码已经保留了 VLM LLM 和 action expert 对 `flex_attention`/`sdpa` 等实现的选择入口，但这不等于公告中的全部 tuned kernel 已经公开。

### 6.3 FP8 E4M3：为什么只压 MLP 也可能收益很大

Transformer 中的 MLP 通常包含大规模的 gate/up/down 投影矩阵乘，算力和显存访问都很重。公告针对 prefix MLP 的 gate/up/down 使用 FP8 E4M3 Tensor Core GEMM，核心收益来自：

1. 权重和激活占用更少的带宽。
2. 4090D 等 GPU 的 Tensor Core 对低精度矩阵乘吞吐更高。
3. 把 FP8 集中放在对吞吐最敏感的矩阵乘上，而不是把所有模块无差别量化。

但 FP8 并不是把 `dtype=torch.float8_e4m3fn` 写上去就结束。还需要处理 scale、动态范围、累加精度、权重布局、校准数据和输出误差；如果没有官方校准及 kernel 实现，读者不应把这个部分当成当前 OpenDM 的可直接复现步骤。

### 6.4 Triton Suffix fused kernels：降低的是调度和访存成本

动作 suffix 往往在每个 Flow Matching step 内重复执行。即使每个小算子单独只耗很短时间，很多 kernel launch、临时 tensor 写回和 Python/框架调度叠加后，延迟也会明显增加。

Triton fused kernel 的典型目标是把一串固定操作合成一个 GPU kernel，例如：

```text
读取输入 -> 归一化/逐元素调制 -> 投影或 GEMM -> 激活 -> 写回
```

实际融合边界需要结合模型维度、寄存器占用、occupancy、L2 命中率和数值误差来定，不能只看 kernel 数量。官方公开的 `realtime-vla/dm0_infer.py` 提供了 DM0 的 Triton 参考，适合学习如何拆解视觉编码器、LLM 和 action expert 的 GEMM-like operations；它不是 DM0.5 官方优化包的替代品。

### 6.5 CUDA Graph：为什么要求启动期捕获、静态地址和一次 replay

CUDA Graph 适合“形状固定、控制流固定、内存地址固定”的推理循环。部署时先用固定形状的 dummy buffer 运行一次并 capture graph，在线阶段只把新观测搬进已经分配好的静态 buffer，再 replay。

公告给出的在线阶段可以压缩成：

```text
静态数据搬运 -> TensorRT engine 执行 -> 一次 graph replay
```

这会减少大量 runtime launch overhead，但也带来约束：

- image count、分辨率、prompt/token 长度和 action chunk shape 需要固定或分桶。
- graph 捕获时使用的 CUDA stream、buffer 地址和执行路径必须保持稳定。
- 不能在线临时改变 diffusion steps；改变步数通常需要重新 capture。
- 复杂的动态 Python 分支、动态内存分配或异常处理不能随意放进 graph。

Dexbotic 已公开的 DM0 realtime 文档也体现了同一类工程约束：服务启动时捕获固定 step 数的 graph，调用端的 sampling steps 必须与 capabilities 返回值一致。这里可以作为理解 CUDA Graph 的相邻实践，但不能把 DM0 的 5.5× benchmark 与 DM0.5 的 9.29× benchmark 直接横向比较。

## 7. 官方公开 OpenDM 怎么跑

本节给的是“读者可以按官方仓库继续操作”的入口，不把它包装成本文已经替读者完成的完整训练复现。

### 7.1 环境前提

OpenDM 当前 README 推荐 Linux、NVIDIA GPU 和 Docker/NVIDIA Container Toolkit；训练通常需要多卡，推理可以单卡。具体 CUDA、PyTorch、显存和镜像版本以仓库当前 README 为准。

建议先确认：

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

不要先下载 11 GB 级 checkpoint 再检查环境。先验证 GPU、驱动、容器 runtime 和 Hugging Face 网络可用。

### 7.2 获取代码和模型

```bash
git clone https://github.com/dexmal/opendm.git
cd opendm

# 使用 Hugging Face CLI 下载公开权重，实际文件较大，请预留磁盘空间
hf download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

仓库也提供 Docker 相关文件和脚本。推荐优先沿用官方容器/启动脚本，不要一开始就自行升级 PyTorch、FlashAttention、xformers 或 CUDA 版本；VLA 推理中的 attention、图像处理和自定义 kernel 对版本比较敏感。

### 7.3 启动 DM0.5 推理服务

官方 README 的服务入口类似：

```bash
script/dm05_launcher.sh \
  --task inference \
  --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --inference-config.port 7891
```

然后使用仓库中的 curl demo：

```bash
bash tests/curl_demo.sh http://127.0.0.1:7891/process_frame
```

请求通常需要提供任务文本、机器人状态和图像字段，`robot_type`、`control_mode`、`speed` 等字段是否必需取决于模型和部署模式。不要直接复制旧版本示例的 JSON：先查看当前 `tests/curl_demo.sh`、processor 和 README，确保图像数量、key 顺序、state 维度与 checkpoint 的 normalization stats 一致。

### 7.4 SFT/LoRA 的公开入口

OpenDM 当前已公开数据注册范例和多个微调指南。通用训练入口类似：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name my_robot \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --trainer-config.num-train-steps 50000
```

真正开始训练前需要完成：

1. 按 `opendm/dataset/` 的范例注册数据集。
2. 确认 episode 的图像字段、状态、动作和语言指令格式。
3. 让自动生成的 normalization stats 与 checkpoint、机器人本体和动作维度匹配。
4. 先用很小的 step 数跑数据加载和 forward smoke test，再扩大训练。
5. 使用 `docs/en/dm05_finetuning.md`、`docs/en/dm05_libero_lora_training.md` 和 `docs/en/dm05_so101_lora_training.md` 的当前配置，而不是使用网络文章中的旧命令。

这条链路能复现的是 DM0.5 的公开训练/微调能力；它不等于能复现官网 9.29× 的部署 kernel。

## 8. 能复现什么，不能复现什么

| 目标 | 当前建议 | 结论 |
| :--- | :--- | :--- |
| 下载 DM0.5 权重 | `Dexmal/DM05` | 可以 |
| 启动单卡 inference server | OpenDM `dm05_launcher.sh` | 可以，需满足官方环境 |
| 注册自定义机器人数据 | dataset registration + normalization stats | 可以，先做小规模 smoke test |
| SFT/LoRA | 官方 playground 和 finetuning 文档 | 可以，训练成本取决于数据/显卡 |
| LIBERO/RoboTwin2.0 评测 | 官方 benchmark guide | 可以按仓库当前版本尝试 |
| 复现 534.04 -> 57.49 ms | 公告未给出完整公开 benchmark harness 和 DM0.5 专用优化包 | 当前不能声称一键复现 |
| 复现 FP8 E4M3 prefix MLP | 需要官方 kernel、scale/校准和权重转换细节 | 当前 OpenDM 不足以直接复现 |
| 复现 TensorRT + Triton Suffix + CUDA Graph 全链路 | 需要对应部署工程、固定 shape contract 和测试脚本 | 当前没有在 OpenDM 中找到完整实现 |

## 9. 性能数字应该怎样读

### 9.1 核心延迟和 API 延迟不是一个指标

公告同时给出模型核心延迟和 API 延迟，说明两者测量范围不同：

- **模型核心延迟**：更接近 GPU 上模型计算本身。
- **API 延迟**：还可能包含请求解析、数据搬运、图像预处理、tokenization、后处理、网络/服务调度等开销。

所以 `57.49 ms` 和 `p50 67.75 ms` 并不矛盾。写实验报告时必须注明计时边界、warm-up 次数、batch、图像路数、prompt 长度、diffusion steps、chunk size、GPU 型号、驱动/CUDA/PyTorch 版本。

### 9.2 98.45% -> 98.40% 说明优化没有明显改变行为，但不等于通用性能证明

LIBERO 2000 episodes 的成功率基本不变，是对“推理优化没有显著伤害任务表现”的有价值证据；但它仍然只覆盖特定 checkpoint、输入协议和 benchmark。要判断真实机器人部署效果，还应测：

- 传感器帧率与延迟抖动。
- action chunk 执行期间的观测新鲜度。
- p50/p90/p99，而不是只看平均值。
- CPU 到 GPU 的搬运、图像预处理和 action postprocess。
- 丢帧、相机阻塞、GPU 温度/频率变化和长时间运行稳定性。
- 不同图像数量、prompt 长度和机器人本体的 shape contract。

## 10. 和 DM0 realtime、OpenDM 的关系

| 项目 | DM0.5/OpenDM | Dexbotic DM0 realtime / realtime-vla |
| :--- | :--- | :--- |
| 主要对象 | DM0.5 VLA 与训练/推理生态 | DM0 及 Pi0/Pi05 的实时推理参考 |
| 公开内容 | 权重、模型结构、SFT/LoRA、评测和服务 | Triton kernel、权重转换、benchmark、CUDA Graph 实现参考 |
| 适合学习 | VLA 结构、Flow Matching、数据和评测 | 低延迟 kernel 拆解、固定 shape 和 graph replay |
| 是否等于 DM0.5 9.29×实现 | 不等于 | 不等于，但技术路线相邻 |

这两个仓库可以一起读：先用 OpenDM 理解“DM0.5 要算什么”，再用 `realtime-vla` 理解“DM0 这类 VLA 如何把计算拆成 Triton kernel 并做实时推理”。但移植时要重新确认模型层数、隐藏维度、视觉分辨率、动作维度、权重布局、norm stats 和 processor 协议。

## 11. 推荐阅读顺序

如果暂时不做训练，建议按下面顺序阅读：

1. 先读 [OpenDM README](https://github.com/dexmal/opendm)，确认模型、权重和评测入口。
2. 再读本文图 2，理解 Dexmal 的 VLM + Action Expert 父代路线。
3. 打开 [`dm05_arch.py`](https://github.com/dexmal/opendm/blob/main/opendm/model/dm05/dm05_arch.py)，重点搜索 `DM05ActionExpert`、`action_in_proj`、`action_out_proj`、`forward` 和 `inference_action`。
4. 结合图 3，手工追一遍 prefix KV cache、动作噪声、Flow Matching 和 Euler steps。
5. 再读 [DM0 realtime 文档](https://github.com/dexmal/Dexbotic/blob/main/docs/DM0RealtimeInference.md)，理解静态 shape 和 startup CUDA Graph。
6. 最后读 [`realtime-vla`](https://github.com/dexmal/realtime-vla) 的 `dm0_infer.py` 和 `benchmark.py`，观察一个真实 Triton runtime 如何组织 vision、LLM、action expert 和 graph。

## 12. 复现时最容易踩的坑

### 12.1 把 DM0.5 和 pi0.5 混为一谈

名称相似不代表基座相同。DM0.5 应以 OpenDM 的 Gemma3 + Action Expert 实现为准，pi0.5 应以 OpenPI/Physical Intelligence 的实现为准。

### 12.2 只看模型核心延迟，不看闭环延迟

机器人是否能实时工作，取决于从相机曝光到控制指令发出的全链路。务必把图像采集、预处理、推理、反归一化、控制周期、网络和执行器延迟分别记录。

### 12.3 动态输入破坏静态优化

图像数量、分辨率、token 长度、chunk size 和 diffusion steps 任意变化，都会让 TensorRT profile、kernel specialization 或 CUDA Graph 失效。部署设计应先确定输入 contract，再决定是否做静态编译和 graph capture。

### 12.4 normalization stats 和动作维度不匹配

动作归一化是 VLA 能否正常执行的基础。自定义数据集不能只把图像和动作文件放进去，还要确认动作维度、机器人状态维度、有效 action mask 和 `norm_stats.json` 的来源。

### 12.5 用旧教程的命令跑新仓库

OpenDM 仍在快速更新。训练脚本、模型目录、权重名称和文档会变化。复现时以当前仓库 README、`docs/en/` 和脚本帮助信息为准，并记录 commit hash。

## 13. 最后总结

DM0.5 这次更新值得放进 VLA 教程，原因不是“9.29× 是一个新的 VLA 算法”，而是它把具身模型落地时最容易被忽略的一层讲得很具体：**模型结构、输入协议、低精度矩阵乘、算子融合、静态内存和 CUDA Graph 必须一起设计。**

对学习者来说，可以把它拆成三个层次：

1. **算法层**：Gemma3 VLM prefix + Action Expert suffix + Flow Matching action chunk。
2. **模型层**：prefix KV cache、FlexAttention、时间调制、action projection 和 Euler sampling。
3. **部署层**：TensorRT、FP8、Triton fused kernels、固定输入 shape、启动期 CUDA Graph 和端到端 latency measurement。

当前 OpenDM 已经足够支持 DM0.5 的模型阅读、权重下载、推理服务、SFT/LoRA 和公开 benchmark 入口；公告中的 9.29× 专用优化栈仍应标注为官方性能更新，等待对应部署代码和完整 benchmark harness 公开后再做严格复现。这个边界本身就是教程需要告诉读者的工程事实。
