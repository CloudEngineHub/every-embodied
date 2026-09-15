---
license: cc-by-nc-sa-4.0
base_model: InternRobotics/InternVLA-N1-DualVLN
tags:
  - robotics
  - navigation
  - vlnverse
  - research-artifact
---

# VLNVerse InternVLA 粗指令适配权重：2026-06-22 实验归档

该包保存一次粗指令侧重训练的增量参数，用于复现训练与退化诊断。它不是最终获奖模型，也不是零样本细指令 48.7% 成绩对应的模型。

原始模型：[InternRobotics/InternVLA-N1-DualVLN](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN)。训练数据来自 [VLNVerse](https://huggingface.co/datasets/Eyz/VLNVerse_data)。本包由 Datawhale 参赛协作实验归档，保留原始模型和数据的署名、非商业使用、相同方式共享要求。

## 训练设置

8 张 H100；3,000 个优化器步；有效批大小 64；学习率 `1e-5`；粗指令训练数据加 30% 细指令训练数据；语言主体冻结。615 个可训练张量，约 6.809 亿参数。参数覆盖视觉融合层、语言输出头及双系统相关模块，不是仅训练小型低秩矩阵。

## 评估记录

同一粗指令 10 条验证中，原始模型成功率 20%，本包 40%；计划扩大到 50 条时，实际完成 12 条后因质量较差停止，成功 1 条，即 8.33%。细指令 10 条成功率 30%，对应原始模型官方链路为 70%。局部测试不能证明整体泛化收益。

## 文件与加载

`trainable_params.bin` 是增量张量字典，不包含冻结的全部基础权重。其散列为：

```text
ea9201197c3150317efefd703bbf749c11a42ea3164b0a5589f41a13813d6990
```

权重以 32 MiB 分片存储在 `weights/`，每片及完整文件的散列记录在 `weight_manifest.json`。下载仓库后执行以下命令，恢复字节一致的增量文件；需要额外约 1.36 GB 可用空间，目标文件必须尚不存在：

```bash
python code/restore_adapter.py /path/to/downloaded-repository
```

先加载原始模型，再覆盖增量参数。比赛集成中的 `model_path` 指向基础模型，`trainable_params_path` 指向增量文件。调用接口需使用支持该参数的适配实现。张量键由 `trainable_params.txt` 列出。缺失的冻结参数属于预期，额外键应逐一核查。

原始加载接口的概念示例如下；需先安装官方 InternNav 并导入其模型类：

```python
model = InternVLAN1ForCausalLM.from_pretrained(base_model_path)
delta = torch.load(adapter_path, map_location="cpu", weights_only=True)
load_result = model.load_state_dict(delta, strict=False)
print(load_result.unexpected_keys)
```

压缩的比赛结果包不是本权重的推理产物，不要互相替代。教程和复盘见 [Every Embodied](https://github.com/datawhalechina/every-embodied)。

本仓库内可直接阅读 [比赛经验分享](https://huggingface.co/Datawhale/vlnverse-internvla-coarse-adapter-0622/blob/main/competition_experience_zh.md) 和 [逐条评估摘要](https://huggingface.co/Datawhale/vlnverse-internvla-coarse-adapter-0622/blob/main/results/local_evaluations.json)。
