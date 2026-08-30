# 01 Confirmation of AMD ROCm Devices and Environment

This task first checks whether the device is truly suitable for subsequent experiments. Many replication failures seem to be model-related issues, but in reality, they may be due to unready ROCm version, memory allocation, cache path, network, or Hugging Face permissions.

Supporting practical notebook: [01_device_env_check.ipynb](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/notebooks/01_device_env_check.ipynb).

## Learning Objectives

After completing this task, you can directly judge:

- Check whether the AMD GPU / APU is correctly recognized by ROCm;
- Determine if there are still sufficient resources for memory, system memory, and temperature during training;
- Store the dataset, checkpoint, and Hugging Face cache on the appropriate disk;
- Prepare the basic connections for remote SSH, proxy, and token;
- Prepare a device resource table that facilitates comparison of experimental results.

## Device Resource Check

Run on AMD devices:

```bash
rocm-smi --showuse --showtemp --showmemuse
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY
```

In the experiment record, it must be clearly written at least:

| Item | Example Entry |
| --- | --- |
| Device Model | AMD Ryzen AI MAX+ 395 / Radeon GPU |
| System Version | Ubuntu 24.04 / WSL / Others |
| ROCm Version | e.g., 7.x |
| PyTorch Version | e.g., ROCm build |
| Idle Temperature | e.g., 30 to 45 degrees Celsius |
| Training Temperature | e.g., 75 to 85 degrees Celsius |
| Idle VRAM | e.g., 9% |
| Training VRAM | Record the usage of ACT / SmolVLA / pi_0 respectively |

## Understanding of Unified Memory and Video Memory

Some AMD APU devices use a unified memory architecture. It is important to note that the operating system, GPU, data loading processes, and browser may all share the same physical memory. A large VRAM allocation does not necessarily ensure stable training; if available system memory is constrained, data loading or model initialization may also fail.

Use experimental records rather than guesses to determine whether resources are sufficient:

```bash
watch -n 2 'rocm-smi --showuse --showtemp --showmemuse; free -h'
```

If training has not yet begun, the GPU utilization is 0%, but the memory of the Python process continues to increase, usually when loading models or downloading weights. If the GPU utilization remains at 100% for a long time, and the temperature rises while the loss decreases normally, it indicates normal training.

## Large File Directory Planning

To enable reuse of commands on different machines, do not hardcode personal machine paths in notes and scripts. Instead, use variables uniformly:

```bash
export PROJECT_ROOT=/path/to/every-embodied/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA
export DATA_ROOT=/path/to/datasets/every_embodied
export HF_HOME=/path/to/cache/huggingface
export CHECKPOINT_ROOT=$PROJECT_ROOT/ckpt
```

Large files include:

- Hugging Face model cache;
- LeRobot dataset;
- checkpoint;
- rollout video;
- batch evaluation JSONL;
- intermediate logs.

These files are usually very large in size. Just clearly indicate the storage location and purpose in the experiment record. In tutorials and reports, simply retain the path variables, directory planning, and cleaning methods.

## Network and Hugging Face Permissions

pi_0 relies on a gated model, such as `google/paligemma-3b-pt-224`. Confirm first:

1. The Hugging Face account has accepted the gated model terms;
2. The token has permission to read the public gated repository;
3. The remote machine can access Hugging Face;
4. The token is not present in the command line, logs, or Notebook outputs.

It is recommended to use the smoke script to verify permissions first, rather than starting long training directly.

## Checkpoint

When completing this task, save a table of reproduction-ready experimental resources:

```markdown
| 项目 | 结果 |
| --- | --- |
| ROCm 能否识别 GPU | 是 / 否 |
| torch.cuda.is_available | True / False |
| 空闲温度 |  |
| 空闲 VRAM |  |
| 数据目录 | 使用变量，说明数据放在哪类磁盘 |
| HF gated 权限 | 已通过 / 未通过 |
```
