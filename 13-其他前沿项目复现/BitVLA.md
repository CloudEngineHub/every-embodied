# BitVLA 在 LIBERO 上的训练与评估

BitVLA 将视觉语言动作模型中的部分权重压缩为一比特表示，用于研究低精度模型在机器人操作任务中的效率与性能。本章以 LIBERO 为空间任务集，完成环境安装、基础模型转换、万步微调和闭环评估，并整理复现中最常见的依赖与模型文件问题。

## 1. 实验产出

完成本章后，实验目录中应包含：

- BitVLA 与 OpenVLA-OFT 源码环境；
- 转换后的基础模型目录；
- LIBERO 数据集统计文件；
- 一份万步微调检查点；
- LIBERO 逐任务评估日志和视频。

本章使用的主要上游入口如下：

- BitVLA 项目：<https://github.com/ustcwhy/BitVLA>
- OpenVLA-OFT：<https://github.com/moojink/openvla-oft>
- LIBERO：<https://github.com/Lifelong-Robot-Learning/LIBERO>
- 基础模型：<https://huggingface.co/hongyuw/bitvla-bitsiglipL-224px-bf16>

## 2. 目录和环境

先定义源码、模型、数据和输出目录。大型文件建议放到容量充足的数据盘。

```bash
export WORK_ROOT=/path/to/bitvla-workspace
export SRC_ROOT="$WORK_ROOT/src"
export MODEL_ROOT="$WORK_ROOT/models"
export DATA_ROOT="$WORK_ROOT/datasets"
export RUN_ROOT="$WORK_ROOT/runs"
mkdir -p "$SRC_ROOT" "$MODEL_ROOT" "$DATA_ROOT" "$RUN_ROOT"
```

创建独立环境并安装 PyTorch。下面的命令使用 CUDA 12.4；其他计算平台需要选择对应的软件包来源。

```bash
conda create -n bitvla python=3.10 -y
conda activate bitvla

python -m pip install \
  torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

克隆源码并安装三个本地包：

```bash
cd "$SRC_ROOT"
git clone https://github.com/ustcwhy/BitVLA.git
cd BitVLA

python -m pip install -e openvla-oft
python -m pip install -e transformers
python -m pip install -e bitvla

cd openvla-oft
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
python -m pip install -e LIBERO
python -m pip install -r experiments/robot/libero/libero_requirements.txt
```

安装后先检查关键模块，避免等到训练启动时才发现环境缺失：

```bash
python - <<'PY'
import torch
import transformers
import libero

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("libero:", libero.__file__)
print("gpu:", torch.cuda.is_available())
PY
```

## 3. 下载并转换基础模型

下载基础模型后，使用项目提供的转换脚本补齐 BitVLA 自定义配置与模型文件：

```bash
cd "$SRC_ROOT/BitVLA"
git clone \
  https://huggingface.co/hongyuw/bitvla-bitsiglipL-224px-bf16 \
  "$MODEL_ROOT/bitvla-bitsiglipL-224px-bf16"

python convert_ckpt.py \
  "$MODEL_ROOT/bitvla-bitsiglipL-224px-bf16"
```

转换完成后检查模型目录。除权重和配置外，还应存在项目自定义的模型实现文件；若日志提示 `bitvla_for_action_prediction.py` 或 `configuration_bit_vla.py` 不存在，说明只下载了权重，没有完成源码转换。

```bash
find "$MODEL_ROOT/bitvla-bitsiglipL-224px-bf16" \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

## 4. 准备 LIBERO 数据

本章使用 `libero_spatial_no_noops`。数据应包含图像、本体状态、动作、语言指令和数据统计。具体下载方式以 OpenVLA-OFT 的 [LIBERO 说明](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md)为准。

准备完成后，将数据路径指向统一目录：

```bash
export LIBERO_DATA_ROOT="$DATA_ROOT/modified_libero_rlds"
test -d "$LIBERO_DATA_ROOT/libero_spatial_no_noops"
```

如果训练日志能够加载 `dataset_statistics.json`，并打印动作维度、本体状态维度和动作块长度，说明数据字段已经进入训练管线。本次记录中的配置为：动作维度 7、本体状态维度 8、动作块长度 8。

## 5. 万步微调

训练命令以 BitVLA 仓库当前脚本参数为准。输出目录单独放在 `$RUN_ROOT`，便于保存配置、数据统计和检查点：

```bash
cd "$SRC_ROOT/BitVLA/openvla-oft"

python vla-scripts/finetune.py \
  --vla_path "$MODEL_ROOT/bitvla-bitsiglipL-224px-bf16" \
  --data_root_dir "$LIBERO_DATA_ROOT" \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir "$RUN_ROOT" \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --max_steps 10000 \
  --image_aug True
```

不同版本的仓库可能调整参数名。运行前先执行 `python vla-scripts/finetune.py --help`，把上述语义映射到当前版本，不要直接删掉未知参数。

本次万步记录使用单卡、批大小 1 和学习率 `1e-4`，训练约 1 小时 27 分钟，平均约 1.94 步每秒。日志在达到最大步数后正常保存模型和 `dataset_statistics.json`。这些数字用于估算同类硬件上的运行时间，不替代闭环评估。

### 检查点 1：训练确实更新模型

训练开始后确认三项内容：

1. 进度条持续增加，损失值为有限数；
2. 计算设备保持占用，且没有持续的数据等待；
3. 输出目录生成配置和数据统计，保存周期到达时出现检查点文件。

## 6. LIBERO 闭环评估

评估必须加载刚才生成的训练目录，而不是重新指向基础模型：

```bash
export BITVLA_CHECKPOINT="$RUN_ROOT/<your-run-directory>"

cd "$SRC_ROOT/BitVLA/openvla-oft"
python experiments/robot/libero/run_libero_eval_bitnet.py \
  --pretrained_checkpoint "$BITVLA_CHECKPOINT" \
  --task_suite_name libero_spatial \
  --info_in_path "$RUN_ROOT/libero_spatial_eval" \
  --model_family bitnet
```

正式记录至少包含任务名称、回合数、成功数、随机种子和视频目录。训练损失只说明优化过程是否正常，任务成功率必须由 LIBERO 环境在闭环运行中统计。

### 检查点 2：评估加载了正确检查点

在评估日志开头核对模型路径和数据统计路径，并确认两者都来自 `$BITVLA_CHECKPOINT`。如果模型路径仍指向基础模型目录，当前结果就不是微调模型的评估结果。

## 7. 常见问题

### NumPy 二进制接口不匹配

报错包含下面信息时，通常是 NumPy 版本与已编译扩展不一致：

```text
RuntimeError: module compiled against API version ...
```

先按照依赖文件固定 NumPy，再重装依赖 NumPy 二进制接口的包：

```bash
python -m pip install --force-reinstall 'numpy<2'
python -m pip check
```

### 自定义模型文件缺失

如果日志提示找不到 `bitvla_for_action_prediction.py` 或 `configuration_bit_vla.py`，重新执行模型转换脚本，并确认当前 `transformers` 来自 BitVLA 工作树。只修改 `config.json` 无法替代自定义模型实现。

### 分布式进程退出警告

训练完成后若出现进程组未销毁警告，应在训练入口结束前调用分布式清理函数。该警告通常发生在保存完成之后，但仍应通过退出码和检查点文件确认本次训练是否完整结束。

### 成功率明显偏低

依次检查模型目录、数据统计、动作归一化、相机键名和任务指令。先用一个固定任务和固定种子保存视频，确认动作方向与夹爪控制正确，再扩大到完整任务集。降低评估分母不会修复接口错位。

## 8. 本章小结

BitVLA 的复现重点不只是安装一比特模型，而是保证基础模型转换、LIBERO 数据统计、微调输出和闭环评估使用同一套接口。万步训练记录证明训练管线能够完成参数更新和检查点保存；最终模型质量则由逐任务闭环结果决定。
