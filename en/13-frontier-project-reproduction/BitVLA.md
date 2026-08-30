# BitVLA Training and Evaluation on LIBERO

BitVLA compresses selected weights in a vision-language-action model to one-bit representations. This chapter uses the LIBERO spatial task suite to cover environment setup, base-model conversion, a 10,000-step fine-tuning run, and closed-loop evaluation.

## 1. Expected Outputs

After completing the chapter, the workspace should contain:

- BitVLA and OpenVLA-OFT source trees;
- a converted BitVLA base model;
- LIBERO dataset statistics;
- a 10,000-step fine-tuned checkpoint;
- per-task LIBERO evaluation logs and videos.

The main upstream resources are:

- BitVLA: <https://github.com/ustcwhy/BitVLA>
- OpenVLA-OFT: <https://github.com/moojink/openvla-oft>
- LIBERO: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- Base model: <https://huggingface.co/hongyuw/bitvla-bitsiglipL-224px-bf16>

## 2. Workspace and Environment

Define portable source, model, dataset, and output directories. Store large artifacts on a volume with sufficient capacity.

```bash
export WORK_ROOT=${WORK_ROOT:-$HOME/bitvla-workspace}
export SRC_ROOT="$WORK_ROOT/src"
export MODEL_ROOT="$WORK_ROOT/models"
export DATA_ROOT="$WORK_ROOT/datasets"
export RUN_ROOT="$WORK_ROOT/runs"
mkdir -p "$SRC_ROOT" "$MODEL_ROOT" "$DATA_ROOT" "$RUN_ROOT"
```

Create an isolated environment and install PyTorch. The following example uses CUDA 12.4; select the appropriate package source for another compute platform.

```bash
conda create -n bitvla python=3.10 -y
conda activate bitvla

python -m pip install \
  torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Clone the source repositories and install the three local packages:

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

Verify the key modules before starting a long run:

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

## 3. Download and Convert the Base Model

Download the base model, then run the project conversion script to install the BitVLA-specific configuration and model implementation files.

```bash
cd "$SRC_ROOT/BitVLA"
git clone \
  https://huggingface.co/hongyuw/bitvla-bitsiglipL-224px-bf16 \
  "$MODEL_ROOT/bitvla-bitsiglipL-224px-bf16"

python convert_ckpt.py \
  "$MODEL_ROOT/bitvla-bitsiglipL-224px-bf16"
```

Inspect the converted directory. In addition to weights and configuration files, it should contain the custom model implementation. A warning that `bitvla_for_action_prediction.py` or `configuration_bit_vla.py` is missing means that only the weights were downloaded and conversion is incomplete.

```bash
find "$MODEL_ROOT/bitvla-bitsiglipL-224px-bf16" \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

## 4. Prepare LIBERO Data

This experiment uses `libero_spatial_no_noops`. The dataset must provide images, proprioceptive state, actions, language instructions, and dataset statistics. Follow the current [OpenVLA-OFT LIBERO instructions](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) for downloading and conversion.

```bash
export LIBERO_DATA_ROOT="$DATA_ROOT/modified_libero_rlds"
test -d "$LIBERO_DATA_ROOT/libero_spatial_no_noops"
```

The data pipeline is ready when the training log loads `dataset_statistics.json` and prints the action dimension, proprioceptive dimension, and action-chunk length. The recorded run used action dimension 7, proprioceptive dimension 8, and action-chunk length 8.

## 5. Fine-Tune for 10,000 Steps

Keep the output under `$RUN_ROOT` so that the configuration, dataset statistics, and checkpoint remain together.

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

Argument names can change between repository revisions. Run `python vla-scripts/finetune.py --help` and map the same semantics to the checked-out version rather than deleting unknown options.

The recorded single-GPU run used batch size 1 and learning rate `1e-4`. It completed 10,000 steps in about 1 hour 27 minutes at approximately 1.94 steps per second, then saved the model and `dataset_statistics.json`. These measurements estimate runtime; closed-loop evaluation determines policy quality.

### Checkpoint 1: Confirm That Training Updates the Model

After training starts, confirm that:

1. the step counter advances and the loss remains finite;
2. the accelerator remains active without persistent data stalls;
3. the output directory contains configuration and dataset statistics, followed by checkpoint files at the configured interval.

## 6. Closed-Loop LIBERO Evaluation

Evaluation must load the checkpoint produced by the preceding training run, not the original base model.

```bash
export BITVLA_CHECKPOINT="$RUN_ROOT/<your-run-directory>"

cd "$SRC_ROOT/BitVLA/openvla-oft"
python experiments/robot/libero/run_libero_eval_bitnet.py \
  --pretrained_checkpoint "$BITVLA_CHECKPOINT" \
  --task_suite_name libero_spatial \
  --info_in_path "$RUN_ROOT/libero_spatial_eval" \
  --model_family bitnet
```

A formal record should include the task name, episode count, success count, random seed, and video directory. Training loss describes optimization behavior; task success must be measured by closed-loop execution in LIBERO.

### Checkpoint 2: Confirm the Evaluated Checkpoint

At the beginning of the evaluation log, verify that both the model path and dataset-statistics path come from `$BITVLA_CHECKPOINT`. A model path that still points to the base-model directory indicates that the fine-tuned policy is not being evaluated.

## 7. Common Problems

### NumPy Binary Interface Mismatch

An error similar to the following usually indicates that NumPy and a compiled extension were built against different interfaces:

```text
RuntimeError: module compiled against API version ...
```

Pin NumPy according to the project dependencies, then reinstall packages that compile against NumPy:

```bash
python -m pip install --force-reinstall 'numpy<2'
python -m pip check
```

### Missing Custom Model Files

If the log reports that `bitvla_for_action_prediction.py` or `configuration_bit_vla.py` cannot be found, rerun the model conversion and confirm that `transformers` is installed from the BitVLA working tree. Editing `config.json` alone does not provide the custom model implementation.

### Distributed Process Cleanup Warning

If the process group is not destroyed when training exits, add distributed cleanup to the training entry point. The warning may appear after checkpoint saving, so verify the exit code and checkpoint contents before using the run.

### Low Closed-Loop Success Rate

Check the model directory, dataset statistics, action normalization, camera keys, and task instructions in that order. First save a video for one fixed task and seed, verify the action directions and gripper control, and only then expand to the full suite.

## 8. Summary

The essential reproduction contract connects base-model conversion, LIBERO dataset statistics, fine-tuning outputs, and closed-loop evaluation through one consistent interface. The 10,000-step run verifies parameter updates and checkpoint creation; per-task closed-loop evaluation measures the resulting policy.
