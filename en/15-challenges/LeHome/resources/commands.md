# LeHome Command Reference

Define the portable directories first:

```bash
export LEHOME_ROOT=${LEHOME_ROOT:-$HOME/lehome-challenge}
export TUTORIAL_ROOT=${TUTORIAL_ROOT:-/path/to/every-embodied}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/path/to/lehome-outputs}
```

## Download Assets

```bash
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

## Download the Merged Training Dataset

```bash
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

## Train ACT

```bash
mkdir -p "$OUTPUT_ROOT/train/act_top_short"

lerobot-train \
  --config_path "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_act_every_embodied.yaml" \
  2>&1 | tee "$OUTPUT_ROOT/train/act_top_short/train.log"
```

## Train Diffusion Policy

```bash
mkdir -p "$OUTPUT_ROOT/train/dp_top_short"

lerobot-train \
  --config_path "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_dp_every_embodied.yaml" \
  2>&1 | tee "$OUTPUT_ROOT/train/dp_top_short/train.log"
```

## Evaluate ACT

```bash
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path "$OUTPUT_ROOT/train/act_top_short/checkpoints/last/pretrained_model" \
  --dataset_root Datasets/example/top_short_merged \
  --garment_type top_short \
  --num_episodes 2 \
  --enable_cameras \
  --save_video \
  --video_dir "$OUTPUT_ROOT/eval/act_top_short" \
  --device cpu
```

## Evaluate Diffusion Policy

```bash
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path "$OUTPUT_ROOT/train/dp_top_short/checkpoints/last/pretrained_model" \
  --dataset_root Datasets/example/top_short_merged \
  --garment_type top_short \
  --num_episodes 2 \
  --enable_cameras \
  --save_video \
  --video_dir "$OUTPUT_ROOT/eval/dp_top_short" \
  --device cpu \
  --policy_device cpu \
  --policy_num_inference_steps 1
```

## Parse and Plot Training Metrics

```bash
python "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/scripts/plot_train_metrics.py" \
  --log_file "$OUTPUT_ROOT/train/act_top_short/train.log" \
  --out_dir "$OUTPUT_ROOT/plots/act_top_short" \
  --title "ACT Top-Short Training Metrics"
```

## Run the Competition-Scale ACT Training

```bash
cd "$LEHOME_ROOT"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --config_path "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_act_competition_l40.yaml" \
  2>&1 | tee "$OUTPUT_ROOT/train/act_four_types_l40.log"
```

## Sample GPU Metrics

```bash
"$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/scripts/sample_gpu_metrics.sh" \
  "$OUTPUT_ROOT/monitor/act_four_types_l40_gpu.csv" \
  10
```

## Plot GPU Metrics

```bash
python "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/scripts/plot_gpu_metrics.py" \
  --csv_file "$OUTPUT_ROOT/monitor/act_four_types_l40_gpu.csv" \
  --out_png "$OUTPUT_ROOT/plots/act_four_types_l40_live/gpu_metrics.png" \
  --title "L40 GPU Monitor During ACT Training"
```
