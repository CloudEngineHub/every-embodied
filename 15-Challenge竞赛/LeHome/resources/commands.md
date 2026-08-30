# LeHome 常用命令速查

先定义目录：

```bash
export LEHOME_ROOT=${LEHOME_ROOT:-$HOME/lehome-challenge}
export TUTORIAL_ROOT=${TUTORIAL_ROOT:-/path/to/every-embodied}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/path/to/lehome-outputs}
```

## 下载资产

```bash
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

## 下载合并训练集

```bash
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

## 训练 ACT

```bash
mkdir -p "$OUTPUT_ROOT/train/act_top_short"

lerobot-train \
  --config_path "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_act_every_embodied.yaml" \
  2>&1 | tee "$OUTPUT_ROOT/train/act_top_short/train.log"
```

## 训练 DP

```bash
mkdir -p "$OUTPUT_ROOT/train/dp_top_short"

lerobot-train \
  --config_path "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_dp_every_embodied.yaml" \
  2>&1 | tee "$OUTPUT_ROOT/train/dp_top_short/train.log"
```

## 评测 ACT

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

## 评测 DP

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

## 解析训练日志并画图

```bash
python "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/scripts/plot_train_metrics.py" \
  --log_file "$OUTPUT_ROOT/train/act_top_short/train.log" \
  --out_dir "$OUTPUT_ROOT/plots/act_top_short" \
  --title "ACT Top-Short Training Metrics"
```

## 比赛强化版 ACT 长训

```bash
cd "$LEHOME_ROOT"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --config_path "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_act_competition_l40.yaml" \
  2>&1 | tee "$OUTPUT_ROOT/train/act_four_types_l40.log"
```

## 采样 GPU 指标

```bash
"$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/scripts/sample_gpu_metrics.sh" \
  "$OUTPUT_ROOT/monitor/act_four_types_l40_gpu.csv" \
  10
```

## 画 GPU 曲线

```bash
python "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/scripts/plot_gpu_metrics.py" \
  --csv_file "$OUTPUT_ROOT/monitor/act_four_types_l40_gpu.csv" \
  --out_png "$OUTPUT_ROOT/plots/act_four_types_l40_live/gpu_metrics.png" \
  --title "L40 GPU Monitor During ACT Training"
```
