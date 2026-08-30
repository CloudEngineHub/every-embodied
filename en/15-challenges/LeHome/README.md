# LeHome Challenge Competition Image Usage Guide

This document is intended for students who have obtained the official LeHome competition image and wish to start training and evaluation as soon as possible.

Focus on answering four questions:

1. Can the current image environment be used directly?
2. Should the training download `merged` or non-`merged` data?
3. How to start training, evaluation, and reproduction experiments with minimal overhead?
4. How to specify the training configuration and where to output the results?
5. Which large files should not be submitted to the tutorial repository?

Video tutorial: LeHome flexible clothing folding ICRA2026 competition video tutorial http://xhslink.com/o/2oxCz0RTXcA


<video src="https://assets.datawhale.cn/1090/dashboard/1773720966495/LeHome%E6%95%99%E7%A8%8B.mp4" width="100%" controls>
  Your browser does not support HTML5 video.
</video>



---

## 0. Portable Directory Configuration

Define the source, tutorial, and output roots once before running the commands in this chapter:

```bash
export LEHOME_ROOT=${LEHOME_ROOT:-$HOME/lehome-challenge}
export TUTORIAL_ROOT=${TUTORIAL_ROOT:-/path/to/every-embodied}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/path/to/lehome-outputs}
mkdir -p "$OUTPUT_ROOT"
```

## 1. Is the current image available?

Conclusion: Available.

The following verifications have been completed in the current image:

- `LeHome` repository path: `$LEHOME_ROOT`
- `isaacsim` available: `5.1.0.0`
- `lerobot` available: `0.4.3`
- GPU: `NVIDIA L40`
- Local HEAD: `2953e2f9c1376a1e49ac10b3ad690efb886f4c6c`
- Current remote `origin/main`: `32b53595da504880592a79ed5e362ad0ba0fac6b`
- One round of ACT testing has been successfully completed.
- One round of DP testing has been successfully completed, and `--policy_num_inference_steps 1` is in effect.
- Old failed testing videos have been cleaned up and will be output to:
  - `$OUTPUT_ROOT/eval/`

Instructions:

- The environment should be able to start up, load models, create scenes, and complete evaluations.
- The current local code is not the latest on the remote server.
- If you want to strictly align with the current image, please switch to:

```bash
git checkout 2953e2f9c1376a1e49ac10b3ad690efb886f4c6c
```

- If you want to align with the latest main branch from the remote, switch to:

```bash
git fetch origin
git checkout 32b53595da504880592a79ed5e362ad0ba0fac6b
```

- The current reproduction results are consistent with the description in the official reproduction package, but the success rate remains low. This seems to be a problem with the task itself or the physical effect, rather than the mirror not starting up.

---

## 2. Which training data to download

### 2.1 Conclusion

If the goal is to start baseline training as soon as possible, prioritize downloading:

```bash
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

That is, prioritize using the `merged` version.

### 2.2 Reasons

The official training configuration defaults to the `merged` data:

- `configs/train_act.yaml` uses `Datasets/example/top_long_merged`
- `configs/train_dp.yaml` uses `Datasets/example/top_long_merged`
- `configs/train_smolvla.yaml` uses `Datasets/example/top_long_merged`

This indicates that the training entry expected by the official is the "merged dataset".

### 2.3 The difference between `merged` and `merged`

It can be understood as follows:

- `dataset_challenge`
  - More focused on the original/dispersed data organization.
  - Suitable for your own data processing, splitting, augmentation, depth filling, and point cloud supplementation.
  - If you want to study the data production process or reconstruct your training set, you can use it.
- `dataset_challenge_merged`
  - More focused on organized results that can be directly used for training.
  - Already merged into directories such as `top_long_merged` and `pant_long_merged` by category.
  - Best for directly feeding into `lerobot-train`.

### 2.4 Should Deep Data be Downloaded?

If you just run the baseline successfully first:

- Use `merged` first
- Do not use depth initially
- Use the official RGB + joint state baseline first

Reason:

- The official documentation explicitly supports `state + RGB` as a validated combination.
- Although depth is available, it increases storage, I/O, and configuration complexity.
- The official data processing documentation also provides a process for "reducing storage by removing depth".

### 2.5 Recommended Options

Recommendation order:

1. Introduction/competition baseline: `dataset_challenge_merged`
2. Requires in-depth experiments: add `observation.top_depth` on top of `merged`
3. Rebuild the data locally: continue with `dataset_challenge` rather than `merged`.

### 2.6 Which type of task should be chosen for teaching start?

If the goal is teaching, rather than trying to cover the entire match from the start, I recommend starting with:

- `top_short`

Start.

It should be clear here: this is an educational choice in engineering, not because it has the highest success rate in the current reproduction package. In the current reproduction package, all four types of historical statistics have a success rate of `0.0%`, so it cannot be said that “the easiest to succeed” has been proven by existing results.

It is still recommended to start with `top_short` because it is more suitable for the beginning of teaching:

- Compared to `top_long`, short-sleeved shirts typically lack the long-sleeve material coupling at the distal end.
- Compared to pants, short-sleeved shirts have a clearer visual target, making it easier for learners to understand what "successful folding" means.
- For the same top-related task, `top_short` is usually more suitable than `top_long` as the first lesson.

Therefore, this tutorial follows a two-layer approach:

1. Teaching route: First run `top_short`
2. Competition route: Then complete all four types of tasks

---

## 3. Minimum startup process

## 3.1 Clone the official repository

```bash
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
```

If the image is already pre-installed, simply enter the existing directory:

```bash
cd "$LEHOME_ROOT"
```

## 3.2 Download Assets

```bash
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

## 3.3 Download Training Sample Data

```bash
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

If you only want to quickly verify a category, you can also download only the required sub-directory, for example:

```bash
hf download lehome/dataset_challenge_merged \
  --repo-type dataset \
  --local-dir Datasets/example \
  --include 'top_long_merged/**'
```

If you plan to study this tutorial in full, it is recommended to ensure that all four types of directories are present:

- `Datasets/example/top_short_merged`
- `Datasets/example/top_long_merged`
- `Datasets/example/pant_short_merged`
- `Datasets/example/pant_long_merged`

It is necessary to make a clear distinction:

- Starting point of teaching: First use `top_short_merged`
- Full competition: All four categories must be run

### 3.3.1 The Difference Between Teaching Process and Competition Process

The teaching process is not about completing all the tasks of the competition on the first day, but rather to enable learners to master them first:

- How to choose the data directory
- How to start training
- How to view logs
- Where is the model saved
- Where are the evaluation videos

So, for the teaching process, only one category is run first. `top_short` is recommended.

The competition process is different. The competition process requires you to ultimately cover four types of tasks:

- `top_short`
- `top_long`
- `pant_short`
- `pant_long`

## 3.4 First Lesson of Teaching: First, Train `top_short`'s ACT

It is recommended to use the configuration specific to the tutorial and save the output to the data disk uniformly:

- Configuration file:
  - `$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_act_every_embodied.yaml`

```bash
mkdir -p "$OUTPUT_ROOT/train/act_top_short"

lerobot-train \
  --config_path "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_act_every_embodied.yaml" \
  2>&1 | tee "$OUTPUT_ROOT/train/act_top_short/train.log"
```

This command does two things:

1. Start the training using the YAML file specific to the tutorial.
2. Use `tee` to save the terminal logs to `train.log` simultaneously, which is convenient for analyzing the loss and generating plots later.

For beginners, it is recommended not to modify the command line for the first time. Instead, understand the function of each field in YAML, and then only change one or two parameters to conduct experiments.

This YAML is already in the `top_short` teaching version, so there is no need for you to make any manual changes.

## 3.5 Lesson 2: Training DP on the same `top_short` task

It is also recommended to use the configuration specific to the tutorial:

- Configuration file:
  - `$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_dp_every_embodied.yaml`

```bash
mkdir -p "$OUTPUT_ROOT/train/dp_top_short"

lerobot-train \
  --config_path "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_dp_every_embodied.yaml" \
  2>&1 | tee "$OUTPUT_ROOT/train/dp_top_short/train.log"
```

DP is usually slower and more resource-intensive than ACT, so in the tutorial configuration:

- `batch_size` set smaller than ACT
- `steps` stretched to `90000`
- `log_freq` refined to `100`

This follows the common training habits of Diffusion Policy, and it is also more convenient to observe the convergence process.

This YAML is already in the `top_short` teaching version, so no manual editing is required.

## 3.6 Teaching Evaluation: Evaluating ACT for `top_short`

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

## 3.7 Teaching Evaluation: Evaluating the DP of `top_short`

It is recommended to explicitly limit the diffusion inference steps during CPU evaluation:

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

### 3.7.1 Why Teaching Does Not Start Directly with All Four Categories Together

Because the four categories are trained together, their memory and compute requirements accumulate for first-time users:

- Four data directories
- Four model output directories
- Four evaluation commands
- Longer training time
- More complex troubleshooting paths

The most important thing in teaching is to clearly explain a complete closed loop first. So, for this lesson, we use `top_short` as an example.

### 3.7.2 How to Implement a Complete Competition Process

A complete race does not mean just running `top_short`. It is recommended to complete the complete race by categories:

1. `top_short`
2. `top_long`
3. `pant_short`
4. `pant_long`

In other words, you should ultimately replicate the process of "training + evaluation + result summary" across the four categories.

### 3.7.3 Complete Training Organization for Four Types of Tasks

The safest and easiest-to-teach solution is:

- Train each category separately
- Evaluate each category separately
- Finally, summarize all together

Recommended directory organization:

```text
$OUTPUT_ROOT/
├── train/
│   ├── act_top_short/
│   ├── act_top_long/
│   ├── act_pant_short/
│   ├── act_pant_long/
│   ├── dp_top_short/
│   ├── dp_top_long/
│   ├── dp_pant_short/
│   └── dp_pant_long/
├── eval/
└── plots/
```

### 3.7.4 How to run the four types of tasks in order

You can treat the `top_short` process as a template, and then replace the following three values:

- `dataset.root`
- `output_dir`
- `garment_type`

For example:

- `top_short`
  - `dataset.root: Datasets/example/top_short_merged`
  - `garment_type: top_short`
- `top_long`
  - `dataset.root: Datasets/example/top_long_merged`
  - `garment_type: top_long`
- `pant_short`
  - `dataset.root: Datasets/example/pant_short_merged`
  - `garment_type: pant_short`
- `pant_long`
  - `dataset.root: Datasets/example/pant_long_merged`
  - `garment_type: pant_long`

So the most important learning point in the tutorial is not to memorize a command, but to learn:

- Changing any three places can transfer it to another type of task.

### 3.7.5 What to Organize Before Submission

The README in the current official repository does not provide a fixed command for submitting requests. It states:

- The submission instructions will be provided on the official website.

So, there is no way to fabricate a "official submission script" here. However, you can definitely organize the "materials that need to be prepared before submission" in advance.

It is recommended to prepare at least:

- Final models for each of the four types of tasks
- Evaluation logs for each of the four types of tasks
- Success rates for each of the four types of tasks
- Representative videos for each of the four types of tasks
- Training configuration YAML
- Corresponding code commits

From a tutorial perspective, this step can be understood as:

- First, organize your "competition experiment package" completely.
- Wait until the official submission format is clear before uploading.

## 3.8 Detailed Explanation of Training Configuration

In the tutorial-specific configuration, the fields that deserve the most explanation are:

- `dataset.root`
  - Training data directory.
  - The first lesson is defaulted to use `Datasets/example/top_short_merged`.
  - This design is because the official baseline configuration points directly to `*_merged`, so beginners can start practicing immediately without needing to understand the multi-dataset merging logic first.
- `policy.type`
  - `act` or `diffusion`.
  - ACT is the recommended first step, as its training logic is more intuitive and usually faster.
  - Diffusion Policy is better for comparison after you have successfully run the ACT baseline.
- `policy.device`
  - Training recommendation: `cuda`.
  - Since both the image encoder and the policy network are heavy, CPU training is almost inefficient.
- `input_features`
  - The baseline recommends `observation.state + top/left/right RGB`.
  - This is a safe starting point from an engineering perspective.
  - `state` provides robotic arm joint information, while RGB provides fabric shape and pose information.
  - For learners, this combination is also easiest to understand: one part is “the robot’s current pose”, and the other part is “the scene seen by the camera”.
- `output_features`
  - The baseline recommends outputting `action`.
  - That is, directly predicting joint movements.
  - This joint-space control is better than `ee_pose` as a teaching baseline, as it eliminates an additional IK error layer.
- `output_dir`
  - Use a data volume, such as `$OUTPUT_ROOT/train/act_top_short`.
  - This is done to avoid filling up the system disk.
- `batch_size`
  - Directly related to memory usage.
  - The ACT teaching version defaults to `64`, while the DP teaching version defaults to `48`.
  - For beginners, it can be simply understood as: how many training samples are fed to the model at a time.
  - Generally, larger values mean faster training, but higher memory usage is also required.
- `steps`
  - Total training steps.
  - ACT can first run `30000`, while DP recommends longer steps; the tutorial configuration provides `90000`.
  - Training steps are not “the larger the better”, but “enough to observe convergence trends”.
  - In teaching, it is more important to use a step that can be completed and observed.
- `save_freq`
  - Checkpoint saving interval.
  - The saving frequency is increased in the tutorial to allow review of intermediate results.
  - This is especially important for teaching, as you can compare the effects of checkpoints at different stages rather than just looking at the final model.
- `log_freq`
  - Log printing frequency.
  - The frequency is increased in the tutorial for more detailed logs, which facilitates later visualization and troubleshooting.
  - For learners, training logs serve as the most intuitive “observation window” of the training process.
- `eval_freq`
  - Intermittent evaluation interval during training.
  - Observing intermediate results is not just about changes in loss.
  - If a model’s loss decreases but the eval success rate does not increase, this usually indicates that “fitting within the training distribution has been learned”, but it may not truly understand the task.

## 3.8.1 Why the parameters of ACT and DP are different

This is a question that many learners ask for the first time.

### ACT

ACT can be understood as a transformer-based policy that predicts action sequences based on vision and state.

In this tutorial, we have given ACT a set of more aggressive parameters that are close to the maximum L40 utilization:

- `batch_size: 64`
- `steps: 30000`
- `save_freq: 5000`
- `log_freq: 200`
- `eval_freq: 5000`

Purpose of this design:

- Easier to run on a single card
- Logs are detailed enough for teaching and observation
- A checkpoint can be obtained every `5000` step
- The total number of steps `30000` is sufficient to see the basic convergence trend of a baseline

### Diffusion Policy

The inference and training of Diffusion Policy are more intensive, especially under multi-camera input.

In the tutorial configuration, we have provided:

- `batch_size: 48`
- `steps: 90000`
- `save_freq: 10000`
- `log_freq: 100`
- `eval_freq: 10000`

Purpose of this design:

- Reduce memory pressure
- Increase the number of training steps to provide sufficient convergence space for the diffusion model
- Observe loss and gradient norm with a more frequent log frequency

So don't mechanically assume that "the parameters for ACT and DP should be the same." Different policy structures should naturally have different training rates.

## 3.8.2 How long does the training take, and what is the approximate VRAM usage

The values here are empirical estimates, not absolute numbers.

Under this type of `top_short_merged + state + 3路RGB` configuration, you can estimate as follows:

### ACT

- Recommended VRAM: at least `16GB`, and more safely `24GB+`
- On cards like L40, single-card training is usually possible
- `30000 step` generally refers to "hourly" tasks, not minute-level ones
- If both `data_s` and `updt_s` in the logs are low, the complete baseline results can usually be obtained within a few hours

### DP

- Recommended video memory: `24GB+` is more stable
- `90000 step` usually lasts longer than ACT
- More suitable as a overnight training task

What learners should be concerned about are:

- Is there any OOM?
- How long does it take for each step?
- Is the loss decreasing steadily?
- Has the eval on the intermediate checkpoint actually improved?

Don't get stuck on whether it's 3 hours or 5 hours right from the start. Instead, check the log in your machine first:

- `updt_s`
- `data_s`
- `step_per_sec`

These values are the true basis for determining the training time.

## 3.8.3 What is recommended for the first training

It is recommended to learn in the following order:

1. Train ACT first, instead of starting DP.
2. Do not add depth first.
3. Do not convert to `ee_pose` first.
4. Run a category successfully first, such as `top_short_merged`.
5. Observe the loss, grad norm, lr, and intermediate eval.
6. After confirming the stability of the pipeline, expand to more categories.

## 3.9 Recommended retained training intermediate results

It is recommended to keep at least these:

- `train.log`
- All checkpoints
- `config.yaml` or training configuration copies
- Training curve graph
- Training metrics CSV / JSON summary
- If there is a periodic eval, the corresponding logs and videos are also retained

For teaching purposes, it is also recommended to add:

- A screenshot of the training curve
- A comparison of "best checkpoint" and "final checkpoint"
- A record of training duration
- A record of VRAM usage

It is recommended to place them uniformly in:

```text
$OUTPUT_ROOT/
├── train/
├── eval/
└── plots/
```

## 3.10 Training Log Parsing and Visualization

The tutorial comes with a script:

- `$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/scripts/plot_train_metrics.py`

It will parse and generate from `train.log`:

- `train_metrics.csv`
- `train_metrics_summary.json`
- `train_metrics.png`

Example:

```bash
python "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/scripts/plot_train_metrics.py" \
  --log_file "$OUTPUT_ROOT/train/act_top_short/train.log" \
  --out_dir "$OUTPUT_ROOT/plots/act_top_short" \
  --title "ACT Top-Short Training Metrics"
```

The current script defaults to recording these metrics:

- `loss`
- `grdn`, which is gradient norm
- `lr`
- `updt_s`
- `data_s`
- `step_per_sec`

The teaching significance of these indicators is as follows:

- `loss`
  - The most basic optimization objective. First, check if the overall performance drops.
- `grdn`
  - Used to check if the gradient explodes or is abnormally unstable.
- `lr`
  - Used to verify if the learning rate scheduling is normal.
- `updt_s`
  - Time taken for single-step parameter update.
- `data_s`
  - Time taken for single-step data loading.
- `step_per_sec`
  - An intuitive metric of training throughput.

## 3.11 What other metrics should be considered besides loss?

From the perspective of reproduction of ACT / Diffusion Policy papers and engineering training, in addition to loss, it is also recommended to keep records of as much as possible:

- Training loss
- Validation loss
- Gradient norm
- Learning rate
- Data loading time
- Update time
- Steps/sec or samples/sec
- Success rate from intermediate checkpoint evaluation
- Average return `return`
- Episode length
- Success rate for different garments/categories

What is truly relevant to the competition results is not just the loss values alone, but:

1. `loss` Whether it decreases steadily
2. `gradient norm` Whether there is an abnormal explosion
3. `lr` Whether scheduling is normal
4. The intermediate checkpoint `eval success rate` Whether to synchronize the upgrade

## 3.12 What is saved after training ends

After training is completed, the most important aspects to focus on are usually:

- `train.log`
  - Original training log
- `checkpoints/`
  - Models saved at each stage
- `last/`
  - Last training status
- `pretrained_model/`
  - Model directory for evaluation and deployment
- Those you generated yourself:
  - `train_metrics.csv`
  - `train_metrics_summary.json`
  - `train_metrics.png`

From a teaching perspective, it is recommended to understand these files in two categories:

### Category 1: Training Outputs

- checkpoint
- last
- pretrained_model

These are "the model itself".

### Category 2: Training Evidence

- train.log
- Metrics CSV/JSON
- Curve charts
- Intermediate eval results

These are the evidence for "why we believe this model is trained properly".

Both types should be retained in the teaching, and only the model files cannot be kept.

If you want to add more later, I suggest prioritizing the following:

- Automatic eval for each checkpoint
- Success rate curve for each category
- Best checkpoint is automatically selected

## 3.13 Smoke results that have been actually run

To ensure that the tutorial doesn't just focus on "the command should work", we keep the minimal training evidence that has been actually run on the local machine.

### 3.13.1 ACT Teaching Version smoke

- Training logs:
  - `$OUTPUT_ROOT/train/act_top_short_probe64.log`
- Checkpoint output:
  - `$OUTPUT_ROOT/train/act_top_short_probe64/checkpoints/000008/pretrained_model`
- Plot:
  - `$OUTPUT_ROOT/plots/act_top_short_probe64/train_metrics.png`
- Metric summary:
  - `$OUTPUT_ROOT/plots/act_top_short_probe64/train_metrics_summary.json`

Key information of this smoke:

- Data: `top_short_merged`
- Valid batch size: `64`
- Number of parameters: approximately `52M`
- Has reached step `8`
- Loss has decreased from `57.147` to `22.134`
- Average `updt_s` is approximately `1.77s`
- Average `data_s` is approximately `0.60s`

This indicates that the default ACT teaching configuration for this tutorial can start operating smoothly on the local L40 machine, and users do not need to manually modify the YAML file to launch it.

### 3.13.2 DP Teaching Version smoke

- Training logs:
  - `$OUTPUT_ROOT/train/dp_top_short_probe48.log`
- Checkpoint output:
  - `$OUTPUT_ROOT/train/dp_top_short_probe48/checkpoints/000008/pretrained_model`
- Curve graph:
  - `$OUTPUT_ROOT/plots/dp_top_short_probe48/train_metrics.png`
- Metric summary:
  - `$OUTPUT_ROOT/plots/dp_top_short_probe48/train_metrics_summary.json`

Key information of this smoke:

- Data: `top_short_merged`
- Valid batch size: `48`
- Number of parameters: approximately `271M`
- Have reached step `8`
- Loss reduced from `1.119` to `1.097`
- Average `updt_s` is approximately `0.94s`
- Average `data_s` is approximately `0.79s`

This indicates that the DP teaching version has been verified on this machine to start smoothly, and can correctly generate checkpoints and training curves.

### 3.13.3 How to understand these smoke results

These results are not official competition scores, nor complete convergence experiments. Their purpose is:

- Prove that the command, configuration, and output path are set up correctly.
- Prove that the `top_short` teaching version configuration in the current image can be started directly.
- Prove that the drawing script included in the tutorial can consume real logs directly.

During official competition training, it is still recommended to continue running at the full distance:

- ACT：`30000` steps
- DP：`90000` steps

If you want to fully utilize the 40G VRAM in L40, the safest approach is not to blindly increase the steps, but to continue testing first:

- ACT is increased further `batch_size`
- DP is increased further `batch_size`
- Or more periodic intermediate evals are enabled

However, the teaching version document retains the currently verified safe starting point by default, making the first run of readers more stable.

### 3.13.4 Smoke evaluation videos that have been exported

In addition to training smoke, this machine has actually exported the first smoke evaluation episode video of `top_short`, and the location is as follows:

- ACT:
  - `$OUTPUT_ROOT/eval/act_top_short_smoke/failure/episode0_observation_images_top_rgb.mp4`
  - `$OUTPUT_ROOT/eval/act_top_short_smoke/failure/episode0_observation_images_left_rgb.mp4`
  - `$OUTPUT_ROOT/eval/act_top_short_smoke/failure/episode0_observation_images_right_rgb.mp4`
- DP:
  - `$OUTPUT_ROOT/eval/dp_top_short_smoke/failure/episode0_observation_images_top_rgb.mp4`
  - `$OUTPUT_ROOT/eval/dp_top_short_smoke/failure/episode0_observation_images_left_rgb.mp4`
  - `$OUTPUT_ROOT/eval/dp_top_short_smoke/failure/episode0_observation_images_right_rgb.mp4`

In the current smoke evaluation, DP has confirmed that it has completed at least the first episode and written the log:

- `Return=102.25`
- `Length=600`
- `Success=False`

The content retained here is the "teaching example output," not the complete competition statistics. Since the official `category eval` will continue to iterate through the garment list under this category, it is usually sufficient to retain the real examples from the first episode when creating tutorials.

## 3.14 Competition Enhanced Version: L40 Single-Card Long Training Configuration

If the goal shifts from "firstly getting it to work" to "creating a baseline that resembles the submission for a competition", it is recommended not to focus solely on that. `top_short`. A more reasonable approach is to directly switch to:

- Data: `four_types_merged`
- Model: ACT
- Device: Single card `L40`

Let’s clarify the boundaries here first: this doesn’t mean “guaranteeing top three”. The competition ranking ultimately depends on data, policy structure, evaluation details, and training time. There is no way to make false promises in a tutorial. However, it is indeed closer to a starting point that can be used to continue competing than the teaching smoke.

### 3.14.1 Why ACT is pushed first, rather than DP

Reasons are practical:

- ACT is easier to get started with stable long-term training on this dataset.
- With a single GPU L40, ACT can more easily control the batch size and training pace.
- If we want to fully utilize the graphics card within an hour and start a long-term experiment, ACT has lower engineering risks.

So, this tutorial designates "Competition Enhanced Version First Shot" as:

- `four_types_merged + ACT + batch_size 64 + 50000 steps`

Corresponding configuration file:

- `$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_act_competition_l40.yaml`

### 3.14.2 Why don't we keep increasing the batch size indefinitely

I performed three levels of exploration on four types of joint training on this machine:

- `batch_size=96`：OOM
- `batch_size=88`：OOM
- `batch_size=80`：OOM

Finally, it returns to `batch_size=64`, and this is for the following reasons:

- Can start long training sessions smoothly.
- Does not crash directly in the first few steps.
- Also maintains a high GPU utilization.

In other words, what is pursued here is the strongest usable point to "run steadily through the entire long training," rather than just aiming to fill up the VRAM in the first step.

### 3.14.3 Current Running Long Training

The local machine has been started:

- Training logs:
  - `$OUTPUT_ROOT/train/act_four_types_l40.log`
- GPU sampling:
  - `$OUTPUT_ROOT/monitor/act_four_types_l40_gpu.csv`

Training command is equivalent to:

```bash
cd "$LEHOME_ROOT"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --config_path "$TUTORIAL_ROOT/15-Challenge竞赛/LeHome/resources/configs/train_act_competition_l40.yaml" \
  2>&1 | tee "$OUTPUT_ROOT/train/act_four_types_l40.log"
```

### 3.14.4 Training signals currently visible

When the long training just started, the following changes occurred in this machine:

- step `50`：loss `12.334`
- step `100`：loss `3.554`
- step `150`：loss `2.940`

This indicates that at least in the early stage of training, the loss decrease is significant, rather than being stuck or experiencing pure noise fluctuations.

GPU sampling has also shown that this long training session indeed fully utilized the L40:

- VRAM usage: approximately `45.1 GiB / 46.1 GiB`
- GPU utilization: consistently close to `100%`
- Power consumption: approximately `280W` to `300W`

From an engineering perspective, this indicates that the current configuration has achieved the goal of being "stable and capable of handling full load."

### 3.14.5 Why is the long training set set at 50000 steps

The text is intentionally not written as an infinitely long string for the following reason:

- Competition and training do not follow the rule that “the longer it lasts, the better it is”.
- In many visual policies, the decline is most rapid in the early to middle stages, and later on, the marginal benefits start to decrease.
- For tutorials and server resources, `50000` falls within the range of “long enough to show a clear trend, but still moderate”.

So this configuration is more suitable as:

- First round official baseline
- Main experiment before the first ranking
- Decide later whether to extend to `70000` or `90000` as the reference starting point

If during the middle to late stage of long training, you find:

- Loss has become significantly platform-dependent.
- The eval success rate of the intermediate checkpoint no longer increases.

Then it may not be worth continuing to burn cards indefinitely.

---

## 3.15 Reference and Review of Public High-Score Solutions

This section compiles the LeHome Challenge solutions that were publicly accessible on 2026-05-13. **No complete open-source solution was found for the top three positions on the official ranking.** The most useful public references are reviews from teams near the top, the VLA improvement repository, and several competition repositories that can serve as comparison baselines.

These contents are not recommended to be treated as a "copy and you'll make it into the top three" formula. A more reasonable approach is: first run through the `ACT/DP + state + RGB` baseline in the previous chapter, and then enhance each item according to the methods in this section.

### 3.15.1 Current Position of Official Rankings

The official website Leaderboard is dynamically loaded via an interface. The corresponding interface in the page source is:

- `https://lightwheel.ai/lwapi/open/lehome/ranking`
- Request method: `POST`
- Request body: `{}`

As of 2026-05-13, the simulation top 8 before the competition returned by this interface are as follows:

| Rank | Registration ID | Team | Long-Sleeved Top | Short-Sleeved Top | Long Pant | Short Pant | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `r84` | `ilya` | 74.5% | 70.0% | 80.5% | 93.5% | 79.63% |
| 2 | `r196` | `Shubham @ Vorwerk` | 73.0% | 62.5% | 71.5% | 87.0% | 73.50% |
| 3 | `r55` | `Dum-E` | 76.5% | 62.0% | 75.5% | 79.5% | 73.38% |
| 4 | `r161` | `SCUT-Unlimited` | 65.5% | 66.0% | 70.0% | 91.0% | 73.13% |
| 5 | `r201` | `GraspYesAI` | 73.5% | 61.0% | 69.0% | 79.0% | 70.63% |
| 6 | `r162` | `sZs` | 70.5% | 64.0% | 68.5% | 75.5% | 69.63% |
| 7 | `r218` | `ClothFolder50k` | 77.0% | 56.0% | 58.5% | 82.5% | 68.50% |
| 8 | `r13` | `sisigakgak` | 68.0% | 52.5% | 64.0% | 77.5% | 65.50% |

There are two points to note:

- The top 8 do not equate to all open sources. The rankings only provide the team name and success rate, without methods, code, or checkpoint information.
- Public repositories and ranking teams may not correspond one-to-one. Unless the repository README or team description explicitly states the correspondence, this tutorial treats them as "public reference solutions" only.

### 3.15.2 Approach 1: UCAS review, classifier + expert ACT + PI0.5

Reference link:

- Participating repository: [wangerforcs/lehome-challenge-ucas](https://github.com/wangerforcs/lehome-challenge-ucas)
- Chinese review: [wangerforcs/EILearn/competition/lehome.md](https://github.com/wangerforcs/EILearn/blob/master/competition/lehome.md)

This review is one of the most valuable public materials for detailed reading at present. It is not just a README with commands, but a relatively complete record of the attempt paths from the baseline to the later stages of the competition.

The first phase is `classifier + ACT`. The core idea is: during evaluation, the official source does not directly indicate which category the current garment belongs to. Therefore, if you want to train expert policies for the four categories of clothing separately, you need to determine the category first. The approach is roughly as follows:

1. Use a visual classifier such as ResNet to identify the current clothing category.
2. Train ACT for `top_long`, `top_short`, `pant_long`, and `pant_short` respectively.
3. During inference, first classify the items and then route them to the corresponding ACT expert model.

The advantage of this approach is its simplicity and clarity, and it is easy to implement within the official LeRobot framework. However, the disadvantages are also obvious: incorrect classification will directly send samples to the wrong expert; if the data quality of a certain category is poor, even individual experts may be misled. The review mentions that around 42% of submissions were made at this stage, indicating that it works, but there is still a gap from exceeding 60%.

The second stage mainly switches to `PI0.5`. They retain the input-output format similar to ACT:

- Input: `observation.state`
- Input: `observation.images.top_rgb`
- Input: `observation.images.left_rgb`
- Input: `observation.images.right_rgb`
- Output: 12-dimensional `action`

Here, depth is not given top priority. This is consistent with the advice earlier in this tutorial: if the pre-trained model is not trained around depth, adding depth directly will increase engineering complexity, but it may not bring any stability benefits.

In the third phase, more detailed data and category processing will be carried out. The review specifically mentions that `top_short` and `pant_short` are the main bottlenecks: some data replayed actions are unstable, and the saving logic for successful/failed samples can also introduce dirty data. This experience is very important: LeHome is not a task that can be solved by simply changing the model; data quality is often more crucial than the model name.

Insights from everyone's replication:

- If you only want to quickly surpass the ordinary baseline, you can first use “four types of expert models + category recognizer”.
- If a certain category significantly lags behind, do not focus solely on the total score; instead, examine the success rate of each of the four categories separately.
- For challenging tasks such as `top_short` and `pant_short`, prioritize checking data replay and failed samples, rather than blindly increasing the model complexity.
- If training resources are limited, first train one category to achieve reproduction-level success rate, and then expand to all four categories.

### 3.15.3 Option 2: SPGVLA, adding progress guidance and world model supervision to VLA

Reference link:

- Code repository: [blackcat0615/spgvla](https://github.com/blackcat0615/spgvla)
- Model link 1: [spgvla](https://huggingface.co/blackcat0615/spgvla)
- Model link 2: [spgvla0.7](https://huggingface.co/blackcat0615/spgvla0.7)

The full name of `SPGVLA` is `Simple Progress Guidance For Vision Language Action Model`. Its origin is that clothing folding is a long-time sequence task, and the model easily fails to know which step it is in “grasping, flattening, folding, and finishing”. As a result, state confusion occurs.

This repository has been enhanced in two ways:

1. `SPG`: Simple Progress Guidance. Provides additional task progress information to the model, helping VLA determine which stage of the task it is currently in.
2. `WM`: world model module. Uses the world model to deliver more intensive supervision signals, addressing the issue where VLA has only sparse behavior cloning signals during training.

The public experimental results provided in the repository README are as follows:

| Experimental Setup | top long | top short | pants long | pants short | mean SR |
| --- | --- | --- | --- | --- | --- |
| baseline SmolVLA | 61.67% | 10.00% | 31.67% | 76.67% | 45.00% |
| baseline + SPG | 55.00% | 21.67% | 45.00% | 80.00% | 50.40% |
| baseline + SPG + bs64 | 63.33% | 25.00% | 33.33% | 88.33% | 52.50% |
| baseline + SPG + bs64 + WM | 70.00% | 25.00% | 45.00% | 86.67% | 56.67% |
| baseline + SPG + bs96 + WM + data aug retrain | 73.30% | 45.00% | 58.33% | 85.00% | 65.40% |

The most useful aspect of this result for everyone is not simply copying the module name, but rather it shows a clear trend: simply replacing the VLA baseline is not enough. Only when progress signals, world model auxiliary supervision, batch size, and data augmentation are combined, can the performance rise from 45% to 65.4%.

It is recommended to understand this when recreating:

- `SPG` is suitable for addressing confusion in long temporal phases, such as folding before the sleeves are even flat.
- `WM` is ideal for providing additional supervision for intermediate states, allowing the model to learn more than just the final action labels.
- `data_aug_retrain` is crucial for LeHome, as the official data for each category is not large, and even a slight change in visual distribution can affect success rates.
- `top_short` is a significant weakness; even after enhancement, its effectiveness is only 45%, indicating that it remains challenging in public solutions.

If you have successfully implemented the `four_types_merged + ACT` in this tutorial, the next step is to follow the concept of SPGVLA. You don't need to implement the full VLA immediately, but instead, perform two lightweight experiments first:

1. Add category or stage conditions to ACT/SmolVLA, such as `garment_type`, `stage_id`, and progress ratio.
2. Perform conservative image enhancement for each type of data to observe whether `top_short` improves.

### 3.15.4 Option Three: LaundryNauts, VLA Tweaked Participating Warehouse

Reference link:

- [cwoodhayes/lehome-laundrynauts](https://github.com/cwoodhayes/lehome-laundrynauts)

This repository is described as `fine-tuned VLA for bimanual garment folding`. The directory structure is basically extended from the official LeHome repository, and it includes `configs`, `docker_policy`, `scripts`, and `source/lehome`, etc. It corresponds to `LaundryNauts` in the official website list, but since the average score on the list is 40.00%, it is more suitable as a reference for “VLA organizational methods” rather than a high-score policy template.

Everyone can focus on three types of content:

- How it organizes custom policies within the official repository structure.
- How it prepares Docker/submission-related files.
- How it connects VLA training and official evaluation scripts.

The value of such repositories lies in engineering reference: when people move from their own methods to `lerobot-train` to custom models, Docker submissions, and remote testing, they can check whether the directories and scripts are complete by referring to it.

### 3.15.5 Option 4: S.N.N Neural Lab, a public but non-privileged control repository

Reference link:

- [alifestone/lehome-challenge_S.N.N](https://github.com/alifestone/lehome-challenge_S.N.N)

This repository is consistent with the `S.N.N Neural Lab` entry on the official list, whose average score is 40.38%. It is useful as a public comparison project built on the official environment, even though its final score remains near 40%.

For tutorials, such repositories remind everyone that the main challenge with LeHome is not "whether it can start training," but rather:

- Whether the data quality is clean enough.
- Whether the four types of clothing are optimized separately.
- Whether it can handle random garment categories during reasoning.
- Whether the model understands long-time progress.
- Whether the evaluation script, checkpoint path, and Docker submission are consistent exactly.

### 3.15.6 Practical Routes Extracted from Public Solutions

If you have completed the smoke test in the previous chapter, this tutorial recommends proceeding in the following order:

1. **First, establish a stable baseline**
    Use `dataset_challenge_merged`, keep `state + top/left/right RGB` as the input, and keep the output as a 12-dimensional joint action. First, ensure `ACT` works properly, then consider `DP` or `SmolVLA`.

2. **Consider the scores of four categories separately, not just the average score**
   In the official rankings, many teams have a high `short pant` and a low `top_short`. The average score hides the weaknesses; everyone should record the success rates of each category separately.

3. **Try expert policies and category classifiers**
    If the differences among the four categories are significant, you can train separate expert policies for each type of clothing, and then use a classifier to route them. This method is simple to implement, but it requires avoiding classification errors.

4. **Data in difficult categories for cleaning and strengthening**
   Data issues are repeatedly mentioned in the public reviews. It is advisable to replay some data before training, especially `top_short` and `pant_short`, and mark out episodes with obvious failures or significant motion deviations separately.

5. **Apply VLA or enhance progress**
    If the baseline is stable, refer to `SPGVLA` to add progress guidance, world model auxiliary supervision, or data augmentation. Do not deploy complex models directly before the environment is fully operational.

6. **Stabilize evaluation and submission processes before submission**
    High-score solutions must also work in official evaluations. Custom policies, Docker, checkpoint paths, and dependency versions need to be verified separately; one cannot rely solely on local training loss.

### 3.15.7 Summary of Reference Links

- LeHome official website: [https://lehome-challenge.com/](https://lehome-challenge.com/)
- Official repository:[lehome-official/lehome-challenge](https://github.com/lehome-official/lehome-challenge)
- Official asset data:[lehome/asset_challenge](https://huggingface.co/datasets/lehome/asset_challenge)
- Official merged training data:[lehome/dataset_challenge_merged](https://huggingface.co/datasets/lehome/dataset_challenge_merged)
- UCAS participation repository:[wangerforcs/lehome-challenge-ucas](https://github.com/wangerforcs/lehome-challenge-ucas)
- UCAS Chinese review:[wangerforcs/EILearn/competition/lehome.md](https://github.com/wangerforcs/EILearn/blob/master/competition/lehome.md)
- SPGVLA repository:[blackcat0615/spgvla](https://github.com/blackcat0615/spgvla)
- SPGVLA model:[blackcat0615/spgvla](https://huggingface.co/blackcat0615/spgvla)
- SPGVLA 0.7 model:[blackcat0615/spgvla0.7](https://huggingface.co/blackcat0615/spgvla0.7)
- LaundryNauts repository:[cwoodhayes/lehome-laundrynauts](https://github.com/cwoodhayes/lehome-laundrynauts)
- S.N.N Neural Lab repository:[alifestone/lehome-challenge_S.N.N](https://github.com/alifestone/lehome-challenge_S.N.N)

---

## 4. Recommended Training Policy

### 4.1 Baseline Configuration

It is recommended to start with the following set first:

- policy: `ACT`
- Input: `observation.state + top/left/right RGB`
- Output: `action`
- Device: `cuda` during training
- `cpu` for the env during evaluation

### 4.2 Why it is not recommended to use depth from the start

- Increasing depth increases data volume and training complexity.
- It is more important to ensure the baseline works properly first.
- The official documentation clearly states that `state + RGB` has been verified to be functional.
- If we want to add depth, it is recommended to conduct a comparative experiment after the baseline converges.

### 4.3 Is it recommended to use EE pose?

Not recommended as the preferred option.

The official documentation also clearly reminds:

- `observation.ee_pose`
- `action.ee_pose`

Due to IK and hardware limitations, the stability is not as good as joint-space control. It is recommended to use the competition baseline instead:

- `observation.state`
- `action`

---

## 5. Cleanup and Repository Control

If you want to integrate this tutorial into your main repository, it is recommended to keep only these lightweight contents:

- Tutorial documents
- Training configuration examples
- Evaluation commands
- A few log snippets
- A few screenshots or short video links
- Necessary patch / overlay descriptions

It is not recommended to submit these large resources:

- `Assets/`
- `Datasets/`
- `outputs/`
- `logs/`
- `models/`
- `videos/`
- `plots/`
- `.cache/`
- Any weights and materials larger than several dozen MB

The reason is simple: these contents should be obtained via download commands, and should not be added to the tutorial sub-repo.

### 5.1 Previous Failed Evaluation Results

If it is just a tutorial cleanup, the videos showing the failure of the previous batch of `0%` success rates can be deleted.

The directory to be deleted currently is:

```text
$OUTPUT_ROOT/legacy-eval/
```

Will be uniformly changed later:

```text
$OUTPUT_ROOT/eval/
```

---

## 6. Recommended repository organization method

It is recommended to add the following structure in `every-embodied`:

```text
15-Challenge竞赛/
└── LeHome/
    ├── README.md
    ├── .gitignore
    └── resources/
        ├── commands.md
        ├── configs/
        └── scripts/
```

If you add other competitions later, you can also use the same structure.

---

## 7. Check commands that can be reused directly

## 7.1 Check GPU

```bash
nvidia-smi
```

## 7.2 Check Core Package

```bash
python - <<'PY'
import isaacsim, lerobot
print("isaacsim ok")
print("lerobot", getattr(lerobot, "__version__", "unknown"))
PY
```

## 7.3 Check if the official assets have been downloaded

```bash
ls Assets
ls Datasets/example
```

## 7.4 Check List for Single Clothing Evaluation

```bash
cat Assets/objects/Challenge_Garment/Release/Release_test_list.txt
```

---

## 8. Image Release Recommendations

If you are promoting this image, it is recommended to emphasize in the description:

- The official LeHome environment is pre-installed.
- Training and evaluation can be run successfully after verification.
- It is recommended to download `asset_challenge` + `dataset_challenge_merged` for the first use.
- It is adapted for `L40` / server scenarios.
- A startup command for ACT/DP baseline is provided.
- By default, large competition resources are not stored in the tutorial repository.

It can be described externally as:

> A ready-to-use image for LeHome Challenge, pre-installed with the Isaac Sim / LeRobot / LeHome runtime environment, providing a minimal closed loop for training, evaluation, data download, and problem reproduction.

---

## 9. Local Verification Results

Local verification completed:

- ACT single garment evaluation can be run successfully, with results `Success Rate = 0.00%`
- DP single garment evaluation can be run successfully, and `policy_num_inference_steps=1` takes effect
- The evaluation video can be exported successfully

This indicates:

- The image is available.
- The training/ evaluation pipeline is functional.
- Tutorials can be written and published directly based on this image.


## Notes

Install the image and video viewer

```bash
sudo apt install celluloid

sudo apt install viewnior

# 将常见图片格式的默认打开程序设为 Viewnior
xdg-mime default viewnior.desktop image/jpeg image/png image/gif image/bmp image/webp

```
