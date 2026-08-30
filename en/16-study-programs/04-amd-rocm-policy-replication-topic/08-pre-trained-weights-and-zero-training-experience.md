# Pre-trained Weights and Zero Training Experience

This page is used to distinguish three types of materials: successful replays that can be viewed directly, pre-trained weights that can be loaded and run, and historical checkpoints that only contain experimental records and cannot be downloaded temporarily. Learners do not need to train for several hours first, confirm correct behavior and evaluation criteria, before deciding whether to use cloud resources to train their own models.

The official weights are officially released on the Datawhale Hugging Face organization. The course only provides reviewed model files, configurations, SHA256 hashes, evaluation JSON files, and loading instructions; it does not upload the optimizer state, complete training cache, or personal machine paths. Learners can download the weights first to watch the success replay, and then choose to train from scratch.

The current model repository has been created and the weights have been uploaded:

- [SmolVLA：every-embodied-smolvla-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-smolvla-mujoco-pnp)
- [Pi0：every-embodied-pi0-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-pi0-mujoco-pnp)
- [ACT：every-embodied-act-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-act-mujoco-pnp)

All three repositories contain model cards, configuration files, evaluation summaries, and `weights/model.safetensors`. The local `huggingface/` directory retains the same release list and model cards for easy maintenance later; it does not upload optimizer status, complete training cache, or personal machine paths.

## Re-download and Reproduction

After cleaning the local history checkpoint, learners can directly restore the course weights from the Datawhale Hugging Face repository. Only the files required for inference and evaluation are downloaded, and the optimizer state is not downloaded:

```bash
export MODEL_ROOT=${MODEL_ROOT:-$HOME/physical-ai/checkpoints}
mkdir -p "$MODEL_ROOT"

# SmolVLA
hf download Datawhale/every-embodied-smolvla-mujoco-pnp \
  weights/model.safetensors weights/config.json weights/train_config.json \
  --repo-type model --local-dir "$MODEL_ROOT/every-embodied-smolvla"

# Pi0
hf download Datawhale/every-embodied-pi0-mujoco-pnp \
  weights/model.safetensors weights/config.json weights/train_config.json \
  --repo-type model --local-dir "$MODEL_ROOT/every-embodied-pi0"

# ACT
hf download Datawhale/every-embodied-act-mujoco-pnp \
  weights/model.safetensors weights/config.json \
  --repo-type model --local-dir "$MODEL_ROOT/every-embodied-act"
```

After downloading, pass the corresponding `weights/` directory through `SMOLVLA_EVAL_POLICY_PATH` / `SMOLVLA_POLICY_PATH`, `PI0_EVAL_POLICY_PATH` / `PI0_POLICY_PATH`, or `ACT_EVAL_POLICY_PATH` / `ACT_POLICY_PATH` of the Notebook to perform zero-training loading and closed-loop evaluation. When continuing training from the protected recipe, use `PI0_PRETRAINED_PATH_OVERRIDE` or `POLICY_PRETRAINED_PATH_OVERRIDE`. The officially released weight SHA256 has been recorded in `huggingface/upload_manifest.json`.

## Zero Training Success Preview

Open [11_mujoco_closed_loop_deploy.ipynb](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/notebooks/11_mujoco_closed_loop_deploy.ipynb), and run the environment setup and the "Zero Training Success Preview" cell in sequence. The Notebook will display using `IPython.display.Video`:

```text
assets/pnp_four_view_strict_success.mp4
```

This video shows a precise playback with fixed environment seed 0 and policy sampling seed 3. The policy is `pi0 + visual/history learned head`, not raw pi0; the input consists only of dual-camera images, language, robot proprio, and historical execution actions, without reading cup coordinates, plate coordinates, GT phase, or oracle action data. It demonstrates the correct order of approaching, grasping, lifting, moving, releasing, and stabilizing placement, but it does not indicate that generalization to random positions has been achieved.

## Weight Release Threshold

The following information must be retained simultaneously before publishing the checkpoint:

| Field | Requirements |
| --- | --- |
| Model | Model type, base version, training steps, and key hyperparameters |
| Data | Dataset version, number of episodes, language tasks, and sampling methods |
| Action | State/action dimensions, absolute or incremental actions, control frequency, and chunk settings |
| Evaluation | Environment version, seed list, `physical_success` success rate, and representative videos |
| Completeness | Weight file SHA256, configuration files, and loading commands |
| Boundary | Clearly indicate whether it is a raw policy, a learned head, or a scaffold with rules |

The weights of these fields that are missing will not be published as a course baseline, even if they can be loaded.

## Current downloadable list

| Candidate | Verified Result | Current Release Status | Description |
| --- | --- | --- | --- |
| SmolVLA weighted500 | Red cup `27/30`, Blue cup `30/30`, Total `57/60` | Recommended for release | Current end-to-end Notebook reconstruction result; model SHA, evaluation JSON, and representative video are provided during release |
| Pi0 protected clean40 | strict `12/14`; no visible position `9/10`; hard group `6/8` | Recommended as advanced weight | Suitable for learning protected continuous training, action alignment, and failure analysis; not described as raw Pi0 zero-sample success |
| ACT stable61 fallback | strict `7/30` | Historical baseline weight | Old reconstruction branch on AMD395, retained for comparison |
| ACT protected repair15 candidate | strict `15/30` | Teaching/diagnostic candidate | Low-learning-rate continuous training of 2500 steps from stable61 protected checkpoint; 10 samples per group of fixed seeds, result is `3/10 + 4/10 + 8/10`, model SHA is `b9b178377995a674a06bc5d1500c8e7e7fc5d02649268855f892b3987bf5bfeb4` |
| ACT protected DAgger artifact | strict `2/30` | Negative control weight | Exact checkpoint recovered and SHA verification completed; retained to explain why data and recipe need item-by-item review |
| pi0 + visual/history head | Fixed environment `6/12`, Random environment `1/4` | Not considered as introductory weight for now | Improvement in fixed scenarios is evident, but not raw pi0, and insufficient generalization at random positions; this chapter only discloses successful playback and method boundaries |
| Pi0.5 canonical/recovery | Current strict still fails to stabilize successfully | Not released as successful baseline | Used only as action representation, chunk alignment, and recovery diagnosis cases |

Here, the principle is to "prefer a later release rather than releasing checkpoints that cannot be reviewed". The protection weights for SmolVLA and Pi0 can serve as download entries for learners; ACT repair15 has a complete strict30 JSON, grouping results, and SHA, and can be considered a candidate for current teaching protection. However, it is `15/30`, and it still lacks 2 items compared to the historical `17/30`, so it cannot be regarded as fully reproducible. The old DAgger `2/30` remains retained as a negative diagnosis branch.

## Policy for the quota during self-training

First, execute 1–2 steps of smoke testing to verify data reading, forward, reverse, optimizer, and saving processes. Then run a short training checkpoint, and start the 4 fixed seed closed-loop small panel as soon as possible. Only when the model shows real proximity, clamping, or lifting, should the training and evaluation scale be increased.

Do not use all the free time for a long training session that cannot be observed. Keep at least:

- A recoverable checkpoint;
- A training log;
- A set of closed-loop JSONL with fixed seeds;
- A successful or typical failed video.

Whether it is "fully trained" ultimately depends on the held-out strict success and video results, not on the number `5000 steps`.
