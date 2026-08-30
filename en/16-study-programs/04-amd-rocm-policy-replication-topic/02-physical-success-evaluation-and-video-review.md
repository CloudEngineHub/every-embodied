# 02 Physical Success Evaluation and Video Review

This task addresses a core issue: whether the policy actually achieves grasping when the log indicates success. Here, the raw environment success rate and physical success rate are counted separately, and video reviews are used to verify typical successes and failures.

Supporting practical notebook: [02_physical_success_review.ipynb](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/notebooks/02_physical_success_review.ipynb).

## Why we cannot only look at the environment success

In the task of catching cups and placing them on plates, the original `check_success()` often aligns more closely with geometric conditions. The policy may not be able to securely hold the cup, but simply push it near the plate; or the cup may have already fallen, yet its position meets the termination criteria.

Therefore, this special report recommends two indicators:

| Metric | Meaning |
| --- | --- |
| `legacy_success` | Original success condition for the environment |
| `physical_success` | The target cup is lifted, placed on the plate, and in a basically upright position |

When they are inconsistent, follow the status of the video and object.

## Recommended physical caliber

Recommended minimum physical diameter:

1. legacy success is true;
2. the target cup is raised at least `0.03 m` relative to the initial height;
3. the lifting state lasts for at least several control ticks;
4. the final state of the cup does not fall significantly.

This set of metrics does not require the threshold to remain constant. More importantly, the same criteria must be used when comparing the same set of models.

## Batch Evaluation Script Format

It is recommended to create a Python script for batch evaluation instead of manually running it repeatedly in the Notebook. The script entry can be designed in the following way:

```bash
python tools/audit_language_policy_physical.py \
  --policy-type smolvla \
  --policy-path "$MODEL_ROOT/checkpoints/000500/pretrained_model" \
  --instruction "Place the red mug on the plate." \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --max-action-steps 600 \
  --output-jsonl outputs/eval_red.jsonl \
  --summary-json outputs/summary_red.json
```

The script output should include:

- seed;
- instruction;
- legacy success;
- physical success;
- first success step;
- maximum lift height;
- xy distance from the final cup to the plate;
- final upright indicator;
- failure reason bucket.

## Video Review

Each checkpoint must contain at least two types of videos:

1. A video of real success;
2. A video of typical failure.

Clearly indicate what type of evidence the video is:

```html
<video controls muted preload="metadata" width="100%">
  <source src="../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/assets/videos/seed0_blue_success.mp4" type="video/mp4">
</video>
```

Example of figure caption:

> Figure 1: The actual successful deployment of SmolVLA on the Blue Cup task. The review focuses on whether the cup is lifted, placed on a plate, and remains upright in the final state.

Don't just show success videos. Failure videos are more suitable for teaching, as they can explain why a stricter evaluation criteria are needed.

## Example Keyframe Sequence

The key frames below come from the same physical success judgment process. When reviewing the video, do not just focus on the cup position in the last frame, but observe along the time axis to see if stable grasping, lifting, moving, and releasing occur.

![SmolVLA baseline Blue Cup failure keyframe ](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/assets/smolvla_blue_failure_sequence.jpg)

Figure 1: Typical failure of SmolVLA baseline under the Blue Cup instruction. The policy repeatedly approaches the plate or the Red Cup, but the target Blue Cup cannot be steadily picked up, so it cannot be considered a physical success.

![SmolVLA Weighted Sampling - Key Frame for Blue Cup Success](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/assets/smolvla_blue_success_sequence.jpg)

Figure 2: Successful cases of the blue cup after blue frame weighted sampling by SmolVLA. The blue cup is first lifted and then moved onto the plate, and the final state also meets strict physical dimensions.

![ACT DAgger Failed Keyframe ](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/assets/act_failure_sequence.jpg)

Figure 3: Typical failures that may occur after ACT DAgger. Although the cup was touched, its posture quickly became unstable, and it never met the physical success criteria for being placed upright.

![ACT DAgger Success Keyframe ](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/assets/act_success_sequence.jpg)

Figure 4: A physical success case of ACT DAgger. This sequence demonstrates the entire process from approaching, gripping, handling, to releasing on the disk, which can be used as a reference for rollout review.

## Checkpoint

After completing this task, keep these evidence:

- One copy of the JSONL evaluation file for the red cup and one for the blue cup;
- One summary JSON or Markdown table;
- At least 1 successful video and 1 failed video;
- A brief classification of the reasons for failure.
