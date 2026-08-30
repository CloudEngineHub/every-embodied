# ROCm End-to-end Collection, Training, and MuJoCo Deployment

The first six tasks focus on device inspection, physical success evaluation, and model diagnosis. This chapter completes the missing basic execution chain: starting from keyboard teaching, generating the LeRobot dataset, training ACT, SmolVLA, and pi_0 on AMD ROCm, and then putting the checkpoint back into MuJoCo for closed loop evaluation.

The "successful execution" here includes five consecutive checkpoints:

1. MuJoCo viewer can receive keyboard manipulation and perform a real grasping operation;
2. LeRobot dataset contains correct images, states, actions, and language instructions;
3. 2-step smoke can complete data loading, forward, backward, and checkpoint writing;
4. Formal training can use AMD GPU stably without OOM, NaN, or kernel exit issues;
5. Checkpoint is used through `physical_success` in the MuJoCo closed-loop that actually changes the object’s initial position, and there is video available for verification.

A decrease in training loss, a reduction in open-loop action error, or no errors in the script do not alone prove the fifth item is valid.

## Learning Path

| Phase | Notebook | Main Outputs |
| --- | --- | --- |
| Data Collection | [07_data_collection_and_audit.ipynb](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/notebooks/07_data_collection_and_audit.ipynb) | LeRobot dataset with dual cameras, 6D state, 7D action, and red/blue cup commands |
| ACT Training | [08_act_training_rocm.ipynb](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/notebooks/08_act_training_rocm.ipynb) | ACT smoke and official checkpoint |
| SmolVLA Training | [09_smolvla_training_rocm.ipynb](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/notebooks/09_smolvla_training_rocm.ipynb) | SmolVLA smoke and official checkpoint |
| pi_0 Training | [10_pi0_training_rocm.ipynb](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/notebooks/10_pi0_training_rocm.ipynb) | gated permission check, pi_0 smoke and official checkpoint |
| Closed Loop Deployment | [11_mujoco_closed_loop_deploy.ipynb](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/notebooks/11_mujoco_closed_loop_deploy.ipynb) | fixed seed success rate, JSONL metrics, and rollout video |
| pi0 Strict-Input | [12_pi0_strict_input_end_to_end.ipynb](../../../16-专题组队学习/04-AMD-ROCm策略复刻专题/notebooks/12_pi0_strict_input_end_to_end.ipynb) | raw/head comparison, seed repair, fixed and random environments tabular data |

During the first learning phase, it is recommended to complete the entire chain using ACT. The ACT model is smaller and trains faster, making it easier to identify issues with data or action definitions. After passing the ACT closed loop, train SmolVLA on the same language data, and finally address the gated permissions of pi_0, model loading, and small data contact control issues.

## Two Starting Points

### Starting from scratch

Run 07, 08, and 11 in sequence to obtain a closed loop result for ACT first; then run 09 and 10, and switch back to 11 for model evaluation. This approach does not require a pre-trained checkpoint to be prepared in advance.

### Existing data or checkpoint

If the upstream `04mujoco复现ACT、Pi0、SmolVLA` has been completed, you can set the following environment variables to directly enter the corresponding training or deployment Notebook:

```bash
export PROJECT_ROOT=/path/to/every-embodied/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA
export DATA_ROOT=/path/to/large-disk/datasets/every_embodied
export MODEL_ROOT=/path/to/large-disk/checkpoints/every_embodied
export OUTPUT_ROOT=/path/to/large-disk/outputs/every_embodied
```

`PROJECT_ROOT` stores lightweight source code; data, checkpoints, videos, and caches should be stored on disk with sufficient capacity. Do not submit these large files to the tutorial repository.

## Data Collection Boundaries

Data collection is independent of the GPU vendor. As long as the MuJoCo and LeRobot versions, scene XML, control frequency, state/action schema, and success criterion are consistent, data can be collected on another desktop machine with an NVIDIA GPU or only a CPU, then synchronized to the AMD training machine.

07 Notebook made five corrections to the upstream interaction collection:

- After saving each trajectory, turn off the recording switch to prevent the reset process from being mixed into the next episode;
- The red and blue cup commands appear alternately to reduce the skew in the distribution of small data tasks;
- Only when the cup is truly lifted, remains upright, has a reasonable height between the cup and plate, and maintains stability for 5 consecutive steps will it be saved;
- Saving is rejected when the plate is pushed beyond 5 cm to avoid treating “pushing the cup onto the plate” as a successful teaching;
- Each trajectory is recorded synchronously in MP4 files from four perspectives: Agent, Egocentric, Top, and Side. The model only uses the first two cameras, and the other two perspectives are reviewed manually.

The `demo_data_language` on the AMD device has been measured to have 20 episodes, 2621 frames, and a frequency of 20 Hz. The state has 6 dimensions, and the action has 7 dimensions. The 07 Notebook will directly display this summary and a real four-view playback, so even before manual collection begins, you can get an overview of the data structure and what the final video should look like.

First, check the random seed of `mujoco_env/y_env2.py`. The old code used `np.random.seed(seed=0)`, and any seed input would yield the same set of object positions. Now it has been corrected to `np.random.seed(seed=seed)`; the old data can still be used for smoke in fixed scenes, but multi-position generalization must be re-collected after fixing.

If the remote JupyterHub or Code Server cannot focus the keyboard for the MuJoCo viewer, do not force data collection in an invisible session. You can use a remote desktop, or collect data locally and then sync it.

## Smoke and Formal Training

08–10 will generate two copies of YAML:

- The `smoke` configuration runs only 2 steps to verify the link;
- The `full` configuration uses the baseline training steps of this model and saves intermediate checkpoints.

`RUN_SMOKE`{} and `RUN_FULL_TRAIN`{} in the Notebook defaultly both become `False`. First, check the data path, output directory, and generation configuration, then explicitly open them. After the smoke test passes, confirm further:

- The GPU utilization and VRAM usage in `rocm-smi` are reasonable;
- The loss and gradient norm are finite values;
- Checkpoints are written to `$MODEL_ROOT`;
- There is sufficient space on the system disk;
- No Hugging Face token is written into the Notebook or logs.

pi_0 also requires Hugging Face gated model permissions. The Notebook only checks whether `HF_TOKEN` exists, and does not print or save tokens.

08–10 will also read the AMD historical training summary, displaying the number of completed training steps, closed loop results, and text progress bars. These are not replaced by static images for this long training: the actual code that triggers smoke/full remains in the previous unit, the training logs can be repeated in Notebook with tail, and the final checkpoint must enter 11 before making a closed loop judgment.

## MuJoCo closed-loop

11 Notebook will execute the following closed loop at 20 Hz:

```text
双相机图像 + 6D 关节状态 + 语言指令
                    ↓
              ACT / SmolVLA / pi_0
                    ↓
          7D 关节目标与夹爪命令
                    ↓
                 MuJoCo
                    ↓
              下一帧真实观测
```

The evaluation only preview the configuration by default and does not automatically load the large model. After confirming `POLICY_TYPE`, `MODEL_RUN_DIR`, data statistics, and session display, `RUN_CLOSED_LOOP=True` will be applied. The small panel can run 4 seeds initially, but the final conclusion requires at least 20–30 held-out seeds, with reports for the red and blue cups respectively.

The output directory contains:

- `results.jsonl`: `legacy_success`, `physical_success`, lift, xy, and upright for each seed;
- `<policy>_seed<seed>.mp4`: Rollout videos for reviewing actual behavior;
- Terminal summary: strictly the number of successes and the total number of episodes.

If `legacy_success` is higher than `physical_success`, it indicates a cup-push, empty grab, drop during transportation, or other geometric misjudgment. In this case, return to Task 02–06 and continue the diagnosis based on observation, action, contact stage, and video evidence.

## pi0 strict-input Review

12 Notebook records the most strict and least ambiguous set of comparisons currently: on the same fixed environment seed 0, raw pi0 is `0/12`, and after adding the visual/history GRU head, it becomes `6/12`. This combination head only reads the dual-camera images, language, 6D robot proprio information, and the 7D EEF/gripper commands that were executed by the previous policy; it does not read target/plate coordinates, GT phase, oracle action, or current/future GT gripper tags. However, since it is an additional learned head, it cannot be written as raw pi0.

After fixing the environment seed, there is only `1/4` in the actual random environment 41–44. Therefore, collecting data from multiple locations is more important than continuing with blind pi0 training steps. The fixed environment evaluation answers whether the policy sampling is stable under the same scenario, while the random environment evaluation answers whether location generalization holds. Both must be reported separately.

The four-view material also highlights a common input protocol issue that is easy to overlook: when using the full instruction `Pick up the blue mug and place it on the plate.` during training, the learned head is activated for 269 steps and succeeds on step 270; however, when a shorter equivalent instruction `Place the blue mug on the plate.` is used, the prototype gate does not take over at all, resulting in failure. The current small-data auxiliary head still cannot claim to have generalization in language rewriting. A formal evaluation should first fix the prompt, and the equivalent instruction enhancement and paraphrase gate can be added in the next round.

## Completion Criteria

After completing this chapter, the experiment record should include at least:

| Item | Minimum Requirements |
| --- | --- |
| Environment | AMD devices, ROCm, PyTorch, and memory/temperature logs |
| Data | Number of episodes, red-blue cup distribution, state/action shape, physical playback audit |
| Smoke | 2-step completion, checkpoint availability |
| Formal Training | Number of training steps, batch size, intermediate checkpoints, and resource curves |
| Deployment | Hold-out seeds, `physical_success`, one successful video and one failed video |
| Conclusion | Clear distinction between raw policy, auxiliary head, and hybrid results with scaffolding |

This result can continue to be used with ACT DAgger, SmolVLA Weighted sampler, and pi_0 tail diagnosis, without the need to rebuild the basic connection.
