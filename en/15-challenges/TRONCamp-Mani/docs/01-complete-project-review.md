# Full Project Review: From 550 Trajectories to Online T4

## 1. What should be answered in this review

In this competition experience, there are three areas where misjudgment is most likely to occur:

1. A normal training loss does not mean that the closed-loop manipulation can achieve three-layer stacking;
2. A successful local fresh3 run does not imply passing the online private seed test;
3. "Upload success" and "entering the queue" are not the same thing, let alone "official score approval".

Therefore, this article records it according to the evidence chain: data identity -> training lineage -> local official testing -> submission package -> online results. Each layer is evaluated separately, and the result of the next layer is not used to infer the correctness of the previous layer.

## 2. Dataset Identity

This T4 main line is fixed as a dataset in canonical official layout:

| Item | Value |
|---|---|
| Number of Trajectories | 550 |
| Shard Attribution | C2=186, C3=182, C4=182 |
| Original Total Bytes | 67,546,203,544 |
| Source Structure | stock=290, alignment retry=3, rescue=257 |
| Canonical Manifest SHA256 | `eeca954a7e64dda818a0cca86534e34f6ffa1a8055bd37810f9781a264199901` |
| Processed Dimension | 16-D |
| Cameras | `cam_high`, `cam_left_wrist`, `cam_right_wrist` |
| Next-State Verification | `max_error=0` |

This is not a “pure stock 550” dataset. The high rescue rate indicates that the goal of data engineering is to cover all official layouts, rather than retaining only the most successful trajectories. It helps achieve a closed loop for the data, but also introduces risks related to mixed distribution, action patterns, and expert profiles. Here, it is treated as a hypothesis that requires further verification, and the high rescue ratio is not directly declared as the sole reason for the online low score.

A clean50 diagnostic set has also been prepared:

- raw manifest SHA256: `7918bc860c595add91c37db8b01a27ea08849b5426955e2c1ba5cd3f7ada051f`;
- processed audit SHA256: `3f5fef447a4dfc47cd6209867560d94091238ec3770fedf4bcbee82caafa7b60`;
- 50 items, 16-D, three cameras, next-state max error is 0.

clean50 is used only to determine whether "the cleanest subset can close the closed loop faster", and cannot be used directly as evidence for exact550 or online rankings.

## 3. Main line ACT configuration

The main version is the official compatible ACT trained from scratch, with approximately 83.9M parameters:

```text
backbone               ResNet18
encoder layers         4
decoder layers         7
hidden dim             512
feed-forward dim       3200
attention heads        8
chunk size             50
KL weight              10
batch size             8
learning rate          1e-5
backbone learning rate 1e-5
state/action dim       16
seed                   120
action shift           0
gripper loss weight     1
augmentation           off
save frequency         every 250 epochs
target horizon         6000 epochs
```

To improve throughput, larger batches, TF32, and multiple workers were used in compatible environments as training engineering optimizations. However, these optimizations should not alter the model structure, data semantics, or evaluation wrappers. All throughput changes must be recorded separately; the phrase "faster" should not be misinterpreted as "better".

## 4. Training and Evaluation Timeline

### 4.1 Phase 1: Train exact550 to global e1250

The first training phase continued from global e250 to e1250, successfully completing 1,000 new epochs as expected. The logs indicate that the stage limit was manually set, not due to OOM, crashes, or checkpoint corruption. Later, due to hardware resources and deadline pressure, the main branch did not complete the official 6,000 epochs naturally.

This experience revealed a planning issue: if the goal is 6000 epochs, a complete computing window must be reserved before the start of the competition, and verification, evaluation, and rollback every 250 epochs should be automated. Only at the last minute is e1250 continued to e2000, which cannot replace the maturity of a complete training process.

### 4.2 Formal fresh3 for global e1250

Under the official runner, fixed public seeds, and clean configurations:

- `sr=0/3`;
- mean graded=`1/9`;
- Reversing layer by layer results in 0 layer=2, 1 layer=1, 2 layers=0, 3 layers=0;
- All three rollouts reached 1200/1200 steps without completing three full bowls successfully.

The results indicate that the model can be loaded, the environment can run, and the trajectory can end, but the task objective has not been completed. We cannot mark it as ready to submit just because “no errors occurred”.

### 4.3 Formal fresh3 for global e1500

After continuing training from e1250 best to global e1500:

- `sr=0/3`;
- mean graded=`2/9`;
- There is a cumulative improvement compared to e1250, but achieving all three bowls remains `0/3`;
- The aggregate of a regular runner cannot uniquely determine the number of layers of seeds, so the layered distribution cannot be fabricated.

Then, the trace version with the same weight is used to supplement the failure stage. Trace is a diagnostic tool that does not override the official results of a normal runner. It shows that three seeds have a layer count of 0, 1, and 1, and there is still no success in three layers.

### 4.4 Formal fresh3 for global e1750

The official result for global e1750 is still:

- `sr=0/3`;
- mean graded=`2/9`;
- All three have been rolled out, but none achieved three full bowls successfully.

This means that the increase from e1500 to e1750 did not result in an observable improvement on the small sample formal access control. A “diagnostic data/alignment/closed loop topology” check should be triggered here, rather than simply blindly adding epochs.

### 4.5 Clean50 e6500 Comparison

Two results occurred after training on the clean50 branch to e6500:

- Regular official runner: fresh3 is `0/3`, graded=`1/3`;
- Trace/diagnostic runner: fresh3 is `1/3`, graded=`5/9`. The layer distribution is 0 layers = 1, 1 layer = 0, 2 layers = 1, 3 layers = 1.

These two runs use the same weight and public seed, but the runner topology is different, and the trace version includes diagnostic packaging for the formal evaluation function. Therefore, they can only be recorded as "evidence of inter-run differences and reachability", and cannot be selected for high scores or claimed as official results with 5/9.

### 4.6 Submission and Online Results

Before the deadline, the final choice was the control candidate with the strongest evidence among historical ordinary official runners, rather than the clean50 trace or the exact550 e1750/e2000 main line:

- Local fresh3: `sr=1/3`;
- Local graded: `2/3`;
- Correct account suffix: `a40Cbw`;
- Official receipt: `#719 T4`;
- `submit_exit=0`.

The second version of exact550 e2000 was uploaded before the deadline, but a `403` from the server was received after the deadline, and the backend description had already been submitted; no second queue ID was formed.

Online Final Ranking:

```text
T1: 0.60  (达标)
T2: 0.67  (达标)
T3: 0.54  (达标)
T4: 0.2533333333333333 = 25.3/100
```

The online result is significantly lower than that of the local control candidate `2/3`. This indicates that the reachability of 3 public seeds is not equivalent to the generalization stability of 100 private seeds. It also shows that the submission boundaries, environment, and private scenarios must be treated as a first-level risk before the competition.

## 5. Hardware and Resource Migration

This time, I used a single 4090 card, RTX PRO 6000 local resources, and cloud PRO 6000 resources. The core issue is not "which card is absolutely faster," but rather the computational window, disk, environment, GPU ownership, and synchronization links are all limited simultaneously:

- The 4090 remote instance shut down due to insufficient balance, forcing the training to be interrupted;
- Temporary efforts to top up resources, waiting for startup, and environment synchronization consume precious time before the deadline;
- The cloud PRO 6000 has limitations on disk space and other user tasks, so a full environment cannot be restarted unconditionally;
- Migrating from C3 to Python 3.10 / PyTorch / SAPIEN / MPlib / cuRobo runtime is more reliable than reinstalling according to requirements, but image synchronization is time-consuming and requires breakpoint resume and SHA verification;
- During testing, a single process and exclusive GPU access are required. Do not start a second SAPIEN task just because there is still some VRAM available.

Final experience: The training plan must include the machine rental period and the risk of power outage or unpaid fees in the versioned runbook; the large environment, raw data, and checkpoints should be stored on persistent storage or recoverable object storage in advance, rather than waiting until the last minute.

## 6. Evidence Index and Publishing Boundaries

The evidence types in the private workspace include: training manifest and processed audit, checkpoint/stats SHA, fresh3/fresh10/fresh20 JSON, submit CLI receipt, leaderboard/queue snapshot, and checkpoint lineage.

When publishing to a public repository, the logs should first remove personal paths, IPs, PID/PGID, complete tokens, SSH private keys, internal server URLs, and asset information without redistribution permissions. Do not directly copy the entire private `logs/` directory into Git.
