# 【AMD ROCm × Datawhale】Achieve a working embodied AI policy in 7 days! Together with leading learners X, truly understand ACT, SmolVLA, and pi_0

> This article is a draft for the public account recruitment copy. Before the official release, please replace the start date, participant introduction, registration link, QR code, number of recruits, and equipment resources description.

Hello, all my learning partners from Datawhale.

Have you run some embodied AI tutorials in a CUDA environment, but are still not very familiar with the AMD ROCm ecosystem? Are you curious whether devices like Ryzen AI MAX+ 395 can run LeRobot, ACT, SmolVLA, and pi_0? Have you encountered this confusion: the training loss decreases, the script runs successfully, but in the video, the cup isn't actually held up properly?

This time, we plan to address these issues in a team learning session, carefully analyze them, and ensure each step is completed successfully.

We plan to launch the **AMD ROCm × Datawhale “Embodied AI Policy Recreation Special” team learning program**. This special topic focuses on the AMD ROCm policy replication content in Every Embodied. On AMD ROCm devices, we will replicate ACT, SmolVLA, and pi_0 in the MuJoCo cup grasping task, and organize environment adaptation, training, evaluation, video review, and failure troubleshooting into a complete learning workflow.

This is not just an environment installation exercise, nor is it a simple reproduction where the task finishes after execution. It is more like a embodied AI engineering debugging and training session: observing the device, data, model, video, and success rate, as well as identifying where failures occur.

## Your Navigator: X

Trainee: X

This special study will be led by X throughout. As a "guide" and "companion," X will guide everyone through the entire process, from confirming the AMD ROCm environment, to replicating the ACT / SmolVLA / pi_0 policy, and finally to analyzing success rates and debugging.

In daily learning, X will release guidance notes explaining the experimental goals to be completed that day, highlighting areas where common mistakes occur, and helping everyone organize their results into reviewable experimental records. The most important thing here is not who runs the fastest, but who can clearly explain why a policy succeeded or failed.

## Why do we need to develop a dedicated embodied AI for AMD ROCm?

Many open-source tutorials on embodied AI default to the NVIDIA/CUDA environment, but real development environments are becoming more diverse. Hardware and software combinations such as AMD Ryzen AI MAX+, Radeon GPU, ROCm, unified memory, and remote learning cloud are entering the sight of learners and developers.

The real problem arises:

- Can PyTorch on ROCm devices accurately recognize the GPU?
- How should we understand the 128G unified memory and VRAM usage?
- When the model weights are large, where should loading and caching be done?
- ACT works well in open-loop, but why fails in closed-loop?
- Why are the success rates of red and blue cups for SmolVLA analyzed separately?
- Although pi_0 loss has decreased, why does the raw policy still get stuck at the final action?
- In the video, is it truly successful when "the cup approaches the plate"?

These issues are very suitable for team-based learning, as they cannot be solved by a single command. Instead, it is necessary to review logs, watch videos, analyze metrics, and examine trajectory data together, and then clearly explain the reasons.

## What will you systematically master in 7 days?

We will focus on the topic of replicating AMD ROCm policy for Every Embodied. The learning goal is not to fill up with parameters, but to develop a method to determine whether the embodied AI policy has been successfully replicated.

Your 7-day learning goals are very clear:

- **Understand AMD ROCm device resources**: Comprehend the relationships between GPU/APU, ROCm, PyTorch, unified memory, memory usage, temperature, and fan modes.
- **Complete the MuJoCo cup-reproduction chain**: Perform training and evaluation checks for LeRobot, ACT, SmolVLA, and pi_0 on AMD ROCm devices.
- **Learn physical success determination**: Distinguish `legacy_success` and `physical_success`, and use videos to determine whether the cup is actually lifted, moved, and released.
- **Identify the different challenges of ACT / SmolVLA / pi_0**: ACT is suitable for closed loop diagnosis; SmolVLA is suitable for analyzing small data sampling biases; pi_0 is suitable for analyzing bottlenecks in large model policies at the final stage.
- **Master debugging and review methods**: Organize a failure into “phenomenon, evidence, root cause, fix, and verification,” rather than leaving only a long list of commands.
- **Prepare a visible reproduction experiment report**: Include the device environment, data audit, success rate table, representative videos, failure cases, and next steps for improvement.

Core learning paths and resources:

- Open-source course entry: [Every Embodied](https://github.com/datawhalechina/every-embodied)
- AMD ROCm policy reproduction topic: [16-Team-based Learning/04-AMD-ROCm Policy Reproduction Topic](README.md)
- Original MuJoCo task tutorial: [04mujoco Reproduction of ACT, Pi0, SmolVLA](../../06-manipulation-and-vla/large-model-control-vla-vlm/04-mujoco-reproduction-act-pi0-smolvla/README.md)
- ROCm compatibility documentation: [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html)
- AMD Ryzen AI MAX+ 395 product page: [AMD Ryzen AI MAX+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html)

## 7-Day System Learning Schedule

The instructor X will teach in the order of "check the equipment first, then run the link, then check the results, and finally conduct a review". Each day has clear goals, as well as outcomes that can be accumulated.

### Day 0: Opening Ceremony and Equipment Awareness

Understand the learning objectives of this topic, how to use remote devices, basic concepts of AMD ROCm, and the relationship between unified memory and VRAM. On that day, we will check the device resources, record the GPU/APU model, ROCm, PyTorch, temperature, VRAM usage, and cache directory planning.

Recommended output: a AMD device resource checklist.

### Day 1: Smoke test of ROCm environment

The ROCm, PyTorch, LeRobot, MuJoCo, and minimum training/inferencing checks are passed. At this stage, the focus is not on model performance, but on ensuring that the GPU is visible, tensors can be sent to the card, dependencies can be imported, and the minimum training does not fail immediately.

Recommended outputs: environment logs, smoke test results, and the first troubleshooting record.

### Day 2: Physical Success Evaluation and Video Review

Enter the evaluation stage of the cup grabbing task. Focus on understanding why "the cup is near the plate" does not necessarily mean success, and how to determine through videos and `physical_success` whether the cup has been picked up, moved, and released.

Suggested outputs: examples of successful/failed videos, a comparison table of `legacy_success` and `physical_success`.

### Day 3: ACT Migration and Closed Loop Diagnosis

Train and evaluate ACT, and observe the differences between open-loop and closed-loop. Focus on understanding why gripper timing, action accumulation error, Dagger/GOracle correction affect the final success rate.

Recommended outputs: ACT success rate table, typical failure types, Dagger debugging ideas.

### Day 4: SmolVLA migration and sampling weighting

Run the SmolVLA red/clear cup fixed instruction evaluation to analyze sampling bias in small data tasks. A important issue is observed: even when "capturing cups," different colors and instruction distributions may lead to significantly different success rates.

Recommended outputs: Red Cup/Blue Cup success rate comparison table, sampling weighting experiment records.

### Day 5: pi_0 permissions, smoke, and training gating

Check the permissions for Hugging Face gated models, 1-step smoke, checkpoint resume, baseline and raw policy failure modes at 500/1500 steps. The focus is not on calling pi_0 “perfectly replicated,” but on learning how to identify which aspects of the large model policy have been validated, and which final actions are still unstable.

Recommended outputs: pi_0 smoke records, training gate list, raw policy failure analysis.

### Day 6: Debugging Review and Report Preparation

Organize the issues encountered in the environment, data, models, and evaluations over the past few days into a readable report. Ultimately, create a concise reproduction experiment report so that others can understand what you have verified, what remains unverified, and what should be done next.

Recommended output: a complete report on the reproduction experiment of the AMD ROCm policy.

## Why is this team learning session worth participating in?

**1. Real heterogeneous GPU environment, not just staying on the CUDA default path**

This topic focuses on AMD ROCm devices. The learning process will involve real engineering issues such as ROCm, PyTorch, TheRock/gfx1151, unified memory, cache directories, and remote training. These experiences are valuable for understanding the heterogeneous GPU ecosystem.

**2. Not only ensuring the command works, but also emphasizing the replication of evidence**

Completing training does not mean the policy is successful. We evaluate the model performance using logs, success rates tables, videos, key frames, and physical success judgments. This habit is very important because failures in embodied AI often lie in the videos, rather than in the loss curve.

**3. Review the three types of models: ACT, SmolVLA, and pi_0 together**

The issues with ACT, SmolVLA, and pi_0 are different: ACT resembles classic closed loop control and imitation learning diagnostics. SmolVLA is better suited for observing small data biases in visual language policies. pi_0, on the other hand, reveals the complexity of large-scale policy mechanisms regarding permissions, loading, training gating, and final actions.

**4. Write about failures in the tutorial, rather than only showing successes**

This workshop encourages everyone to document failure cases: where things broke, where misjudgments occurred, where training was done but not learned, and where videos appeared successful but actually did not capture the target. These details are very useful for beginners to develop a real engineering intuition.

**5. The final output can be compiled into public learning materials**

Each student can prepare a small experimental report that includes equipment, data, models, success rates, videos, and troubleshooting records. This is more valuable than just taking a screenshot of a training command, and it is also more suitable for writing blogs, sharing, or further research later on.

## What support will you receive?

- **Leading Student X**: Responsible for controlling the pace, providing key explanations, collecting questions, offering centralized answers, and sharing summaries.
- **Datawhale Community**: Provides team-based learning organization, maintains resource access, and offers support during the learning process.
- **Every Embodied Open Source Tutorials**: Offer complete topic documents, Notebooks, scripts, and representative experimental materials.
- **Remote AMD Device Practice Opportunities**: Focus on environment check, training, evaluation, and video review using AMD ROCm devices.
- **Peer Debugging Environment**: Everyone can discuss failure logs, video phenomena, and success rate tables to form a more solid judgment.

## Suitable for such learners

- Have already understood Python, GitHub, and virtual environments, and hope to gain further exposure to real training and evaluation processes;
- Interested in AMD ROCm, Ryzen AI MAX+, Radeon GPU, or heterogeneous GPU training environments;
- Want to learn about ACT, SmolVLA, pi_0, LeRobot, MuJoCo, and VLA small-data replication;
- Hope to learn how to determine whether a model has truly completed its task based on success rates, videos, logs, and troubleshooting records;
- Not satisfied with just "running the command", but also want to know why it succeeded or failed, and how to adjust next.

If you have never encountered robot simulation, you can still participate. However, it is recommended to read the original MuJoCo tutorial in advance to understand the basic meanings of observation, action, rollout, and checkpoint.

## What will you take with you in the end?

After completing this topic, you will have at least four types of outcomes:

- A record of the AMD ROCm device and environment inspection;
- A set of results from training, evaluation, and video review for ACT, SmolVLA, and pi_0;
- A set of metrics and video review methods to determine whether the "policy is truly effective";
- A report on the AMD ROCm policy reproduction experiments that can be further improved.

The report is recommended to include: device environment, data auditing, ACT success rate, SmolVLA red-blue cup comparison, failure analysis of pi_0 smoke and raw policy, representative videos, troubleshooting cases, and next steps.

In public reports, there is no need to include model weights, full caches, private accounts, remote addresses, tokens, long logs, or personal machine paths. Only keep the tutorial Markdown, small scripts, screenshots, short videos, summary tables, and reproduction experiments for evaluation.

## Get on the bus now and jointly implement and understand the embodied AI policy

We plan to recruit **[Number of Recipients to Be Determined]** learners to jointly start this AMD ROCm embodied AI policy replication study.

Please scan the QR code below to join the exclusive learning group for “AMD ROCm × Datawhale Embodied AI Policy Recreation Special Topic”.

[QR Code]

Scan the code to join the group, secure your seat, and join the students' X session. In 7 days, we will fully understand and implement the ACT, SmolVLA, and pi_0 policies on AMD ROCm!

Embodied AI is not just about concepts or losses. It involves considering hardware, data, actions, videos, and failures—as well as whether we can clearly explain a complex debugging process. Looking forward to working with you to make the phrase “successful replication” more solid in this special session.
