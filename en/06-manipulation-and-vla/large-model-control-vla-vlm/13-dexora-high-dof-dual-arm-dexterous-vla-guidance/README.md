# Dexora: An open-source VLA system for 36-DoF dual dexterous hands

> Introduction to this paper: **Dexora: Open-source VLA for High-DoF Bimanual Dexterity**. This work was conducted by teams from Tsinghua University, Beijing AI Institute, Hong Kong University, Shanghai Jiao Tong University, ShanghaiTech University, and Peking University. Its arXiv ID is `2605.18722v1`. The official ICRA 2026 award page listed Dexora as a candidate for the **Best Paper Award on Robot Manipulation and Locomotion**.

## 1. First, let's state the conclusion: Why it deserves to be included in the VLA section

Dexora addresses a very clear issue: most existing VLA systems rely on single-arm grippers, dual-armed low-degree-of-freedom grippers, or dexterous hands. They can perform tasks such as grasping, placing, and opening drawers, but when multiple tasks require **dual-arm coordination** and **fine contact with multiple fingers**, the motion space, data collection, and training stability become challenging.

Dexora aims to push VLA into a higher high-DoF robotic form:

```text
双臂 + 双灵巧手 + 36 DoF
    -> 切菜、揉面、拧瓶盖、从书架抽书、双手搬运、装配拆卸
```

Here, we need to correct a common misreading: In the promotional materials, it is often written as “6 degrees of freedom for both hands”, but in the papers and project pages, it is emphasized as **36-DoF dual-arm dual-hand system**. The `6-DoF Fingers` shown in the figure is a control condition, meaning that the finger with low degrees of freedom perform worse than `12-DoF Fingers` in the bottle cap turning task. It does not mean that Dexora has only 6 degrees of freedom.

The key to Dexora is not just replacing a more articulate VLM, but integrating three things into a single system link:

1. **High-DoF hardware and MuJoCo digital twin**: Physical robots and simulation robots have a consistent dual-arm dual-hand embodiment.
2. **Mixed teleoperated data collection**: The exoskeleton backpack handles large-scale dual-arm movements, while Apple Vision Pro markerless hand tracking handles fine motor actions.
3. **Quality-aware VLA training**: The offline discriminator assigns a quality score to the teleoperation clip, and then converts this score into a weighted loss for the diffusion-transformer policy, reducing the negative impacts of jitter, failures, and collisions.

Therefore, it should be classified under `06-策略抓取或抓取VLA/大模型控制、VLA、VLM`, rather than the world model section. It is the systems engineering and training paradigm of VLA for high-DoF hand manipulation, not an action condition video prediction model.

## 2. Paper, project, code, and data entry points

| Entry | Link | Current Status |
| :--- | :--- | :--- |
| Paper | [arXiv:2605.18722](https://arxiv.org/abs/2605.18722) / [HTML](https://arxiv.org/html/2605.18722v1) | Submitted on 2026-05-18 |
| Project Page | [dexoravla.github.io](https://dexoravla.github.io/) | Official project page, containing animated images, method diagrams, data visualizations, and result explanations |
| Code Entry | [dexoravla/Dexora](https://github.com/dexoravla/Dexora) | Repository referenced by the project page Code; README indicates it includes training, inference, data processing, and teleoperation code |
| Dataset | [Dexora/Dexora_Real-World_Dataset](https://huggingface.co/datasets/Dexora/Dexora_Real-World_Dataset) | Hugging Face dataset, containing real teleoperation data and task-level views |
| ICRA 2026 | [Award Finalists](https://2026.ieee-icra.org/awards/) | Shortlisted for Best Paper Award in Robot Manipulation and Locomotion |

The open-source status can be understood as follows: the project page, papers, code entry, and real dataset pages are already public. Currently, it is more suitable for **method reading, data structure understanding, material review, and light code entry inspection**. Full reproduction of high-DoF real robot experiments still requires dual hands hardware, remote operation devices, MuJoCo digital twin assets, model weights, and training computing power. Tutorials should not promise “one-click reproduction of real robot cooking/lever turning effects”.

## 3. Overview Diagram: Four Core Blocks of Dexora

![Dexora Overview](../../../../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/13-Dexora高自由度双臂灵巧VLA导读/assets/img_teaser.JPG)

**Figure 1: Overview of Dexora.** Above, three examples—Piston Insertion, Book Retrieval, and Bottle Opening—are used to illustrate why a single arm, gripper, and low-degree-freedom fingers are insufficient. In the middle, Dexora is highlighted for covering dual-arm, dual-hand, and dexterous capabilities. Below, dataset, architecture, performance, and cross-body generalization are presented in sequence.
Source: [ Official Dexora project page ](https://dexoravla.github.io/).

This image is the most suitable main image for this entire paper to be placed in a tutorial. It can be divided into four layers for viewing.

The first layer is **motivation**. For piston insertion, two robotic arms need to position and apply force simultaneously; for book retrieval, fingers need to insert into the creases of books, grasp, and extract them, which is difficult to achieve with ordinary grippers; for bottle opening, fingers need to cover, rotate, and stabilize the bottle body, and low-degree-of-freedom fingers lack the appropriate contact posture.

The second layer is **data**. Dexora constructed an embodiment-matched dataset: part of it comes from large-scale simulation trajectories in the MuJoCo digital twin, and the rest comes from real robot teleoperation. The paper abstract refers to `100K` simulated trajectories, `6.5M` frames, as well as `10K` real teleoperated episodes and `2.92M` frames; the current Hugging Face data card and code README describe the real data release scope as `12.2K` teleoperated episodes, `2.92M` frames, and `40.5h`. These scopes differ slightly, but in the tutorial, it can be uniformly understood as: Dexora used **10K+ level real teleoperation episodes + 100K level simulation trajectories**.

The third layer is the **training architecture**. Dexora does not feed all demonstrations equally to the policy; instead, it first trains a discriminator to estimate the quality of clips. Smooth and stable clips with effective tasks have higher weights; those with jitter, noise, and high failure rates have lower weights.

The fourth layer is **performance and generalization**. The paper reports that Dexora achieves a success rate of approximately `90%` on basic tasks, with an average success rate of `66.7%` on the dexterous benchmark, which is higher than the baseline `51.7%`. It also demonstrates the migration from a 36-DoF two-armed policy to a single-armed gripper, a two-armed gripper, and a single-armed low-degree-of-freedom hand.

## 4. Why is the high-DoF two-handed VLA difficult?

The action space for ordinary gripper tasks is usually low-dimensional:

```text
末端位姿增量 + 夹爪开合
```

The task of dexterous hand on both arms is much more complex:

```text
左臂姿态 + 右臂姿态
    + 左手多指关节
    + 右手多指关节
    + 双手接触时序
    + 物体受力和摩擦状态
```

This brings three difficulties.

First is the **increase in action dimension**. The higher the dimension, the sparser the state-action space that can be covered by the same amount of data. The gripper can learn to “approach, close, and lift” through a small amount of demonstration, while the dexterous hand must also learn when, how much, and how to avoid pushing objects away with each finger.

Second is **two-arm timing coupling**. Tasks involving both hands are not performed by separate left and right arms. For example, when turning the bottle cap, one hand stabilizes the bottle body while the other hand rotates the cap; when picking up a book, one hand may open up space, while the other hand inserts and pulls out. Any delay, overshoot, or contact failure on one side will be transmitted to the other side.

Thirdly, **behavior cloning introduces more data noise**. In high-DoF teleoperation, issues such as finger jitter, joint micro-vibration, unstable contact, collisions, and poor trajectory quality may occur. If these data are directly subjected to behavior cloning, the policy will also learn those unstable movements.

Dexora's discriminator-guided training addresses the third issue: acknowledging that teleoperation data is imperfect, and minimizing bad segments while maximizing good ones during the training process.

## 5. Hardware and teleoperation: exoskeleton tubes for arms, Vision Pro tubes for fingers

![Dexora robot](../../../../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/13-Dexora高自由度双臂灵巧VLA导读/assets/head.png)

**Figure 2: Appearance of Dexora dual-arm dexterous hand robot.** The Dexora target robot is a dual-arm, dual-dexterous hand, high-DoF platform designed for simultaneous hand operation and precise finger manipulation.
Source: [ Official Dexora project page ](https://dexoravla.github.io/).

Dexora's remote operation link is crucial. The paper states that it uses a hybrid teleoperation pipeline:

```text
外骨骼背包
    -> 捕捉双臂大范围 kinematics

Apple Vision Pro markerless hand tracking
    -> 捕捉手指精细 articulation

同一套动作
    -> 驱动物理双臂双手机器人
    -> 同步驱动 MuJoCo digital twin
```

This design has two advantages.

First, there is a division of labor between the arms and fingers for data collection. The position, speed, and large-scale movement of the arms are more stable with an exoskeleton; the finger joints are more convenient using Vision Pro's markerless tracking, eliminating the need for complex markers on each finger.

Second, real and simulation use the same interface. When the operator manipulates the physical robot, the MuJoCo digital twin can also provide a mirrored demonstration. This reduces the embodiment mismatch between sim-to-real, making it easier to train on both simulated and real data.

The project page also shows low-latency teleoperation, with a text description stating that the average end-to-end latency is approximately `11 ms`. This value is crucial for high-DoF manipulation: if the latency is too high, the operator will compensate excessively, resulting in more jittery movements, and the quality of subsequent VLA training will also decline.

## 6. Dataset: Why both “simulation scale” and “realistic detail” are required

![Dexora real dataset](../../../../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/13-Dexora高自由度双臂灵巧VLA导读/assets/img_realdataset.png)

**Figure 3: Dexora real-world objects and scenes.** The project page displays categories such as tools, assemblies, joints, and grippers. The real environment includes a variety of everyday objects, obstructions, messy backgrounds, and different lighting conditions.
Source: [ Official Dexora project page ](https://dexoravla.github.io/).

The data structure of Dexora can be divided into two parts:

| Data | Function | Public Statement |
| :--- | :--- | :--- |
| MuJoCo simulation data | Provides scale, task coverage, and embodiment-matched pre-training | 100K trajectories, 6.5M frames, approximately 361h of video |
| Real teleoperation data | Provides real contact, vision, friction, and finger details | Approximately 10K episodes on the paper/project page; 12.2K episodes, 2.92M frames, and 40.5h in the HF/README release |

Simulation data and real data have different responsibilities. Simulation is suitable for covering large-scale basic skills, such as pick-and-place, assembly, and articulation; real data, on the other hand, is ideal for providing high-DoF finger contact, object material, lighting, occlusion, and manipulation strategies.

The Hugging Face dataset page states that the real data includes four synchronized perspectives: the head ego-centric camera, the left and right wrist cameras, and a third-person scene camera, along with 36-DoF robot proprioception. The data structure follows the LeRobot / LIBERO-2.1 style, including observation, robot state, action, and language instruction.

For learners, the value of this dataset lies not only in “having many videos”. More importantly, it breaks down high-DoF hand manipulation into a standard format that can be studied:

```text
multi-view RGB
    + 36-DoF proprioception
    + low-level action
    + language instruction
    + task/category metadata
```

This allows subsequent research to conduct experiments such as action representation, quality filtering, imitation learning, data resampling, and ontology cross-transferment using the same set of data.

## 7. Method Diagram: discriminator-guided quality-aware training

![Dexora method](../../../../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/13-Dexora高自由度双臂灵巧VLA导读/assets/img_method.png)

**Figure 4: Dexora’s quality-aware training process.** On the left, data filtering is performed using kinematic smoothness and simulation replay. In the middle, the pre-trained diffusion transformer is frozen, and the discriminator is trained to assign a quality score to each clip. On the right, during post-training, the quality score is converted into diffusion loss weights.
Source: [ Dexora official project page ](https://dexoravla.github.io/).

The training core of Dexora can be described in three steps.

The first step is **data filtering**. Real teleoperation data are first screened based on kinematic metrics, such as acceleration and jerk. If a segment has very erratic joint changes or chaotic contact, it usually indicates poor teaching quality. Subsequently, through replay / post-validation, we check whether the task is completed and whether collisions occur, and filter out obvious bad segments.

Step 2 is **discriminator training**. While the pre-trained diffusion transformer policy is frozen, Dexora uses observations and language as conditions to train a discriminator that outputs a clip quality score `d(C_t) ∈ (0, 1]`. The quality score of a high-quality clip approaches 1, while that of a low-quality clip approaches 0.

The third step is **quality-aware post-training**. When training the policy, different clips contribute differently to the loss:

```text
高质量 clip -> 更大权重
低质量 clip -> 更小权重
```

This is particularly important for high-DoF. Low-DoF gripper policies can sometimes mask teach jitter through post-processing or controller filtering; in two-handed dexterous manipulation, minor vibrations at the fingers can directly lead to slipping, jamming, hitting objects, or asynchronous left-right hand movements. The discriminator in Dexora actually tells the model: do not learn all teaches equally, but prioritize learning segments that are smooth, stable, and capable of completing tasks.

During inference, only the policy is used, and the discriminator is not involved in online control. Therefore, the discriminator is a quality regulator during training, not a reward model at runtime.

## 8. An official GIF: Why It Is Difficult for the Gripper to Knead dough

![Dexora roll dough gripper](../../../../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/13-Dexora高自由度双臂灵巧VLA导读/assets/RollDough_gripper.gif)

**Figure 5: Comparison example of the Roll Dough gripper.** The project page uses a comparison between the gripper and the hand to illustrate that tasks such as kneading dough, turning bottle caps, picking books, and writing with a pen are not simply about grasping and releasing. The contact area of the fingers, posture, and continuous force application directly affect the success rate.
Source: [ Dexora official project page ](https://dexoravla.github.io/).

The focus of these examples is not simply “looking like a human hand,” but rather to illustrate that the task itself has requirements for embodiment. Kneading dough requires continuous rolling, pressing, and maintaining contact; turning a bottle cap requires one hand to hold the bottle steady while the other hand applies torque; picking up a book requires fingers to insert into thin gaps and maintain a stable grip. Grippers can perform many industrial grasping tasks, but they are easily limited in these everyday dexterous tasks.

This also holds research value for Dexora: instead of creating another tabletop pick-and-place VLA, it expands the motion range of the VLA to a region closer to the daily operations of a human hand.

## 9. How to interpret the results: Strong, but not "all delicate manipulations have been resolved"

The core results presented on the paper and project pages can be summarized as follows:

| Evaluation Direction | Conclusion on paper/project page | How to understand it |
| :--- | :--- | :--- |
| Basic tasks | Average success rate is approximately 90% for pick-and-place, assemble/disassemble, and articulated objects | Bilateral VLA can handle not only dexterous tasks but also basic operations |
| Dexterous tasks | Average dexterous success `66.7%`, compared with baseline `51.7%` | High-DoF fingers and mass perception training yield significant benefits |
| OOD generalization | Tests unseen background, lighting, objects, occlusion, clutter, and height changes | Ensures stability under visual and physical disturbances |
| Cross-embodiment | Migration from 36-DoF policy to single-arm gripper, dual-arm grippers, and single-arm low-DoF hand | Migration from high-DoF to low-DoF is relatively feasible, but from low-DoF to high-DoF remains challenging |
| Discriminator ablation | Actions are smoother and tasks are more likely to succeed when a discriminator is present | Data quality is one of the core contributions |

Here, we cannot claim that Dexora’s high-DoF manipulation is completely solved. The project page also highlights failure cases, including instruction ambiguity, misjudgment of visually similar objects, insufficient grasping friction, asynchronous bilateral arms, pose drift, and violation of the workspace boundary. These failures clearly indicate that high-DoF VLA still has many open issues: language grounding, physical contact, long-term error accumulation, bilateral arm synchronization, and hardware workspace constraints.

## 10. How to reproduce now

Dexora is not suitable for being described as a tutorial that “everyone can reproduction the real robot cooking in one click”. A more reasonable approach is to learn it in three stages.

### 10.1 Review of Paper and Project Pages

The first level only requires reading the papers, project pages, and official figures. Focus on answering:

- Why is a single-arm/gripper/low-degree-of-freedom hand insufficient?
- Which parts are included in the 36-DoF action space?
- What problems do simulation and real-world data respectively address?
- Why can the discriminator reduce teleoperation noise?
- What are the OOD and cross-embodiment results that prove, and what they do not prove?

This is the most suitable activity for team-based learning for Task04. No hardware or high computing power is required.

### 10.2 Dataset Structure Check

The second tier can start with Hugging Face dataset. The dataset page shows the episode-centric / task-level structure, and the core fields include:

```text
data/chunk-xxx/episode_xxxxxx.parquet
videos/chunk-xxx/observation.images.*
meta/info.json
meta/episodes.jsonl
meta/episodes_stats.jsonl
meta/modality.json
meta/stats.json
meta/tasks.jsonl
```

Lightweight tasks that can be performed include:

1. Download a subset of small tasks.
2. Review `meta/info.json` and `modality.json`.
3. Read a parquet episode.
4. Align multi-view videos, 36-DoF state/action, and language instructions.
5. Write a report page explaining the differences between the Dexora data format and standard gripper dataset.

### 10.3 Code entry and complete reproduction boundary

The third level is where we review the code. The current Code on the project page points to:

```bash
git clone https://github.com/dexoravla/Dexora
cd Dexora
```

According to the README description, the repository is organized around these parts:

```text
training pipeline
real-robot inference stack
BSON -> LeRobot v2.1 converters
Vision-Pro teleoperation tools
MuJoCo digital twin / replay validation
```

To reproduce the experiment completely, the following is required:

- Dexterous hand hardware or equivalent digital twin.
- Exoskeletons and remote manipulation links for Apple Vision Pro.
- MuJoCo assets and robot models.
- Permission to download real or simulation data and sufficient disk space.
- Training power for diffusion-transformer policy.
- Safety clipping, collision checking, synchronization control, and emergency stop mechanisms.

Therefore, the tutorial currently recommends using Dexora as a **method breakdown + data structure learning + navigation to code entries**. Once the code, model, simulation data, and hardware configuration become more stable, we can then focus on reproduction step by step.

## 11. Relationship with Other VLAs

| Method | Key Enhancements | Differences from Dexora |
| :--- | :--- | :--- |
| OpenVLA / OpenVLA-OFT | Open-source VLA foundation and fine-tuning pipeline | Dexora focuses on high-DoF dual-arm dual-hand embodiment |
| π0 / π0.5 | Strong general manipulation VLA baseline | Dexora emphasizes dual-hand dexterous manipulation and high DoF data |
| EventVLA | Long-time-domain visual evidence memory | Dexora focuses more on action space, teleoperation data, and quality weighting |
| 3DVLA | 3D space and instance understanding | Dexora focuses more on dual-arm dual-hand actuator morphology and high-dimensional actions |
| PhysBrain | Physical常识 pre-training | Dexora is trained directly on real high-DoF contact data |
| PRTS | Target accessibility/CRL pre-training | Dexora uses a discriminator to handle teleoperation quality |
| G0.5 | Autoregressive CoT + action token flow | Dexora uses a diffusion-transformer policy and quality-aware training |
| HumanoidMimicGen | Generation of full-body data via minimal humanoid teaching | Dexora is a dual-arm dual-hand dexterous manipulation VLA, not the generation of full-body loco-manipulation data |

If categorized by "Why VLA failed":

- EventVLA believes the failure stems from the loss of long-term visual evidence.
- 3DVLA believes the failure arises from insufficient understanding of 3D spatial instances.
- PRTS believes the failure is due to inadequate target accessibility and task progress perception.
- G0.5 believes the failure results from the fragmentation of reasoning, action tokens, and cross-ontology interfaces.
- Dexora believes the failure also stems from the simplicity of the embodiment: high-DoF grippers cannot replicate the real-world dexterous manipulation of two hands, while high-DoF remote operation data itself requires quality modeling.

## 12. Recommended Tasks

In team learning, you can place Dexora in the manipulation control direction Task04:

> Comparing G0.5, PRTS, and Dexora: G0.5 focuses on unified autoregressive manipulation sequences, PRTS emphasizes target accessibility pre-training, and Dexora highlights 36-DoF dual arms embodiment, hybrid teleoperation, and quality perception training. Which bottleneck of VLA does each of these models address?

The assignment can answer these questions:

- Why isn’t Dexora a regular two-arm gripper VLA?
- What are exoskeletons and Apple Vision Pro responsible for in remote manipulation?
- Why are both simulation data and real data required?
- Why is discriminator-guided quality-aware training suitable for high-DoF teleop data?
- What does the入围 of ICRA 2026 indicate, and what doesn’t?
- Which step is currently feasible with the public entry, and what conditions are still missing for complete reproduction on a real robot?

## 13. Citations and Image Sources

The images and GIFs in this article come from [Dexora's official project page ](https://dexoravla.github.io/), and have been saved locally `assets/` to ensure stable access for the tutorial. The text is based on [arXiv:2605.18722](https://arxiv.org/abs/2605.18722), [dexoravla/Dexora GitHub repository ](https://github.com/dexoravla/Dexora), [Dexora Real-World Dataset ](https://huggingface.co/datasets/Dexora/Dexora_Real-World_Dataset), and [ICRA 2026 Award Finalists ](https://2026.ieee-icra.org/awards/).
