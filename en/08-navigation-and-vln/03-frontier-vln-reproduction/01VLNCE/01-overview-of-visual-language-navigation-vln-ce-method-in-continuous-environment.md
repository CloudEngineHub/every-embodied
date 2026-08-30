# Visual-Language Navigation in Continuous Environments (VLN-CE)

## Task Definition

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-12.png" width="80%">
</div>

Visual Language Navigation (VLN) is a key research issue in embodied AI, aiming to enable agents to navigate in an invisible environment according to linguistic instructions. VLN requires robots to understand complex and diverse visual observations while interpreting instructions at different granularities.

VLN input typically consists of two parts: visual information and natural language instructions. Visual information can be videos of past trajectories or a collection of historical-current observation images. Natural language instructions include the goals that the embodied AI needs to achieve, or the tasks that the embodied AI expects to perform. The embodied agent must use the above information to select one or a series of actions from a candidate action list to fulfill the requirements of the natural language instructions.

### Task Setup

The Visual Language Navigation (VLN) task can be uniformly modeled in mathematical form as a partially observable Markov decision process (POMDP), described by tuples:

$$
\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{O}, \mathcal{R}, \mathcal{J} \rangle
$$

Among them, $\mathcal{S}$ represents the state space, $\mathcal{A}$ is the action space of the agent, $\mathcal{T}$ is the state transition function, $\mathcal{O}$ is the visual observation space, $\mathcal{R}$ is the reward function, and $\mathcal{J}$ is the natural language instruction.

Under this framework, the core of navigation problems lies in learning a policy mapping function $a_t = \pi(o_t, I, s_t)$ (typically parameterized as a deep neural network model $F$ ). That is, at time step $t$, the agent derives the optimal action $a_t$ based on the current first-person visual observation $o_t$, global natural language instructions $I$, and the agent's state $s_t$, in order to maximize the cumulative reward and approach the target position.

Based on differences in the state space, action space, and state transition function, VLN tasks are divided into discrete VLN and continuous VLN.

---

#### Discrete-Environment Visual-Language Navigation Task Setup

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-13.png" width="80%">
  <br>
  <b> Figure 1: Discrete environment VLN. Blue connecting lines represent the predefined connected graph; green lines represent expert trajectories. </b>
</div>
<br>

Discrete VLN research is primarily carried out using the Matterport3D simulator. As shown in Figure 1, the environment is abstracted as a predefined connected graph $\mathcal{G}=\langle V,\mathcal{E}\rangle$. The blue nodes in the graph represent a set of fixed panoramic viewpoints, and $V$ indicates the connectivity between these viewpoints, while $\mathcal{E}$ represents the connectivity relationships.

* **State space (** $\boldsymbol{S}_{disc}$ **):** The state $S_t$ of the agent is strictly limited to a node in the graph $\mathcal{G}$, namely $s_t \in V$.
* **Action space (** $\mathcal{A}_{disc}$ **):** It is assumed that at time $t$, all connected nodes in the graph $\mathcal{G}$ surrounding the agent’s state $s_t$ are represented as $N(s_t)$. Each node in $N(s_t)$ is called a candidate node, and $N(s_t)$ is the action space of the agent at time $t$. The action decision at time $t$ by the agent is represented as “selecting a candidate node”. The policy function $F$ outputs a probability distribution for $N(s_t)$, which is used to choose a node from the candidate nodes as the next “jump point”.
* **State transition (** $\mathcal{T}_{disc}$ **):** State transition occurs through a deterministic “invisible teleportation” mechanism. Once the next “jump point” is generated, the system instantly switches the agent’s position to the coordinates of the selected candidate node. This process ignores intermediate paths and does not consider physical collisions or friction. Although this “Oracle Navigation” setting simplifies control complexity, it significantly deviates from the motion laws of real robots in real environments.

---

#### Continuous-Environment Visual-Language Navigation Task Setup

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-14.png" width="80%">
  <br>
  <b> Figure 2 Continuous Environment VLN. Green lines represent expert trajectories. </b>
</div>
<br>

To bridge the gap between the simulation environment and the real world, VLN-CE is trained in the Habitat simulator. As shown in Figure 2, this environment loads a highly realistic indoor 3D scanning scene from the Matterport3D dataset, providing a first-person visual experience consistent with the real world. Meanwhile, Habitat introduces a navigation grid to define the physical motion space. This grid transforms the walkable areas of the environment into a series of tightly connected polygonal surfaces, allowing the agent to remain at any continuous coordinate point within the grid range. It is this spatial continuity that enables the agent to perform basic actions such as “moving a fixed distance” and “rotating by a fixed angle” with the support of the physics engine. When the agent executes a step forward command (e.g., 0.25 meters), the navigation grid ensures that this is not just a jump in coordinate values, but a real sliding displacement process on the continuous geometric surface, thereby handling continuous pose changes similar to those in the real world. Compared to the “jumping” movement of the agent in discrete VLN from one node to an adjacent node, the smooth motion of the agent in continuous VLN based on the navigation grid more closely mirrors the movement patterns in a real environment.

* **State space (** $\boldsymbol{S}_{cont}$ **):** The state of the agent is represented by precise three-dimensional coordinates $P=(x,y,z)$ and orientation angles $\Phi =(\theta,\psi)$ in the world coordinate system, where the former represents the pitch angle and the latter represents the yaw angle.
* **Action space (** $\mathcal{A}_{cont}$ **):** The action space of the agent at any given moment consists of a set of low-level action primitives, as shown by the “Low-Level Actions” label in Figure 2. These include moving forward at a fixed distance (e.g., 0.25 meters), rotating by a fixed angle to the left (e.g., 15°), rotating by a fixed angle to the right (e.g., 15°), and stopping. The action decision at time $t$ is characterized by “selecting an action primitive”, and the policy function outputs a probability distribution over all action primitives, from which one specific motion control command is chosen.
* **State transition (** $\mathcal{T}_{cont}$ **):** State transitions $\mathcal{T}(s_t,a_t)$ are strictly constrained by the dynamics rules of the simulator’s physics engine. When the agent performs an action that causes contact with a physical object in the environment, a “sliding” or collision mechanism is triggered.

---

## VLN-CE Datasets

### Common dataset

#### 1. R2R-CE

**Simulator**
* **Simulation Environment:** Habitat-Sim v0.1.7
* **Environment Scenarios:** 90 real indoor scenarios based on the Matterport3D (MP3D) dataset (houses, offices, etc.).

**Input **

**Observation space at every moment ($O_t$)**

* **Visual input:** RGB-D images (RGB + depth maps).
* **Resolution:** Typically, the resolution of RGB images is (224×224), and that of Depth images is (256×256).
* **Field of view (FOV):** 90 degrees.
* **Viewing angle:** Here, it defaults to a monocular view. Many models such as ETPNav construct a panoramic view by rotating from left to right 12 times (at intervals of 30°).

**Text Instruction**
* **Language:** English only.
* **Type:** **Step-by-Step**. The instruction not only tells the agent “where to go (destination)”, but also provides a description of an **action sequence**. It explains how the agent should navigate step by step from the current location to the endpoint.
* **Example:** “Go down the stairs, turn left at the picture and enter the bedroom.”

**Output (Action Space)**
* The agent outputs **discrete low-level control commands**:
    1. `MoveForward` (move forward 0.25 meters)
    2. `TurnLeft` (rotate left by 15 degrees)
    3. `TurnRight` (rotate right by 15 degrees)
    4. `Stop` (stop and check if the goal is reached)
* **Decision criterion:** When the agent performs `Stop`, if the distance to the target point is within **3.0 meters**, it is considered a success.

**Dataset Size**
* **Number of paths:** Approximately **4,475** continuous trajectories (reconstructed from discrete R2R paths).
* **Number of instructions:** Approximately **13,425** (with an average of 3 instructions per path).
* **Average path length:** Approximately **9.89 meters**.
* **Division:** Training set (10.8k), Validation set (Seen: 778 / Unseen: 1.8k), Test set (3.4k).

#### 2. RxR-CE

**Simulator**
* **Simulation Environment:** Habitat-Sim v0.1.7
* **Environment Scene:** Also based on the Matterport3D (MP3D) dataset. However, compared to R2R-CE, RxR-CE sampling has a higher path coverage rate, exploring more complex areas and loops.

** Input **

**Observation space at every moment ($O_t$)**

* **Visual input:** RGB-D images (RGB + depth maps).
* **Resolution:** Typically, the resolution of RGB images is (224×224), and that of Depth images is (256×256).
* **Field of view (FOV):** 79 degrees.
* **Note:** The default is a monocular perspective.

**Text Instruction**

* **Instruction language:**  **Multilingual**. Includes **English (English), Hindi (Hindi), Telugu (Telugu)**.
* **Instruction type:**  **Process-Oriented**. The instructions are very detailed, describing not only the sequence of actions but also all visual details along the way, like a "guide". Most importantly, the words in the instructions align with the agent’s pose on the path over time.
* **Example:** "Stand on the rug, turn right and face the wooden cabinet. Walk past the cabinet, keeping it on your left, and enter the hallway. You will see a painting of a flower on the wall. Stop just after passing the painting."

**Output**

* **Action space:** Consistent with R2R-CE.
* **Decision criteria:**
    * **Success criterion:** Within 3.0 meters of the target point.
    * **Path consistency (NDTW):** The task emphasizes whether the agent **truly** follows the route described in the instructions.

**Dataset Scale**
* **Number of paths:** Approximately **42,000+** continuous trajectories (one order of magnitude larger than R2R).
* **Number of instructions:** Approximately **126,069**.
* **Average path length:** Approximately **15 meters**.

> **Dataset acquisition address:** [https://github.com/jacobkrantz/VLN-CE](https://github.com/jacobkrantz/VLN-CE)

---

### Evaluation Criteria

**(1) Navigation Error (NE)**
Defined as the average geodesic distance (in meters) between the final stopping position of the agent and the target position.

$$
NE = \frac{1}{N} \sum_{i=1}^N \mathrm{dist}(p_{final}^{(i)}, p_{target}^{(i)})
$$

**(2) Success Rate (SR)**
Defined as the proportion in which the geodesic distance between the agent's stopping position and the target position is less than a specific threshold (standardly set at 3m).

$$
SR = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\mathrm{dist}(p_{final}^{(i)}, p_{target}^{(i)}) < 3m)
$$

**(3) Success weighted by Path Length, SPL**
SPL is a core metric of VLN-CE. It aims to balance "success rate" and "path length".

$$
SPL = \frac{1}{N} \sum_{i=1}^N S_i \frac{l_i}{\max(l_i, p_i)}
$$

**(4) Oracle Success Rate (OSR)**
If the geodesic distance between any point in the agent’s trajectory and the target is less than the threshold, it is considered a success. OSR measures the agent’s potential to “pass” through the target. The gap between OSR and SR usually reflects defects in the agent’s stop policy.

$$
OSR = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\min_t \mathrm{dist}(p_t^{(i)}, p_{target}^{(i)}) < 3m)
$$

**(5) Normalized Dynamic Time Warping (nDTW)**
nDTW uses a dynamic programming algorithm to calculate the minimum cumulative distance between the predicted trajectory $P$ and the reference trajectory $Q$, and then performs exponential normalization.

$$
nDTW = \exp\left(-\frac{DTW(P,Q)}{\sqrt{|P|\cdot|Q|}}\right)
$$

**(6) SDTW (Success weighted by normalized Dynamic Time Warping)**
SDTW calculates the nDTW metric of tasks that successfully reach the endpoint, serving as a comprehensive indicator of "command compliance" and "task completion rate".

$$
SDTW = \frac{1}{N} \sum_{i=1}^N S_i \cdot \text{nDTW}_i
$$

## Methods for Continuous-Environment Visual-Language Navigation

### 1. Traditional Single-Stage Methods

#### Basic End-to-End Baseline Model (ECCV 2020)

**Paper Title:** Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments
**Authors:** Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, Stefan Lee
**Institution:** Oregon State University, Georgia Tech, FAIR
**Paper Link:** [PDF Link](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730103.pdf) | **Project Link:** [Project Page](https://jacobkrantz.github.io/vlnce/)

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-6.png" width="80%">
</div>

**Method Overview:**
It mainly includes the Sequence-to-Sequence (Seq2Seq) model and the Cross-Modal Attention (CMA) model. It adopts an end-to-end architecture, and the model directly outputs the probability of low-level actions.
* **Seq2Seq Architecture (Figure a):** The language encoder (LSTM) compresses the instruction into semantic vectors; the action decoder (RNN) predicts the action distribution based on the semantic vectors and current visual features.
* **CMA Model (Figure b):** An attention mechanism is introduced. The current visual features are used as Queries, and cross-attention calculations are performed with the instruction feature sequence. This dynamically focuses on relevant instruction segments, achieving fine-grained alignment between vision and language.

---

### 2. Traditional Two-Stage Navigation Methods

Using a hierarchical structure, the navigation is divided into two stages: "decision-making" and "execution".
1.  **Decision-making stage:** Based on the panoramic map and instructions, the next high-level goal point (Local Goal / Waypoint) is predicted, usually at the $r, \theta$ coordinate.
2.  **Execution stage:** The underlying controller receives the target point coordinates, and an algorithm (such as DWB) automatically plans the path and controls the agent to reach that point.

#### (1) Waypoint Predictors: The Beginning of Two-Stage Navigation

**Paper Title:** Bridging the Gap Between Learning in Discrete and Continuous Environments for VLN (2022 CVPR)
**Authors:** Yicong Hong, Zun Wang, Qi Wu, Stephen Gould
**Institution:** ANU, University of Adelaide
**Paper URL:** [PDF Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Hong_Bridging_the_Gap_Between_Learning_in_Discrete_and_Continuous_Environments_CVPR_2022_paper.pdf) | **Project URL:** [GitHub](https://github.com/YicongHong/Discrete-Continuous-VLN)

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-7.png" width="80%">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-8.png" width="80%">
</div>

**Methodological Innovations:**
* **Candidate Waypoint Predictor:** Generates real-time "virtual nodes" that are accessible in continuous environments based on depth maps, making the continuous space appear like a discrete navigation map.
* **Hierarchical Navigation System:** Divided into “high-level planning” and “low-level execution”.
* **Discrete-to-Continuous Migration:** Demonstrates that fine-tuning a model trained in a discrete environment (Nav-Graph) can achieve SOTA performance in a continuous environment.

#### (2) Map-Based Approaches

##### Topological Map: ETPNav

**Paper Title:** ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments (2024 TPAMI)
**Authors:** Dong An, Hanqing Wang, Wenguan Wang, Zun Wang, Yan Huang, Keji He, Liang Wang
**Institution:** CASIA, UCAS
**Paper URL:** [PDF Link](https://arxiv.org/abs/2304.03047) | **Project URL:** [GitHub](https://github.com/MarSaKi/ETPNav)

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-9.png" width="80%">
  <p>ETP navigation framework</p>
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-10.png" width="80%">
  <p>The process of ETPNav generating the topology diagram at time t</p>
</div>

**Method Overview:**
The core of ETPNav is **Online Evolving Topological Mapping**. It converts the unknown continuous physical space into a dynamically updated topological map in real time, storing historical information through the topological map.
* **Cross-modal planner:** Performs long-range semantic reasoning based on the topological map and instructions to select the optimal waypoint.
* **Trial-and-Error Controller:** Responsible for low-level action execution and robust obstacle avoidance, solving the problems where traditional models tend to get stuck or collide.

##### BEV Map: BEVBert

**Paper Title:** BEVBert: Multimodal Map Pre-training for Language-guided Navigation (2023 ICCV)
**Authors:** Dong An, Yuankai Qi, Yangguang Li, Yan Huang, Liang Wang, Tieniu Tan, Jing Shao
**Institution(s):** CASIA, UCAS, University of Adelaide, SenseTime, NJU, PJLab
**Paper Link:** [PDF Link](https://arxiv.org/pdf/2212.04385) | **Project Link:** [GitHub](https://github.com/MarSaKi/VLN-BEVBert)

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-11.png" width="80%">
</div>

**Method Overview:**
Proposes a multimodal map pre-training paradigm based on spatial perception.
* **Hybrid Map:** Combines a local bird's-eye view (BEV) map used to eliminate visual overlaps, and a global topological map for remembering long-distance paths.
* **Pre-training Tasks:** Through tasks such as mask map modeling, the model can perform cross-modal inference directly on map representations.

#### (3) Methods Based on Future Imagination

##### HNR

**Paper Title:** Lookahead Exploration with Neural Radiance Representation for Continuous Vision-Language Navigation (2024 CVPR)
**Authors:** Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, Junjie Hu, Ming Jiang, Shuqiang Jiang
**Institution:** ICT, CAS, UCAS
**Paper Link:** [PDF Link](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Lookahead_Exploration_with_Neural_Radiance_Representation_for_Continuous_Vision-Language_Navigation_CVPR_2024_paper.pdf) | **Project Link:** [GitHub](https://github.com/MrZihan/HNR-VLN)

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-15.png" width="48%">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-16.png" width="48%">
</div>

**Method Overview:**
The **Hierarchical Neural Radiation Representation (HNR) framework** is proposed.
* **Lookahead Exploration:** By using CLIP semantic embedding to synthesize environmental features, and constructing a **future path tree** that extends beyond the current horizon, the agent can simultaneously evaluate the semantic match of multiple potential branches, thereby achieving a globally optimal path selection that balances long-term planning.

##### NavMorph (World Model)

**Paper Title:** NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments (2025 ICCV)
**Authors:** Xuan Yao, Junyu Gao, Changsheng Xu
**Institution:** CASIA, UCAS, Peng Cheng Lab
**Paper URL:** [PDF Link](https://openaccess.thecvf.com/content/ICCV2025/papers/Yao_NavMorph_A_Self-Evolving_World_Model_for_Vision-and-Language_Navigation_in_Continuous_ICCV_2025_paper.pdf) | **Project URL:** [GitHub](https://github.com/Feliciaxyao/NavMorph)

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-17.png" width="80%">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-18.png" width="80%">
</div>

**Methodological Innovations:**
* **Self-evolutioning world model:** Builds an evolving latent space through online interaction.
* **Contextual Evolution Memory (CEM):** Uses a forward iteration update mechanism to quickly adapt to new scenarios during online testing.
* **Hierarchical architecture:** Comprises a “world-aware navigator” and a “forward action planner”. It abandons pixel-level prediction and replaces it with future prediction and reconstruction at the **feature level**, significantly improving efficiency.

#### (4) Sim-to-Real Methods

##### Sim2Real-VLN-3DFF

**Paper Title:** Sim-to-Real Transfer via 3D Feature Fields for Vision-and-Language Navigation (2024 CoRL)
**Author:** Zihan Wang, et al.
**Paper Link:** [PDF Link](https://proceedings.mlr.press/v155/anderson21a/anderson21a.pdf) | **Project Link:** [GitHub](https://github.com/MrZihan/Sim2Real-VLN-3DFF)

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-19.png" width="48%">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-20.png" width="48%">
</div>

**Core idea:**
To address the contradiction where real robots are typically equipped with **monocular cameras** but require **panoramic input**.
* **3D Feature Field:** Using NeRF concepts, map monocular observations into 3D space to render virtual panoramas.
* **Semantic traversable map:** A central map for agents, which also performs environmental semantics alignment and obstacle prediction.

##### monoVLN

**Paper Title:** monoVLN: Bridging the Observation Gap between Monocular and Panoramic Vision and Language Navigation (2025 ICCV)
**Authors:** Renjie Lu, Yu Zhou, Hao Cheng, Jingke Meng, Wei-Shi Zheng
**Institution:** SYSU, HNU
**Paper Link:** [PDF Link](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_monoVLN_Bridging_the_Observation_Gap_between_Monocular_and_Panoramic_Vision_ICCV_2025_paper.pdf)

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-21.png" width="80%">
</div>

**Methodological Innovations:**
* **3D Gaussian Splatting (3DGS):** Use 3DGS to rapidly reconstruct 3D feature fields and render virtual panoramas.
* **Implicit Local Completion:** Automatically predict and fill in the feature voids caused by occlusions.
* **Uncertainty-Aware Policy:** When the feature confidence is low, actively perform "turning" actions to collect data.

---

### 3. Zero-Shot Visual-Language Navigation

#### Open-Nav (ICRA 2025)
**Paper:** [Open-Nav: Exploring Zero-Shot VLN with Open-Source LLMs](https://arxiv.org/abs/2409.18794)
**GitHub:** [Link](https://github.com/YanyuanQiao/Open-Nav)
<div align="center"><img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-23.png" width="80%"></div>

#### CA-Nav (TPAMI 2025)
**Paper:** [Constraint-Aware Zero-Shot VLN](https://arxiv.org/pdf/2412.10137)
**GitHub:** [Link](https://chenkehan21.github.io/CA-Nav-project/)
<div align="center"><img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-24.png" width="48%"><img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-25.png" width="48%"></div>

#### Other Recent Work
* **LaViRA:** [Language-Vision-Robot Actions Translation](https://arxiv.org/abs/2510.19655)
* **SmartWay (2025 IROS):** [Enhanced Waypoint Prediction](https://arxiv.org/abs/2503.10069)
* **Fast-SmartWay:** [Panoramic-Free End-to-End Zero-Shot](https://arxiv.org/pdf/2511.00933)

---

### 4. Methods Based on Vision-Language-Action Models

The VLA model integrates perception, understanding, and control into a end-to-end framework, gradually replacing the modular architecture.

#### NaVILA

**Paper Title:** NaVILA: A Legged Robot Vision-Language-Action Model for Navigation
**Authors:** An-Chieh Cheng, et al.
**Institution:** UCSD, USC, NVIDIA
**Paper Link:** [PDF Link](https://navila-bot.github.io/static/navila_paper.pdf) | **GitHub:** [Link](https://github.com/AnjieCheng/NaVILA)

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-26.png" width="80%">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-27.png" width="80%">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-28.png" width="80%">
</div>

**Method:**
* **Advanced VLA Layer:** Outputs “intermediate actions” (such as “turn left by 30 degrees”), using internet video data augmentation.
* **Low-Level Execution Layer:** A policy based on RL, converting language commands into motor control commands.

#### StreamVLN

**Paper Title:** StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling
**Author:** Meng Wei, et al.
**Institution:** PJLab, HKU
**Paper Link:** [PDF Link](https://arxiv.org/abs/2507.05240) | **GitHub:** [Link](https://github.com/InternRobotics/StreamVLN)

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-29.png" width="80%">
</div>

**Core: SlowFast Context Modeling**
* **Fast-Streaming Context:** Sliding window mechanism, focusing only on recent frames to ensure real-time response.
* **Slow-Updating Memory:** 3D Token pruning strategy, retaining only key landmarks to solve VRAM overflow issues and enable long-term streaming navigation.

#### DualVLN

**Paper Title:** GROUND SLOW, MOVE FAST: A DUAL-SYSTEM FOUNDATION MODEL
**Author:** Meng Wei, et al.
**Institution:** PJLab, HKU, Tsinghua
**Paper Link:** [PDF Link](https://arxiv.org/pdf/2512.08186) | **GitHub:** [Link](https://github.com/InternRobotics/InternNav?tab=readme-ov-file)

<div align="center">
  <img src="../../../../08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-30.png" width="80%">
</div>

**Method: Simulating Human "Dual Process Theory"**
* **System 2 (Slow and Grounded):** Brain. The InternVLA large model is used for global planning, generating mid-term points (Pixel Goal) with low-frequency outputs.
* **System 1 (Fast Movement):** Cerebellum. A lightweight Diffusion Transformer outputs line speed and angular velocity, handling obstacle avoidance and motion control.
