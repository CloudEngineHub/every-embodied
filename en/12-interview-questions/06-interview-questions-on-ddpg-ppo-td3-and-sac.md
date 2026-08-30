# Interview Questions on Ddpg Ppo Td3 and Sac

## 1. You must be familiar with the principles and differences of algorithms such as DDPG, PPO, TD3, and SAC.

These four algorithms are the most commonly used reinforcement learning algorithms in continuous control (such as robot control).

**PPO (Proximal Policy Optimization)**

- **Principle**: PPO is a on-policy policy gradient algorithm. It uses a clipping mechanism to limit the update steps of old and new policies, preventing policy updates from failing.

- **Core formula**:

  $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$

- **Numerical calculation example**:

Assume that at time step $t$, the GAE calculates a dominance function $\hat{A}_t = 10$ (positive, indicating this action is good).

  Set the truncation hyperparameter $\epsilon = 0.2$.

If the new network is too aggressive, resulting in a probability ratio between old and new policies of $r_t(\theta) = 1.5$ (the probability of the new policy taking this action is 1.5 times that of the old policy):

  1. Original target: $1.5 \times 10 = 15$

  2. Truncation target: $\text{clip}(1.5, 0.8, 1.2) \times 10 = 1.2 \times 10 = 12$

  3. Take the minimum value: $\min(15, 12) = 12$

    **Conclusion**: Even if the network wants to be updated significantly to 15, PPO forces it to be truncated to 12, ensuring the stability of training.

**DDPG (Deep Deterministic Policy Gradient)**

- **Principle**: An Actor-Critic algorithm with an off-policy policy. The Actor outputs deterministic actions, and the Critic evaluates Q-values.

- **Core formula (Updated by Critic)**:

  $$y_i = r_i + \gamma Q_{\phi'}(s_{i+1}, \mu_{\theta'}(s_{i+1}))$$

- **Numerical calculation example**:

  Assume the robot takes a step forward and receives a reward $r_i = 1.0$. Discount factor $\gamma = 0.99$.

  The target network (Target Actor and Target Critic) predicts the optimal Q value for the next state as $Q_{\phi'} = 100$.

  Calculate the TD target value (Note 1): $y_i = 1.0 + 0.99 \times 100 = 100.0$.

If the Q value predicted by the current Critic network is 90, the Mean Squared Error (MSE) Loss will be $(100.0 - 90)^2 = 100$.

**TD3 (Twin Delayed DDPG)**

- **Principle**: To address the issue of over-estimated Q values in DDPG, a dual Q network, delayed update, and target policy smoothing are introduced.

- **Core formula (truncated double Q learning and smoothing)**:

  $$y = r + \gamma \min_{j=1,2} Q_{\phi'_j}(s', a')$$

  $$a' = \mu_{\theta'}(s') + \epsilon, \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$$

- **Numerical calculation example**:

  The target Actor gives the next action $a_{raw} = 0.5$. Add noise $\epsilon = 0.05$ (after truncation), and the final target action is $a' = 0.55$.

  Enter $a'$ into two target Critic networks:

  Assume that Critic 1 predicts the Q value to be $105$, and Critic 2 predicts the Q value to be $95$.

TD3 takes the minimum value to suppress overestimation: $\min(105, 95) = 95$.

  Final TD target $y = r + \gamma \times 95$.

**SAC (Soft Actor-Critic)**

- **Principle**: A reinforcement learning algorithm for the maximum entropy of alternative policies, which encourages exploration.

- **Core formula (the entropy term in the objective function)**:

  $$J(\pi) = \mathbb{E} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

- **Numerical calculation example**:

Assuming there are two discrete possible actions in the current state, the policy output probability distribution is $[0.9, 0.1]$.

  Computing information entropy: $\mathcal{H} = -(0.9 \ln 0.9 + 0.1 \ln 0.1) \approx 0.325$.

If the temperature coefficient is $\alpha = 0.2$, the "soft reward" additional item for this step is $0.2 \times 0.325 = 0.065$.

If the policy becomes more random $[0.5, 0.5]$, the entropy $\mathcal{H} \approx 0.693$, and the additional reward becomes $0.138$. SAC rewards exploration behavior in this way substantially.

------

### 2. Understand concepts such as GAE (Generalized Advantage Estimation) and PG (Policy Gradient).

**PG (Policy Gradient)**

- **Formula**: $\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right]$

- **Numerical calculation examples**:

  Assume a total reward of $R(\tau) = 100$ per round. The probability of a certain action $a_t$ is $\pi(a_t) = 0.2$.

$\log(0.2) \approx -1.6$. The update magnitude of the gradient will be multiplied by $100$. If the total reward for the same action in another trajectory is only $10$, the update strength becomes $10$. The network will gradually tend to increase the probability of the action in high-reward trajectories.

**GAE (Generalized Advantage Estimation)**

- **Principle**: Balance the bias and variance of TD error.

- **Core formula**:

  $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$

  $\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$

- **Numerical calculation example (hand calculation of 2-step trajectory)**:

  Let $\gamma = 0.99$ and $\lambda = 0.95$.

There are three states $s_0, s_1, s_2$ (end).

  Reward: $r_0 = 1, r_1 = 2$.

  Value network prediction: $V(s_0) = 2.0, V(s_1) = 3.0, V(s_2) = 0.0$ (end state is 0).

  1. Calculate the single-step TD error for each step ($\delta$):

     $\delta_0 = 1 + 0.99 \times 3.0 - 2.0 = 1.97$

     $\delta_1 = 2 + 0.99 \times 0.0 - 3.0 = -1.0$

  2. Reverse recursive calculation advantage ($A$):

      Last step: $A_1 = \delta_1 = -1.0$

      Second-to-last step: $A_0 = \delta_0 + \gamma \lambda A_1 = 1.97 + (0.99 \times 0.95) \times (-1.0) = 1.97 - 0.9405 = 1.0295$

**Conclusion**: The advantage value of action $a_0$ is positive (1.0295), so it should be encouraged; the advantage value of action $a_1$ is negative (-1.0), so it should be suppressed.



```Python
import torch
# 带入上述数值验证伪代码
rewards = torch.tensor([1.0, 2.0])
values = torch.tensor([2.0, 3.0])
next_value = 0.0
dones = torch.tensor([0.0, 1.0])
gamma, lam = 0.99, 0.95
# (循环逻辑见原始回答，计算结果与手算完全一致)



```

------

## 1. Macro Perspective: The “Coach and Athlete” in Reinforcement Learning

We can imagine the entire process as a process where a **coach (GAE/value network)** guides **athletes (policy network)**.

### Policy Gradient (PG) —— Athlete Training

The core logic of PG is very simple: **"If you win this round, the probability of all my actions in this round will increase; if you lose, it will decrease."**

- It doesn't care whether the action itself is good or not; it only cares about the **result**.
- **Pain point**: If a game has 1,000 steps, the 5th step is executed brilliantly, but the 999th step results in a mistake and a loss. The PG will punish that brilliant move in the 5th step as well because the outcome is "loss". This is **high variance**.

### Generalized Advantage Estimation (GAE) —— Coach's Review

GAE is designed to address the above pain points. It no longer uses "final result" to evaluate each step, but instead assesses based on **"how much better this step is than I expected."**

- It combines “current immediate rewards” and “predictions for the future”.
- It is like a rational coach, telling the athlete: “Although you lost in the end, your manipulation in step 5 actually increased your winning rate from 20% to 60%—that extra point!”

------

## 2. Complete Pipeline (Iteration Process)

In actual PPO or A2C algorithms, data flows as follows:

### Step 1: Interaction (Sampling)

The athlete (policy network $\pi_\theta$) plays $N$ steps in the environment.

- **Output**: A set of data $\{s_t, a_t, r_t, s_{t+1}\}$ is obtained (state, action, reward, next state).
- **State**: At this point, the network parameters $\theta$ are fixed and not updated.

### Step 2: Evaluation (Scoring - GAE Takes Effect)

Now we need to score each action in this $N$ step with $a_t$. This score is the **advantage function $A_t$**.

1. **Value estimation**: Input $s_t$ into the "value network $V$", which will predict how many points each state value is worth.
2. **Calculate TD error ($\delta$)**: Check if the actual reward received $r_t$ plus the expectation for the future exceeds the current expectation.
3. **GAE smoothing**: Using the $\lambda$ parameter, add up the errors for now, one second later, and two seconds later with weighting.
   - After this step, each action $a_t$ will have a precise **$\hat{A}_t$** (positive numbers indicate better than expected, negative numbers indicate worse than expected).

### Step 3: Learning (Update - PG in action)

With the score $\hat{A}_t$, we need to go back and modify the muscle memory of the athlete (policy network).

1. **Calculate gradient**: Find the probability $\log \pi(a_t|s_t)$ of selecting action $a_t$ under state $s_t$.

2. **Weighted update**:

   $\text{更新量} = \nabla_\theta \log \pi(a_t|s_t) \times \hat{A}_t$

  - If $\hat{A}_t$ is positive (good ball!), the gradient direction will increase the probability of that action.
  - If $\hat{A}_t$ is negative (bad ball!), the gradient direction will decrease the probability of that action.

3. **Gradient ascent**: Update $\theta$, making the network smarter.

### Step 4: Loop

Use the updated "athlete" to play again, and repeat step 1.

------

## 3. Why Must GAE Be Used? (Deep Intuition)

What would happen if we only used PG without adding GAE?

- **PG (REINFORCE) without GAE**:

Use $R(\tau)$ (the total score of the whole game) to multiply by each action.

- *Disadvantages*: The updates are extremely slow and unstable. It's like taking a 3-hour exam, and at the end you only get a total score—you have no idea which questions were correct and which were wrong.

- **With the GAE PG**:

  Multiply each action by $\hat{A}_t$.

- *Advantages*: It assigns a "small fraction" to each question (each action). The $\lambda$ parameter acts like a control valve:
    - When $\lambda=0$, it only considers the next second, which is very forward-looking but potentially short-sighted.
    - When $\lambda=1$, it looks at the overall situation, which is very fair but may be disrupted by random factors (noise).
    - GAE takes an intermediate value, achieving a **perfect balance between bias and variance**.

------

## Summary: How was your calculation process integrated?

1. **$V(s)$ network** provides the expected "baseline line".
2. **$\delta_t$ (TD Error)** calculates the "gap between reality and ideal".
3. **GAE (Advantage)** uses $\gamma \lambda$ recursion to connect these gaps into a line, determining the nature of the action (good or bad).
4. **PG (Policy Gradient)** uses this qualitative result to fine-tune the weights in the neural network.





### 3. Be able to clearly explain the design concept of the reward function, as well as how to handle the Sim-to-Real gap.

**Design Concept of Reward Function**

- **Formula**:

  $$R_t = \omega_{task} \mathbb{I}(success) + \omega_{shape} \exp(-\alpha || p_{ee} - p_{obj} ||_2) - \omega_{pen} || \tau_t ||^2_2$$

- **Numerical calculation example**:

  Assume the grasping task weight $\omega_{task} = 100$, distance guidance $\omega_{shape} = 5$, smoothness parameter $\alpha = 10$, and penalty weight $\omega_{pen} = 0.1$.

  Current robotic arm end coordinate $p_{ee} = (0.5, 0, 0.5)$, target object coordinate $p_{obj} = (0.5, 0, 0.4)$.

  The square of the torque output two norm $||\tau_t||^2_2 = 4.0$.

  1. Calculate the distance: $|| p_{ee} - p_{obj} ||_2 = \sqrt{0 + 0 + (0.5-0.4)^2} = 0.1$ meters.

  2. Calculate the formation reward: $5 \times \exp(-10 \times 0.1) = 5 \times \exp(-1) \approx 5 \times 0.367 = 1.835$.

  3. Calculate the penalty: $0.1 \times 4.0 = 0.4$.

4. No target caught, $\mathbb{I}(success) = 0$.

Current step total reward: $R_t = 0 + 1.835 - 0.4 = 1.435$.

**Handle the Sim-to-Real Gap (Example of Domain Randomization Values)**

In the simulation, we set the mass of the robotic arm links to a normal distribution. The real mass is 2.0 kg, and in the simulation, we define the sampling mass per round $m \sim \mathcal{N}(2.0, 0.2^2)$. This means that during training, the RL network must learn a control policy that works within the range of 1.6 kg to 2.4 kg, so that when deployed in the physical world, it can ignore the error in the motor friction model.

------

### 4. Familiarize yourself with the architecture and concepts of mainstream models such as Open-VLA and RT-2.

**RT-2 (Robotic Transformer 2)**

- **Core idea**: Discretize the actions as text words (Tokenization) and feed them directly to the large model for prediction.

- **Numerical calculation example (action tokenization)**:

  Formula: $a_{token} = \lfloor \frac{a - a_{min}}{a_{max} - a_{min}} \times 255 \rfloor$

  Assume that we need to predict the opening and closing degree of the gripper $a$. The real physical value range is $[-1.0, 1.0]$ (-1 for fully closed, 1 for fully open).

In the current sample data, the gripper opening and closing value is $a = 0.12$.

   Substitute into the formula:

  $a_{token} = \lfloor \frac{0.12 - (-1.0)}{1.0 - (-1.0)} \times 255 \rfloor = \lfloor \frac{1.12}{2.0} \times 255 \rfloor = \lfloor 0.56 \times 255 \rfloor = \lfloor 142.8 \rfloor = 142$。

At this point, the physical action is mapped to the 142nd special token in the large model dictionary (e.g., `<action_142>`).

**Open-VLA**

- Based on 7B parameters. If using FP16 precision inference, the weights themselves require approximately 14GB of VRAM. Plus the KV Cache, a 24GB RTX 3090 or 4090 is usually needed for smooth single-step inference.





------

### 5. Understand the principles of fine-tuning methods such as LoRA.

**LoRA (Low-Rank Adaptation)**

- **Formula**: $h = W_0 x + B A x$

- **Numerical calculation example (simple manual calculation of matrix)**:

  Assume the weights of a linear layer in the pre-trained model $W_0 \in \mathbb{R}^{2 \times 2}$. It is frozen due to being too large. Input feature $x = [1, 2]^T$.

We set the Rank $r = 1$.

Initialize the low-rank matrix $A \in \mathbb{R}^{1 \times 2}$ to a random Gaussian distribution, assuming the current $A = [1, 1]$.

Initialize the low-rank matrix $B \in \mathbb{R}^{2 \times 1}$ to all zeros, but assume $B = [1, -1]^T$ after several steps of training.

  Set the scaling factor to $\text{scaling} = \alpha / r = 16 / 1 = 16$.

  Bypass calculation process:

  1. $A x = [1, 1] \times [1, 2]^T = 1 \times 1 + 1 \times 2 = 3$ (dimensional reduction to r=1)

  2. $B (Ax) = [1, -1]^T \times 3 = [3, -3]^T$ (return to level 2)

  3. Multiply by the scaling factor: $16 \times [3, -3]^T = [48, -48]^T$.

The final output is based on the output of the original pre-trained model, which contains $W_0 x$, plus the $\Delta = [48, -48]^T$ introduced by the fine-tuning.

     **Parameter count comparison**: Originally, updating all parameters for fine-tuning would require modifying $2 \times 2 = 4$ parameters; using $r=1$ LoRA only requires updating $1 \times 2 + 2 \times 1 = 4$ parameters. However, in large models (such as $4096 \times 4096$), the parameter count of LoRA ($r=8$) is only $0.4\%$ of the total parameter count!

------

### 6. Ability to analyze the advantages and disadvantages of VLA compared to other methods (such as pure reinforcement learning), and consider its feasibility in industry applications.

- **Numerical Comparative Analysis**:
  - **Success Rate and Generalization**: In a task of “grasping cups of any color at a table,” pure RL may see its success rate plummet from 95% to 5% when dealing with unseen red cups; whereas VLA, due to its semantic understanding similar to VLM, can still maintain a success rate of over 85%.
  - **Control Frequency**: Pure RL networks are very small (such as MLP), with inference latency within 1ms, enabling 500Hz underlying force control; while a 7B-parameter VLA (even when quantized to INT8) requires 80ms - 200ms for a single forward inference, and can only perform advanced action planning at 5Hz - 10Hz.
- **Industrial Implementation**: Therefore, the industry adopts a two-layer architecture of **upper-level VLA (2Hz) + lower-level MPC/RL (200Hz)**.

------

### 7. Understand basic concepts such as inverse kinematics (IK), Jacobian matrix, PID control, and MPC.

**Forward and Reverse Kinematics (using a 2-degree-of-freedom robotic arm in a 2D plane)**

- **Numerical examples**:

  The lengths of the two connecting rods are $L_1 = 1.0$ meters and $L_2 = 1.0$ meters.

  **Forward Kinematics (FK)**: Given joint angles $\theta_1 = 90^\circ (\pi/2)$, $\theta_2 = 0^\circ$.

  End point coordinate $x = L_1 \cos(\theta_1) + L_2 \cos(\theta_1+\theta_2) = 0 + 0 = 0$.

  End point coordinate $y = L_1 \sin(\theta_1) + L_2 \sin(\theta_1+\theta_2) = 1 + 1 = 2$.

  Conclusion: The end is at $(0, 2)$.

**Inverse Kinematics (IK)**: Given the target point at $(0, 2)$, it can be deduced that $\theta_1 = 90^\circ, \theta_2 = 0^\circ$ (valid when no singular point exists).

**Jacobi Matrix**

- **Numerical example**: $\dot{X} = J(\theta) \dot{\theta}$

Assume the currently calculated $J(\theta) = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$. We want the end effector to translate in the positive x-direction at a linear velocity of $\dot{X} = [0.1, 0]^T$ (m/s).

  Solve the joint velocity $\dot{\theta} = J^{-1} \dot{X} = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} \begin{bmatrix} 0.1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ -0.1 \end{bmatrix}$ (rad/s).

That is, joint 1 remains stationary, while joint 2 reverses at a speed of 0.1 rad/s.

**PID Control**

- **Formula**: $u(t) = K_p e(t) + K_i \int e(\tau)d\tau + K_d \dot{e}(t)$

- **Numerical calculation example**:

  Target position: 10.0, current real position: 8.0. Current error: $e(t) = 2.0$.

  Cumulative historical error integral $\int e = 5.0$.

  The error is decreasing, with a rate of change $\dot{e}(t) = -0.5$.

  Set the parameter $K_p = 2.0, K_i = 0.1, K_d = 0.5$.

  Motor output torque $u(t) = (2.0 \times 2.0) + (0.1 \times 5.0) + (0.5 \times -0.5) = 4.0 + 0.5 - 0.25 = 4.25$ Nm.

------

### 8. Understand the principles and applications of common sensors (such as Realsense depth cameras).

**Realsense Depth Camera (depth disparity calculation)**

- **Formula**: $Z = \frac{f \cdot B}{d}$

- **Numerical calculation examples**:

Assume the infrared lens focal length of the camera is $f = 600$ pixels.

The physical baseline distance between the left and right cameras is $B = 0.05$ meters (5 centimeters).

A point on the target object has a horizontal coordinate of 300 in the left eye image and 280 in the right eye image. The parallax is $d = 300 - 280 = 20$ pixels.

  Calculation depth distance: $Z = \frac{600 \times 0.05}{20} = \frac{30}{20} = 1.5$ meters.

The camera can output the depth of this pixel point as 1.5m.

------

### 9. "Why is the reward function set up this way in your project? What other approaches have you tried?"

*(Numerical example integrating potential energy calculation)*

"In addition to sparse rewards, I used Potential-based Reward Shaping:

$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

Among them, $\Phi(s)$ is a kind of scaling factor for the inverse of the distance between the end and the object.

**Specific calculation**: Assuming the distance from $d$, we define the potential energy function $\Phi(s) = \frac{1}{d + 0.1}$. $\gamma = 0.99$.

In status $s$ (previous step), 0 km from $d=0.4$, potential energy $\Phi(s) = \frac{1}{0.5} = 2.0$.

In state $s'$ (after the action is performed), the robotic arm approaches the object, reducing the distance to $d=0.15$ meters, with potential energy $\Phi(s') = \frac{1}{0.25} = 4.0$.

The guidance reward obtained from this step is: $F = 0.99 \times 4.0 - 2.0 = 3.96 - 2.0 = 1.96$.

The positive reward of 1.96 effectively encourages the robotic arm to approach the object, and mathematically ensures that the 'looping' flaw (Reward Loop) that leads to local optimization is avoided.

------

### 10. "What information is included in your observation space? Why was it designed this way?"

*(Example of values integrated into dimension calculation)*

“My observation space strictly controls the dimension size to optimize sample efficiency. The specific concatenation is as follows:

1. **Body information**: The joint angles (6 dimensions) and angular velocity (6 dimensions) of the 6-degree-of-freedom robotic arm = 12 dimensions.

2. **End pose**: Position $(x,y,z)$, 3D position + quaternion pose 4D + gripper opening 1D = 8D.

3. **Visual features**: Images from the hand-eye camera $84 \times 84 \times 3$. If flattened directly without dimensionality reduction, it has a dimension of 21168, which is highly overfitting. I extract features using the frozen pre-trained ResNet-18, and output $z_{img}$ with a dimension of 512.

**Total dimension**: The observation vector $O_t$ fed into the RL Actor network has a dimension of $12 + 8 + 512 = 532$, and is a 1D tensor. This design retains the Markov property of the environment while compressing the state space to a minimal level.”

------

### 11. "Please explain the multi-head attention mechanism of Transformer."

**Example of Numerical Calculation for Self-Attention Mechanism**

- **Formula**: $\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) V$

- Assume that there are only two tokens in the input (such as "red" and "cup"). The $Q, K, V$ dimension after projection is set to a small positive value, such as $d_k = 4$.

  Let $Q = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix}$ (a Query with two lines representing two words).

  Let $K = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix}$.

  1. Calculate the inner product $Q K^T = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$.

  2. Divide by $\sqrt{d_k} = \sqrt{4} = 2$, resulting in the attention score matrix: $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$.

  3. Softmax (simplified for demonstration since non-diagonal elements are 0), with the results heavily concentrated on itself. Finally, multiply by $V$ to extract the information.

The multi-head mechanism involves performing the above process $h$ times in several parallel spaces, and then concatenating them together. This ensures that the model can extract features from different sub-spaces such as "color" and "shape" when observing the same word.

------

### 12. "What is the difference between Diffusion Models (DM) and Autoregressive Models (AR) in VLA?"

- **Autoregressive (AR) numerical example**:

  Predict a motion sequence $[a_1, a_2]$.

  Step 1 prediction probability $P(a_1 | obs) = 0.9$. Put $a_1$ back into the input, and for Step 2, predict $P(a_2 | obs, a_1) = 0.8$.

The overall trajectory joint probability is $0.9 \times 0.8 = 0.72$. The issue is that once the prediction of $a_1$ has a slight error, the error will accumulate exponentially with the number of steps (Exposure Bias).

- **Diffusion Model (DM) numerical example**:

  Predict the entire trajectory. The task of the network $\epsilon_\theta$ is to predict the current added noise.

Assume that the Ground Truth action is combined with the Gaussian noise from real sampling $\epsilon = [0.1, -0.2]^T$.

The network forward inference predicts noise $\hat{\epsilon} = [0.08, -0.22]^T$.

  Calculate the MSE Loss of DDPM: $|| \epsilon - \hat{\epsilon} ||_2^2 = (0.1 - 0.08)^2 + (-0.2 - (-0.22))^2 = 0.0004 + 0.0004 = 0.0008$.

Each DM directly denoises and outputs a Chunk containing multiple steps (with a length of 16 steps). The trajectory is naturally smooth and consistent, completely solving the cascading error problem in AR.

------

### 13. "Why did you choose our lab/this direction?"

"I chose the field of embodied AI because it enables closed loop intervention in the physical world.

As for choosing your lab:

1. **Technical Fit**: I have read your team’s paper on diffusion models for manipulation of dexterous hand systems with long views. I noticed that you set the Chunk Size for action prediction at 32, which increased the grasping success rate by 15%. In my previous project, I also tried adjusting the window length of Action Chunking and found that 16 to 32 is a sweet spot—this really resonated with the engineering details you have implemented.
2. **Computational Power and Hardware Implementation**: You have 4 real robot platforms equipped with multi-finger dexterous hands, along with an A100 cluster in a private cloud. Compared to my current setup, which can only run 500,000 steps of reinforcement learning simulation using a single RTX 3090 card, your platform allows me to perform 10 million steps within the same cycle, significantly accelerating the closed loop from simulation to reality.





### 1. What do DDPG/PPO/TD3/SAC do? What are the inputs and outputs?

They are a set of general “brain learning rules” (continuous control framework). You can use them to train the robotic arm to grasping, train robot dogs for parkour, and even train an autonomous car to steer.

The core structure of these four algorithms typically consists of two neural networks: **Actor (responsible for outputting actions)** and **Critic (responsible for scoring)**.

#### 1. Model input (Observation/State)

When performing tasks, what is input to the model (mainly the Actor network) is usually a **one-dimensional vector**, representing the state currently perceived by the robot $s_t$. This vector can contain various types of information:

- **Physical state (most common)**: For example, the angles and angular velocity of the 6 joints of the robotic arm, and the $(x,y,z)$ coordinates of the gripper at the end. If it is a 12-dimensional array, the network input layer size is 12.
- **Image input (Visual RL)**: If you want the model to “see” images, you usually do not feed tens of thousands of pixels directly into these reinforcement learning algorithms. The standard approach is to first use a CNN (such as ResNet) to compress the RGB image into a fixed-length feature vector (e.g., 256 dimensions), and then combine it with the physical state above before feeding it into the algorithm.
- **Language commands**: For popular language condition control, the method is similar. The phrase “pick up a red apple” is converted into a series of numerical vectors using a text encoder like CLIP, and this vector is also combined with the input state.

#### 2. Model Output (Action)

The output of the Actor network $a_t$ is the continuous action command to be executed in the next step.

- **Specific form**: For example, a 6-dimensional vector representing the target torque of the 6 robotic arms' motors, or the desired joint angular velocity.
- **Output flow**: This output is directly sent to the physics engine (such as the Isaac Gym simulator) or real hardware controllers for execution.

#### 3. Intermediate Iteration Process (The RL Loop)

Taking the robotic arm grasping objects as an example, the entire closed loop iteration process is as follows:

1. **Observe the environment**: The sensor obtains the current state $s_t$ (e.g., the robotic arm is at point A, and the target is at point B).
2. **Network decision**: Feed $s_t$ into the Actor network, and after calculation, the Actor outputs an action $a_t$ (e.g., each motor rotates a certain angle).
3. **Environmental feedback**: After the robot executes action $a_t$, it moves to a new state $s_{t+1}$. At this point, the environment (or your code) provides a reward $r_t$ (e.g., +1 point for approaching the target; -10 points for hitting a wall).
4. **Collect data**: Store the experience from this step $(s_t, a_t, r_t, s_{t+1})$ in an “experience replay pool” (a large data table).
5. **Update the network**:
   - **Critic update**: The Critic retrieves data from the pool to learn how to more accurately predict “how many points can be earned by taking a certain action under a specific state in the future.” (TD target values will be discussed soon).
   - **Actor update**: The Actor adjusts its parameters based on the Critic’s scores, ensuring that when similar situations occur next time, it will likely output an action with higher scores.
6. **Loop**: Return to step 1 and repeat this process tens of thousands to millions of times until the model learns the optimal policy.

------

### II. What is "TD Target"?

TD (Temporal Difference) is one of the most sophisticated concepts in reinforcement learning. **You can consider the TD target value as: a “most recent and accurate prediction” of the total future reward, combined with the “actual reward” that just occurred.**

Let me give you an example from daily life so you can understand immediately:

Suppose you are driving from Beijing to Shanghai. Before leaving, your brain (Critic network) estimates that it will take **12 hours** to get there.

When driving to Tianjin, you spent **2 hours** (which is equivalent to the real reward for the current step $r_t$). At this point, you use navigation to check again: it takes **9 hours** from Tianjin to Shanghai (which is the model's estimate for the future state $s_{t+1}$).

So, considering the already completed part of the journey, your **latest total time estimate** for the "Beijing to Shanghai" trip becomes:

2 hours (actual time spent) + 9 hours (latest forecast remaining time) = **11 hours**.

This **11 hours** is the **TD target value**!

Your brain will use these 11 hours to correct the 12 hours of estimates made before departure. This method of continuously refining old predictions with new information by "taking one step at a time" is the core of TD learning.

As it stands in the formula, it is like this:

$$y_t = r_t + \gamma Q(s_{t+1}, a_{t+1})$$

- $r_t$: The actual reward obtained from the just executed action (the 2 hours spent).
- $Q(s_{t+1}, a_{t+1})$: The Critic network's prediction of how many points can be earned in the future after reaching a new state (how long it will take to travel from Tianjin to Shanghai).
- $\gamma$: Discount factor (usually 0.99, indicating that the future estimate is discounted, slightly reducing the weight).
- $y_t$: The calculated **TD target value**.

During training, the task of the Critic network is to strive to make its predicted values $Q(s_t, a_t)$ approach the TD target value $y_t$, thereby becoming more accurate.





Macroscopically, DDPG/PPO/TD3/SAC all operate within the same "actuator-critic" framework of "observation-action-reward-update". However, microscopically, in order to address various "failure" issues during training, scientists have made different modifications to the **network output format**, **Q-value calculation method**, and **reward mechanism**.

The core difference lies in these micro-level aspects. We can first take PPO, and then compare the three "related algorithms"—DDPG, TD3, and SAC—side by side.

### 1. The difference in fundamental camps: How are the data used?

- **PPO (Same policy, On-policy):** It is a typical example of “learning by doing”. The Actor personally collects a batch of data in the environment, and after updating the network, this data is **discarded**. The next update requires using the latest Actor to collect new data.
- **DDPG / TD3 / SAC (Different policy, Off-policy):** They have a large “Replay Buffer”. The data collected by the Actor is stored there, and the Critic can repeatedly retrieve old data from it for learning. This approach saves costs associated with interaction with the environment (high sample efficiency).

------

### 2. Micro-surgical scalpels: What exactly did they change in Q values and policies?

These three different policy algorithms (DDPG -> TD3 -> SAC) are actually a "tragic history of filling gaps". Their subtle differences precisely answer your questions:

#### DDPG: The pioneer (but also full of flaws)

- **Action output**: The Actor directly outputs a definite value (e.g., “motor output 2.5 Newtonm”). This is called a deterministic policy.
- **Q-value calculation**: The Critic acts like a naive teacher, directly updating based on the standard TD target formula $y = r + \gamma Q(s', a')$.
- **Fatal flaw**: Since neural networks always have errors in prediction, the Critic in DDPG is extremely prone to **blind optimism (over-estimating Q-values)**. It will assign very high scores to actions that are actually poor, causing the Actor to fail to learn properly.

#### TD3: The "Powerful Patch Version" of DDPG

To address the "blind optimism" of DDPG, TD3 makes extremely sophisticated modifications in the micro-level Q value calculation:

- **Minimize the dual Q network (key!)**: It directly uses two independent Critic networks. When calculating future Q value predictions, both networks score separately, and then the **smaller of the two is forced to be selected**.
  - *Real-life example*: You are evaluating the future value of a house and hire two appraisers. One says it is worth 5 million, and the other says it is worth 3 million. To be safe (to avoid being caught in an overvaluation), you make a decision based on 3 million.
- **Target policy smoothing**: When calculating the TD target value, a little random noise is intentionally added to the next action $a'$. This forces the Critic to understand: “Don’t just give high scores to a single specific action; check if the entire surrounding area is good.”
- **Delayed update**: The Critic updates two or three times before the Actor updates once. Once the Critic has calculated the scores accurately, it then guides the Actor.

#### SAC: The current king, directly changed the definition of "rewards"

SAC takes a different approach. The biggest change at the micro level is **the modification of the objective of the reward function (maximum entropy framework)**:

- **Microscopic change in rewards**: It inserts a "Entropy Bonus" directly after the traditional real reward $r$.
  - The goal becomes: maximizing $[ 任务奖励 + \alpha \times 策略的随机性 ]$.
- **Action output**: Unlike the action outputs of DDPG/TD3, which are deterministic, the Actor output of SAC is a **Gaussian distribution (mean and variance)**.
- **Effect**: This means that SAC not only requires the robot to complete the task, but also encourages the robot to **complete the task in as many random postures as possible**. This microscopic design enables SAC to exhibit strong robustness when facing sudden disruptions in the physical environment.

------

### 3. One picture beats a thousand words: Summary table of microscopic differences

| **Policy** | **Actor Output Form**       | **Q-Value Prediction Manipulation (Critic)** | **Reward/Update Special Mechanism**                     | **Applicable Scenario Characteristics**                   |
| -------- | ------------------------ | ----------------------------- | ------------------------------------------- | ---------------------------------- |
| **PPO**  | Probability distribution (Gaussian/discrete) | Use advantage function (Advantage)      | **Clipping**, to prevent update steps from being too large | Extremely stable, simplest parameter tuning, least hassle       |
| **DDPG** | Deterministic specific values | Single Q network, direct prediction         | No special mechanism, directly compute TD error                  | Prone to crashes, rarely used alone now           |
| **TD3**  | Deterministic specific values | **Minimum of two Q networks**         | **Noisy target action**, **delaying Actor update**        | Fast training speed, relatively low computational cost       |
| **SAC**  | Probability distribution (Gaussian mean/variance) | Minimum of two Q networks             | **Maximize policy entropy** (exploration bonus added to reward) | High sample utilization, strongest robustness for real robot |

By removing the complex mathematical formulas, when implemented in code, they often only involve adding small manipulations such as `torch.min(q1, q2)` or `+ alpha * entropy` to the few lines of code involving `Critic.forward()` or `Loss`, yet this results in a significant improvement in overall performance.