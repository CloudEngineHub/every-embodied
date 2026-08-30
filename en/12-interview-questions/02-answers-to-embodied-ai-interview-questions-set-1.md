# Answers to Embodied Ai Interview Questions Set 1

## 1. Since you are using a diffusion model, everyone refers to it as End-to-End. Why do you separate VLM from Diffusion? If I directly use a large multimodal model to output coordinates, where do you think the biggest bottleneck occur in a precision scenario like surgery?

Although end-to-end is the trend, in a highly precise dynamic scenario like surgery, directly using a large model to output coordinates causes a fatal **"perception-execution" delay**.

- **Decoupled logic**: VLM is responsible for low-frequency high-level semantic understanding and planning (the brain), while Diffusion handles high-frequency, continuous underlying action generation (the cerebellum). Between discrete visual updates, high-frequency policies (such as accelerating inference from single-digit Hz to dozens of Hz) are essential for continuous intra-frame trajectory interpolation, thereby truly addressing the “perception-execution” latency issue in real-world dynamic manipulation.

- **End-to-end bottleneck**: The biggest bottleneck lies in the mismatch between the control frequency and real-world dynamics, as well as the **spatial resolution disaster** caused by VLM discrete tokens. If the frequency of real-world dynamics in the environment is $f_{env}$, the inference frequency of VLM is $f_{vlm}$. When $f_{vlm} \ll f_{env}$ occurs, the dispatched action sequence $a_t$ will face severe phase lag:

  $$a_t = \pi_{vlm}(o_{t-\Delta t})$$

  Where $\Delta t$ represents the inference delay of VLM. During surgery, this leads to the real tissue state evolving from $o_{t-\Delta t}$ to $o_t$ when the robotic arm performs action $a_t$, resulting in an irreversible medical incident. After decoupling, we can execute $a_t \sim p_\theta(a_t | o_t, z_{vlm})$ at a high frequency of $f_{diff} \gg f_{env}$.

### 2. The diffusion model generates Action Chunks, but during surgery, tissue deformation occurs in real time. If your Chunk has not completed its processing yet, and the tissue has already undergone nonlinear displacement, how can your system perform closed loop correction?

Diffusion-generated Action Chunks will create blind spots in open-loop execution. In the face of the real-time nonlinear deformation of tissues during surgery, strict closed-loop correction is essential:

- **Receding Horizon Control (RHC)**: The model generates a Chunk containing the $T$-step trajectory: $\hat{A}_t = [a_t, a_{t+1}, \dots, a_{t+T-1}]$. However, we only execute the first $k$ steps ($k \ll T$), and obtain new observations $o_{t+k}$ at $t+k$ to generate a new Chunk.

- **High-frequency multi-modal interruption**: During a very short period of execution, a ultra-high-frequency torque feedback (Force/Torque Sensor) is introduced. The currently measured torque vector is defined as $F_{ext}$, and the safety threshold is $\tau_{safe}$. The controller logic is as follows:

  $$a_{exec} = \begin{cases} \hat{A}_t[i], & \text{if } \| F_{ext} \| \le \tau_{safe} \\ a_{suspend}, & \text{if } \| F_{ext} \| > \tau_{safe} \end{cases}$$

Once the $\| F_{ext} \|$ mutation exceeds the boundary (indicating a change in the tissue shape as expected), the underlying controller immediately cuts off the current Chunk.

### 3. Provide an example from surgery using a diffusion model to handle multi-peaked distributions? How does the model avoid decision hesitation when faced with two reasonable obstacle avoidance paths?

- **Surgical examples**: Suppose the robotic arm encounters a blood vessel in front while dissecting tissue. It can bypass it from the left side (trajectory $\mu_{left}$) or from the right side (trajectory $\mu_{right}$). The actual motion distribution is a bimodal distribution:

  $$p(a|o) = w_1 \mathcal{N}(\mu_{left}, \Sigma) + w_2 \mathcal{N}(\mu_{right}, \Sigma)$$

If the traditional MSE Loss $\mathcal{L} = \| a - \hat{a} \|_2^2$ is used, the model's predicted action $\hat{a}$ will converge to the mathematical expectation $\mathbb{E}[a|o] \approx \frac{\mu_{left} + \mu_{right}}{2}$, resulting in the robotic arm directly hitting the blood vessels in the middle. The denoising process of diffusion learns the score function $\nabla_a \log p(a|o)$, which can accurately fit the true distribution, and through Langevin dynamics sampling, it collapses to “absolute left” or “absolute right”.

- **Avoid decision hesitation**: Introduce **Action EMA (Action EMA)**. Let the actual action executed at $t$ be $a_t^{(final)}$, which is a weighted combination of the current generated action and historical actions:

  $$a_t^{(final)} = \alpha a_t^{(new)} + (1 - \alpha) a_{t-1}^{(final)}$$

  Among them, $\alpha \in (0, 1)$ is the smoothing coefficient. By assigning weights to historical decisions, the left-right symmetric fluctuations caused by sampling randomness are eliminated.

### 4. Operations take several hours, and the Transformer’s KV Cache will explode over time. Although Mamba has been used, the linear compression of Mamba is inherently lossy. How can we ensure that it accurately remembers the logic corresponding to the past hour after recording it, without generating hallucinations?

Mamba's state compression, by fixing the dimension of the hidden state $h_t \in \mathbb{R}^n$, is essentially a lossy compression in a Markovian manner. A way to prevent confusion in the logic a few hours ago:

- **State Reset (State Reset)**: The surgery has strict stages. When switching between stages, the underlying physical hidden state $h_t$ is multiplied by a decay gate $g_t \to 0$, i.e., $h_t \leftarrow g_t \odot h_{t-1}$, to clear the unnecessary high-frequency historical noise.

- **Hierarchical semantic anchors**: Do not let Mamba remember all the underlying action sequences $a_{0:t}$. Convert the past "big logic" into a set of discrete Milestones through VLM $S = \{s_1, s_2, \dots, s_k\}$ (such as "organizational separation completed"). These semantic information are regularly injected as high-dimensional condition vectors $c_{semantic}$:

  $$h_t = \bar{A} h_{t-1} + \bar{B} x_t + W c_{semantic}$$

(($W$ is the projection matrix)), to maintain precise long-term logic.

### 5. We just mentioned Transformer and Mamba. Can you explain the difference between these two structures?

- **Transformer**: The core is the Self-Attention mechanism, and the calculation formula is:

  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

  Here, $Q, K, V \in \mathbb{R}^{N \times d_k}$ ($N$ is the sequence length, and $d_k$ is the feature dimension). It needs to provide a global overview of all tokens and capture explicit global spatial and temporal dependencies.

- **Mamba**: Based on the Selective State Space Model (SSM), it is the discretization of a continuous-time dynamical system. Essentially, it is a linear RNN with a data-dependent gating mechanism. The hidden state $h_t$ is recursively computed sequentially over time steps:

  $$h'(t) = A h(t) + B x(t)$$

It does not need to go back to calculate all the tokens from history at the current moment.

### 6. Explain why Transformer is N squared, and Mamba is linear.

- **Transformer ($O(N^2)$)**: When computing the attention score matrix $S = QK^T$, the dimension of $Q$ is $N \times d_k$, and that of $K^T$ is $d_k \times N$. The computational complexity of matrix multiplication is $\mathcal{O}(N^2 \cdot d_k)$. Since each token must compute a dot product with all other $N-1$ tokens in the sequence, memory and computational costs explode quadratically with the sequence length $N$.
- **Mamba ($O(N)$)**: Mamba processes sequences using a fixed-dimensional hidden state $h_t \in \mathbb{R}^n$. The discretized state update is $h_t = \bar{A} h_{t-1} + \bar{B} x_t$. For each new token $x_t$ in the sequence, only vector and matrix multiplication is required, and the single-step update complexity remains constant at $\mathcal{O}(n^2)$ or lower (after diagonalization). For a sequence of total length $N$, the overall complexity is $\mathcal{O}(N \cdot n)$, which grows linearly.

### 7. Have you conducted quantitative experiments to see if there are performance differences between Mamba and other methods, specifically in terms of accuracy?

In real embodied AI experiments:

- **Local precision loss**: For short sequence actions that require high-frequency spatial fine-tuning, the absolute success rate of Mamba may be slightly lower than that of Transformer (usually around 1%-3%). Mathematically, this is a **Information Bottleneck** problem. Let the information entropy of the input sequence be $H(X)$, and the capacity of the fixed-dimensional hidden state be $C(h_t)$. When $H(X) > C(h_t)$ occurs, high-frequency spatial features are inevitably truncated and lost.
- **Advantages of long sequences**: In long-term control $N \to \infty$, the softmax distribution of Transformer will experience Attention Dilution due to the large denominator $\sum_{j=1}^N \exp(q_i \cdot k_j)$, resulting in performance degradation; whereas Mamba can maintain a very high inference frame rate and smoothness.

### 8. You mentioned that there is also a high-level planning model here. What is this model made of?

Usually a multimodal large language model (VLM), such as a model fine-tuned based on Qwen-VL, LLaVA, or Gemini architectures.

Expressed in function form:

$$z_t = f_{vlm}(I_{img}, s_{proprio}, P_{text} ; \theta_{vlm})$$

Among them, $I_{img}$ is the current surgical view image, $s_{proprio}$ is the status of the robotic arm, $P_{text}$ is the text instructions, and $\theta_{vlm}$ is the model weights. The output $z_t$ represents the next discrete sub-task instruction (such as `[GRASP, NEEDLE]`).

### 9. How many historical images should be input? If we only create the high-level planning model, only three to five images are needed. Can it remember all the previous manipulations? Why?

Because the high-level tasks strictly conform to the **Markov Decision Process (MDP)** at the macro-semantic level.

In MDP, the future state depends only on the current state and action, that is:

$$P(S_{t+1} | S_t, A_t, \dots, S_0, A_0) = P(S_{t+1} | S_t, A_t)$$

The high-level model does not need to know the microscopic trajectory 10 minutes ago. It only needs to calculate the first derivative of the current environment (i.e., dynamic trends) using 3-5 images (current frame + a few historical frames).

$$\dot{O}_t \approx \frac{O_t - O_{t-k}}{k \cdot \Delta t}$$

The "current visual state" combined with the "small number of historical frames" representing the trend, along with the final goal given by the text instruction, is sufficient to provide a complete semantic context for solving the MDP.

### 10. How is the training of this entire high-level model conducted?

Training is divided into three stages:

1. **Large-scale pre-training**: Establish basic image-text representations.

2. **SFT (Supervised Fine-Tuning)**: Use a large number of doctor surgery videos to extract key frames and actions and assign labels. Minimize the negative log-likelihood loss:

   $$\mathcal{L}_{SFT} = - \sum_{t=1}^T \log p_{\theta}(a_t^* | O_{\le t}, P_{text})$$

    Among them, $a_t^*$ represents the doctor's actual manipulation instructions.

3. **Alignment training (RLHF/DPO)**: Professional doctors provide preference feedback. Taking DPO (Direct Preference Optimization) as an example, the policy is optimized for safe and dangerous operations $\pi_\theta$:

   $$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

$y_w$ is a safe manipulation, while $y_l$ is a dangerous violation operation.

### 11. In Mamba's Selective Scan, how is it defined what should be forgotten and what should be remembered? What if the system accidentally forgets a fleeting tiny point?

- **What to forget, what to remember**: The Mamba time step parameter $\Delta_t$ is a linear projection function of the input $x_t$:

  $$\Delta_t = \text{softplus}(\text{Linear}(x_t))$$

  When the network learns that $x_t$ is a useless background, $\Delta_t \to 0$. According to the discretization formula $\bar{A} = \exp(\Delta_t A)$, at this time $\bar{A} \to I$ and $\bar{B} \to 0$, resulting in the hidden state update becoming $h_t \approx I h_{t-1} + 0 \cdot x_t = h_{t-1}$. This is the concept of "forgetting the present and remembering the past". Conversely, if $\Delta_t$ becomes larger, the current new features are written into it.

- **What to do about the lost micro-points**: This is a major structural flaw. A **hybrid architecture** is introduced in engineering: a high-frequency anomaly detection branch $f_{anomaly}(x_t)$ runs in parallel outside Mamba. Its features are directly connected via Residual Connection to the output:

  $$y_t = C h_t + \text{ReLU}(f_{anomaly}(x_t))$$

  Ensure the fleeting key points (such as bleeding points) directly bypass the Mamba state compression.

### 12. If the failure is caused by occlusion, can blind surgery be performed? Or does the system have to sit idle and wait for a doctor?

In real medical robots, **pure visual blind operations are absolutely not allowed**.

If severe occlusion occurs, the confidence score of the visual encoder will decrease sharply. Let the confidence of visual action prediction be $c_{vis} = \max_a P(a | o_{occluded})$, and the confidence threshold be $\gamma$.

When $c_{vis} < \gamma$ occurs, the system must downgrade to a six-dimensional torque feedback $F_{ext}$ (force control mode) that relies on the end effector. If $\exists \|F_{ext}\| > \tau_{safe}$ occurs under this condition, the safety shutdown policy $a_{suspend}$ must be executed immediately (e.g., retracting the device a short safe distance along its current axis and locking the joint), with an alert issued for the doctor to take over (Supervised Autonomy).

### 13. The tissue mechanical parameters (such as Young's modulus) in the simulation environment vary greatly compared to live pigs. How did you align them?

- **Domain Randomization**: In simulation, the mechanical parameter set of the tissue $\Phi$ (such as Young’s modulus $E$, Poisson’s ratio $\nu$) is set to a broad uniform distribution $\Phi_{sim} \sim \mathcal{U}(\Phi_{min}, \Phi_{max})$. The forced control policy $\pi_\theta$ remains robust to unknown mechanical parameters during training.

- **Online System ID + Residual Policy**: During live operation, by using real-time manipulation $a_t$ and feedback force $F_t$, the implicit mechanical characteristics of the current tissue $z_{phy} = \phi(a_{t-k:t}, F_{t-k:t})$ are inferred online. Then, a residual policy network $\pi_{res}$ is used to compensate for the Sim-to-Real gap:

  $$a_{real} = \pi_{sim}(o_t) + \pi_{res}(o_t, z_{phy}, F_t)$$

### 14. Everyone knows that Mamba's state compression is a linear recursion. If, at the 2-hour mark of the surgery, an “extensive tissue collapse” (OOD scenario) that has never appeared in the training set occurs, will Mamba’s Hidden State develop cognitive inertia due to “excessive historical memory,” causing it to force-fit the current abnormal state into a normal trajectory from past times?

Yes, cognitive inertia occurs. Because $h_{t-1}$ has accumulated a strong normal prior, while the input gatekeeper $\Delta_t$ cannot produce a reasonable response to OOD features $x_{ood}$, resulting in the model passively "following the history".

**Response Strategy**: Introduce a system-level **Uncertainty Estimation**. The reconstruction error of the visual Encoder can be monitored:

$$E_{recon} = \| o_t - \text{Decoder}(\text{Encoder}(o_t)) \|_2^2$$

If $E_{recon} > \epsilon_{threshold}$, it indicates that the current image manifold is not within the training set (OOD), and a hardware-level fuse is triggered directly, rather than relying on the internal weights of Mamba to force fitting.

### 15. What is the approximate true success rate of clinical validation on animals?

At the current academic level (such as the SOTA papers at ICRA/IROS), the success rate of defining sub-tasks $SR$ is defined as:

$$SR = \frac{1}{N} \sum_{i=1}^N \mathbb{I} \left( d(S_{end}^{(i)}, S_{goal}) < \epsilon \right)$$

Here, $N$ is the total number of experiments, $\mathbb{I}$ is the indicator function, $d$ is the distance metric, and $\epsilon$ is the task tolerance.

For specific restricted single sub-tasks (fixed path cutting, knotting), the success rate on live pigs is typically between **70% - 85%**. For complex surgeries involving the whole process, strong dynamics, and bleeding, the rate of complete autonomy is extremely low. Currently, it remains a supervised autonomous form where "the doctor leads and the system assists".

### 16. Hand tearing: 1. What is the probability that breaking a wooden stick into three pieces will form a triangle? 2. Given a 2D grid composed of '1' (land) and '0' (water), please calculate the number of islands in the grid.

**1. The probability that breaking a wooden stick into three segments will form a triangle**

Let the total length of the stick be 1. When folded into three segments, the lengths are $x, y, z$. The following constraints must be satisfied:

$$x > 0, \quad y > 0, \quad z > 0$$

$$x + y + z = 1$$

The necessary and sufficient condition for forming a triangle is that the sum of any two sides is greater than the third side (equivalent to any side being less than 0.5):

$$x < 0.5$$

$$y < 0.5$$

$$z = 1 - x - y < 0.5 \implies x + y > 0.5$$

In the $x-y$ two-dimensional coordinate system, the total sample space is a right triangle that satisfies $x > 0, y > 0, x + y < 1$, with an area of $S_{total} = \frac{1}{2} \times 1 \times 1 = \frac{1}{2}$.

The feasible region that satisfies the triangle condition is a small internal triangle with vertex $(0.5, 0), (0, 0.5), (0.5, 0.5)$, and its area is $S_{valid} = \frac{1}{2} \times 0.5 \times 0.5 = \frac{1}{8}$.

Therefore, the probability is geometric probability:

$$P = \frac{S_{valid}}{S_{total}} = \frac{1/8}{1/2} = 25\%$$

**2. Number of Islands (LeetCode 200)**

Core assessment uses depth-first search (DFS).



```Python
def numIslands(grid):
    if not grid:
        return 0

    count = 0
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        # 边界检查及是否为水的判断
        if r < 0 or c < 0 or r >= rows or c >= cols or grid[r][c] == '0':
            return
        # 标记已访问（原地修改为'0'以节省空间）
        grid[r][c] = '0'
        # 深度优先搜索四个方向
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)

    for i in range(rows):
        for j in range(cols):
            # 找到陆地起点，发起一次 DFS 连通分量遍历
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1

    return count
```

### 17. Can you list your mathematical equation?

**1. Core equations of the Continuous State Space Model (SSM) dependent on Mamba**

$$h'(t) = A h(t) + B x(t)$$

$$y(t) = C h(t)$$

- $t$: Continuous time steps.
- $x(t) \in \mathbb{R}$: One-dimensional input signal.
- $h(t) \in \mathbb{R}^n$: $n$-dimensional implicit state vector of the continuous system.
- $A \in \mathbb{R}^{n \times n}$: State transition matrix.
- $B \in \mathbb{R}^{n \times 1}$: Input projection matrix.
- $C \in \mathbb{R}^{1 \times n}$: Output projection matrix.
- $y(t) \in \mathbb{R}$: System output signal.

**2. The Discrete Process of Mamba's Zero-Order Retainer (ZOH)**

Processing discrete tokens requires introducing a dynamic step size parameter dependent on input $\Delta_t \in \mathbb{R}$ for discretization.

The discretized state recurrence formula is:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$

$$y_t = C h_t$$

Strict derivation of discretized parameters:

$$\bar{A} = \exp(\Delta_t A)$$

$$\bar{B} = (\Delta_t A)^{-1}(\exp(\Delta_t A) - I) \cdot \Delta_t B$$

- $x_t$: Discrete input token for step $t$.
- $h_t$: Discrete implicit state for step $t$.
- $\bar{A}$: State transition matrix after discretization (determines how much history to retain).
- $\bar{B}$: Input projection matrix after discretization (determines how much new input to incorporate).
- $I$: Identity matrix.

**3. Diffusion Model Reverse Denoising Sampling Equation**

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

- $\theta$: Learnable parameters of the neural network (such as U-Net or Transformer).
- $x_t$: Noisy data at step $t$ in the diffusion process (usually a noisy action trajectory in embodied control).
- $x_{t-1}$: Data state after the denoising step.
- $\mathcal{N}$: Gaussian distribution.
- $\mu_\theta(x_t, t)$: The mean predicted by the network (guiding the denoising direction, usually obtained by predicting noise $\epsilon_\theta$ or directly predicting a noise-free action $x_0$).
- $\Sigma_\theta(x_t, t)$: Variance matrix (controlling the randomness and diversity of the generation process).