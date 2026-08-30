# Supplementary questions



1. Continuous intra-frame trajectory interpolation is essential, thereby truly addressing the "perception-execution" latency issue in real-world dynamic manipulation.



How is this difference calculated specifically?



2. What does $a_suspend$ mean? When will the fuse operate? What actions are taken after the fuse operates? Does it stay off permanently?



3. The learning score function $\nabla_a \log p(a|o)$ can accurately fit the true distribution, and it collapses through Langer-Kuhn dynamics sampling to "absolute left" or "absolute right". What is this specific formula about?



4. However, the linear compression of Mamba is essentially lossy. What does "lossy" mean?



5. Hierarchical semantic anchors: Do not let Mamba remember all the underlying action sequences $a_{0:t}$. The past "big logic" is abstracted into a set of discrete Milestones through VLM $S = \{s_1, s_2, \dots, s_k\}$ (such as "organizational separation completed"). These semantic information are periodically injected as high-dimensional condition vectors $c_{semantic}$:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t + W c_{semantic}$$

(($W$ is the projection matrix)) to maintain precise long-term logic.



How does this work specifically? Provide a detailed case explanation, a concrete example.



**Mamba ($O(N)$)**: Mamba processes sequences through a fixed-dimensional hidden state $h_t \in \mathbb{R}^n$. The discretized state is updated to $h_t = \bar{A} h_{t-1} + \bar{B} x_t$. For each new Token $x_t$ in the sequence, only vector and matrix multiplication is required, and the single-step update complexity remains at $\mathcal{O}(n^2)$ or lower (after diagonalization). For a sequence with a total length of $N$, the total complexity is $\mathcal{O}(N \cdot n)$, which increases linearly.



Why is there an O(n^2) here? How do A and B in the Mamba formula come about?



6. **Local precision loss**: For short-sequence actions that require very high-frequency spatial fine-tuning, the absolute success rate of Mamba may be slightly lower than that of Transformer (usually around 1%-3%). Mathematically, this is a **Information Bottleneck** problem. Let the information entropy of the input sequence be $H(X)$, and the capacity of the fixed-dimensional hidden state be $C(h_t)$. When $H(X) > C(h_t)$ occurs, the high-frequency spatial features are inevitably truncated and lost.

How are H and C calculated specifically here?



7. **Long sequence advantage**: In long-term span $N \to \infty$ control, the softmax distribution of Transformer will experience Attention Dilution due to the large denominator in $\sum_{j=1}^N \exp(q_i \cdot k_j)$, resulting in performance degradation; whereas Mamba can maintain a very high inference frame rate and smoothness.

Why doesn't mamba have attention dilution?



8. The "current visual state" plus the "small number of historical frames" representing trends, combined with the final goal given by text instructions, is sufficient to provide a complete semantic context for solving MDP.

Why is this so?



9. Training is divided into three stages:

1. **Large-scale pre-training**: Establish basic image-text representations.

2. **SFT (Supervised Fine-Tuning)**: Use a large number of doctor surgery videos to extract key frames and actions and assign labels. Minimize the negative log-likelihood loss:

   $$\mathcal{L}_{SFT} = - \sum_{t=1}^T \log p_{\theta}(a_t^* | O_{\le t}, P_{text})$$

    Among them, $a_t^*$ represents the doctor's actual manipulation instructions.

3. **Alignment training (RLHF/DPO)**: Professional doctors provide preference feedback. Taking DPO (Direct Preference Optimization) as an example, the manipulation is optimized for safe and dangerous operations $\pi_\theta$:

   $$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

$y_w$ is a safe manipulation, while $y_l$ is a dangerous violation operation.

Please give an example of a specific scenario.



10、$$\Delta_t = \text{softplus}(\text{Linear}(x_t))$$

What does the `softplus` parameter mean?

**What to forget, what to remember**: The Mamba time step parameter $\Delta_t$ is a linear projection function of the input $x_t$:

$$\Delta_t = \text{softplus}(\text{Linear}(x_t))$$

When the network learns that $x_t$ is a useless background, $\Delta_t \to 0$. According to the discretization formula $\bar{A} = \exp(\Delta_t A)$, at this time $\bar{A} \to I$ and $\bar{B} \to 0$, resulting in the hidden state update becoming $h_t \approx I h_{t-1} + 0 \cdot x_t = h_{t-1}$. This is the concept of "forgetting the present and remembering the past". Conversely, if $\Delta_t$ increases, the current new features are written into the model.

It is not written here how AB are calculated.





11. **What to do about the lost micro-points**: This is a major structural flaw. A **hybrid architecture** is introduced in engineering: a high-frequency anomaly detection branch $f_{anomaly}(x_t)$ runs in parallel outside Mamba. Its features are directly connected via Residual Connection to the output:

$$y_t = C h_t + \text{ReLU}(f_{anomaly}(x_t))$$

Ensure the fleeting key points (such as bleeding points) directly bypass the Mamba state compression.

What does "residual connection" mean here? What is the internal mapping of the $f_{anomaly}(x_t)$ function?



12. What is the failure caused by occlusion?

In real medical robots, **pure visual blind operations are absolutely not allowed**.

If severe occlusion occurs, the confidence score of the visual encoder will decrease sharply. Let the confidence of visual action prediction be $c_{vis} = \max_a P(a | o_{occluded})$, and the confidence threshold be $\gamma$.

When $c_{vis} < \gamma$ is present, the system must downgrade to a mode that relies on the six-dimensional torque feedback of the end effector $F_{ext}$ (force control mode). If $\exists \|F_{ext}\| > \tau_{safe}$ occurs under this condition, the safety shutdown policy $a_{suspend}$ must be executed immediately (e.g., retracting the device a short safe distance along its current axis and locking the joint), with an alert issued for the doctor to take over (Supervised Autonomy).

The system must be downgraded. What does downgrading mean?

What does "six-dimensional torque feedback $F_{ext}$ (force control mode) depending on the end effector" mean?



13. **Online System ID + Residual Policy**: During live operation, by using real-time manipulation $a_t$ and feedback force $F_t$, the implicit mechanical characterization of the current tissue $z_{phy} = \phi(a_{t-k:t}, F_{t-k:t})$ is inferred online. Then, a residual policy network $\pi_{res}$ is used to compensate for the Sim-to-Real gap:

$$a_{real} = \pi_{sim}(o_t) + \pi_{res}(o_t, z_{phy}, F_t)$$



Why is residual control required here? Adding them directly seems odd to me. Is there any theoretical basis for this approach?



14. To process discrete tokens, a dynamic step size parameter dependent on input $\Delta_t \in \mathbb{R}$ is required for discretization.

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

How is this $\Delta_t$ calculated? Please give me an example.





15. Diffusion-based generation of Action Chunks creates blind spots in open loop execution. To handle the real-time nonlinear deformation of tissues during surgery, strict closed loop correction is necessary:

- **Receding Horizon Control (RHC)**: The model generates a Chunk containing the $T$ step trajectory: $\hat{A}_t = [a_t, a_{t+1}, \dots, a_{t+T-1}]$. However, we only execute the first $k$ steps ($k \ll T$), and obtain new observations $o_{t+k}$ at $t+k$ to generate a new Chunk.

- **High-frequency multi-modal interruption**: During the extremely short execution time, a ultra-high-frequency torque feedback (Force/Torque Sensor) is introduced. The currently measured torque vector is defined as $F_{ext}$, and the safety threshold is $\tau_{safe}$. The controller logic is as follows:

  $$a_{exec} = \begin{cases} \hat{A}_t[i], & \text{if } \| F_{ext} \| \le \tau_{safe} \\ a_{suspend}, & \text{if } \| F_{ext} \| > \tau_{safe} \end{cases}$$

  Once the $\| F_{ext} \|$ mutation crosses the boundary (indicating a change in the tissue shape as expected), the underlying controller immediately cuts off the current Chunk.

How should this be corrected? The organization has undergone a non-linear displacement. How does your system perform closed loop correction? It is equivalent to the observed state changes.





## Answer to the question



### 1. How is continuous intra-frame trajectory interpolation implemented?

In embodied AI, the inference speed of VLM (vision-language model) is very slow (e.g., only 5Hz), while underlying real robot control (such as the Franka or Unitree you are familiar with) typically requires high-frequency commands (e.g., 70Hz to over 500Hz, similar to the 71Hz 1-NFE rapid inference scenario you may be studying).

**How interpolation is done:**

Here, “interpolation” is not simply drawing a straight line between two points (linear interpolation), but rather **generating a high-frequency continuous trajectory (Action Chunk) through the Diffusion model**.

Assume that at the moment $t$, the VLM receives the visual observation $O_t$, and takes $200$ milliseconds to output a high-dimensional feature or target point $z_t$. During these $200$ milliseconds, if interpolation is not performed, the robot will freeze.

The Diffusion model operates at a very high frequency (e.g., 71Hz). It receives $z_t$ and the current body state $q_t$ (joint angle), and directly generates a smooth action sequence for $k$ time steps into the future:

$$a_{t:t+k} \sim p_\theta(a | q_t, z_t)$$

These $k$ actions (such as $k=14$) perfectly fill the “perception blind spot” between two visual updates of VLM in physical execution, ensuring that the robotic arm’s movement is continuous and smooth.

------

### 2. What does $a_{suspend}$ mean? When will it trip? What action will be taken?

- **What it is**: $a_{suspend}$ means “system suspend” or “safe emergency stop policy”.

- **When the fuse is blown**: It triggers immediately when the underlying high-frequency torque sensors (such as 1000Hz sampled six-dimensional force sensors) detect that the external force $\| F_{ext} \|$ exceeds the preset tissue safety tear threshold $\tau_{safe}$. In a very short time (milliseconds), the underlying controller directly takes control, blocking the output of the large model.

- **What action to perform**: **It is absolutely not a deadlock (stasis) in place**. If the scalpel has already caused tissue deformation due to pressure, staying still in place will continue to cause damage to the tissue. The standard $a_{suspend}$ action is **Compliant Retraction**:

  $$a_{suspend} = - \lambda \cdot v_{approach}$$

  That is, move backward by a very small safe distance (e.g., 3 mm) in the direction where the end effector just approached (vector $v_{approach}$), release the contact force, then lock the motor brake, and wait for the doctor to take over.

------

### 3. What are the specific formulas for the Langer–von Neumann dynamics sampling collapse to "absolute left" or "absolute right"?

In the Diffusion model, the generation process is an iterative denoising procedure. Langevin dynamics uses a score function $\nabla_a \log p(a|o)$ to gradually guide pure random noise toward a high-probability data manifold (true physical trajectory).

The specific iterative sampling formula is:

$$a_{k-1} = a_k + \frac{\alpha_k}{2} \nabla_a \log p(a_k | o) + \sqrt{\alpha_k} \epsilon_k$$

- $a_k$: The action sequence with noise in step $k$.
- $\alpha_k$: A hyperparameter controlling the sampling step size (similar to the learning rate).
- $\nabla_a \log p(a_k | o)$: The score function, predicted by U-Net or Transformer. Its physical meaning is "the steepest direction toward a realistic and reasonable action in the current noisy action space".
- $\epsilon_k \sim \mathcal{N}(0, I)$: The injected standard Gaussian noise.

**How to avoid oscillation**: At the beginning of denoising (when the noise is high), the model oscillates between “left” and “right”. However, as $k$ decreases, the injected noise $\epsilon_k$ becomes smaller and smaller. The gradient term of the score function takes precedence, causing the system to experience **symmetry breaking**, and randomly collapse into one of the peak regions (such as the absolute left side). Ultimately, a definite obstacle avoidance trajectory is generated.

------

### 4. Mamba linear compression is essentially lossy. What does "lossy" mean?

Lossy is a concept in information theory, corresponding to Lossless.

- **Transformer is lossless**: Its KV Cache stores all past Tokens completely in the visible memory. It’s like using a camera to record the entire surgery process, allowing you to replay each moment at original quality.
- **Mamba is lossy**: Whether the surgery lasts one minute or an hour, all past information is forcedly compressed and combined into a vector $h_t$ of a fixed dimension (such as $1024$). It’s like compressing a two-hour movie into a 100KB JPG image; older or less important details (such as the slight shaking of the robotic arm 10 minutes ago) are mathematically erased during the matrix multiplication iterations and can never be accurately restored.

------

### 5. How does the hierarchical semantic anchor work? Where do the $O(n^2)$ in Mamba and A, B come from?

**Specific case explanation:**

Suppose we are having a "cholecystectomy".

The underlying Mamba controls the robotic arm to move at 50Hz, handling images and angles. After 30 minutes, if it relies solely on the underlying hidden state $h_t$, Mamba might forget which step it is currently in.

At this point, the high-level VLM (e.g., running every 5 seconds) noticed that the gallbladder duct in the image had been clamped, and it output a discrete semantic token: "cut off the gallbladder duct".

This concept is transformed into a high-dimensional vector $c_{semantic}$.

Insert formula:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t + W c_{semantic}$$

- $\bar{A} h_{t-1}$: The underlying Mamba remembers the physical position of the scissors at the previous millisecond.
- $\bar{B} x_t$: New visual features input at the current moment.
- $W c_{semantic}$: **This is the anchor point.** It acts like applying a "thought seal" to $h_t$, forcing the subsequent state update to focus on the physical distribution of "cutting" rather than "sealing", thereby maintaining the stability of the long logic.

**Why does Mamba have $O(n^2)$? Where do A and B come from?**

- **Sources of A and B**: In SSM (state space model) theory, the $A$ matrix is typically initialized as the HiPPO matrix (a special matrix that can optimally remember historical polynomials), and then fine-tuned and updated during neural network training as a learnable parameter. $B$ is a linear layer parameter that projects the input $x_t$ into the hidden space.
- **Regarding $O(n^2)$**: In the formula $h_t = \bar{A} h_{t-1}$, $h_t \in \mathbb{R}^n$ (with a hidden state dimension of $n$) and $\bar{A} \in \mathbb{R}^{n \times n}$ is a square matrix. The multiplication of a $n \times n$ matrix by a $n$-dimensional vector has a computational complexity of exactly $O(n^2)$.* (Note: However, in the actual Mamba implementation, the author cleverly imposes a diagonalization assumption constraint on $\bar{A}$, turning it into element-wise multiplication, thereby further reducing the step complexity in engineering to $O(n)$. This can be mentioned during an interview and will be very beneficial.)*

------

### 6. In the information bottleneck problem, how are $H$ and $C$ calculated?

- **$H(X)$ (information entropy)**: Measures the uncertainty or complexity of the current manipulation trajectory. Assuming the surgical motion follows a Gaussian distribution, the formula for its differential entropy is:

  $$H(X) = \frac{1}{2} \log((2\pi e)^D |\Sigma|)$$

$D$ is the action dimension, and $\Sigma$ is the covariance matrix of action distribution. If it is extremely high-frequency and complex fine-tuning (hand tremor, minor obstacle avoidance), the variance is very large, and $H(X)$ will be large.

- **$C(h_t)$ (hidden state capacity)**: Measures the number of bits that the Mamba hidden vector can hold. Assuming the hidden state $h_t$ has a dimension of $n$, and uses 16-bit floating points (FP16), its absolute physical capacity limit is $n \times 16$ bits.

- **Conclusion**: When the surgical action is extremely precise, and the number of bits required exceeds the upper limit of the total bits in the hidden state vector, data overflow occurs, and high-frequency spatial features are physically cut off.

------

### 7. Why does Mamba not have Attention Dilution?

- **Reason for dilution in Transformer**: Its attention weights are $softmax(QK^T)$. The denominator of Softmax is $\sum_{j=1}^N \exp(q_i \cdot k_j)$. When the sequence length $N$ reaches tens of thousands (during a few hours of surgery), the denominator becomes infinite, causing most useful local weights to approach $0$ (and thus to be diluted).
- **Mamba has no dilution**: Mamba does not compute a global Softmax! It decides whether to "cover" or "retain" through a gate variable $\Delta_t$. When important information is encountered, it directly overwrites the corresponding neurons in $h_t$, without having to compete with the previous tens of thousands of Tokens for normalization. Therefore, no matter how large $N$ is, the response strength of key information is not diluted.

------

### 8. Why "visual + a small number of historical frames + instructions" is sufficient to solve MDP?

In physics, to describe the complete dynamic state of an object, you only need to know its **position (zero order), velocity (first derivative), and acceleration (second derivative)**.

- Current visual frame: Provides the **position** of all objects $O_t$.

- History 3 frames: Through finite difference, it implicitly provides **velocity** $V_t \approx (O_t - O_{t-1})/\Delta t$ and **acceleration** $A_t$.

  In line with the target set by the text instructions, this includes all the state variables required for the physics engine to simulate the next step. It satisfies the Markov property: the future state depends only on the dynamic features calculated from these current figures, and has nothing to do with where it was on the table one hour ago.

------

### 9. Specific scenarios of RLHF/DPO

Assume the task is: **"remove polyps"**.

- $x$ (context): The screen shows a polyp next to a blood vessel, with a robotic arm holding an electrocautery device.
- $y_w$ (good action): The model slowly moves the electrocautery device and makes a precise cut in the direction away from the blood vessel.
- $y_l$ (bad action): The model moves too much, cutting the polyp but also tearing the blood vessel beside it.
- **DPO approach**: When the doctor sees the action $y_l$, it is marked as “dangerous”. $\pi_\theta$ is the model we are training, and $\pi_{ref}$ is the frozen initial model. In the formula, $\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}$ increases the output probability of the good action $y_w$ continuously, while $-\log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$ forces the probability of the bad action $y_l$ to decrease. As a result, the model becomes “extremely timid and cautious”.

------

### 10. What does `softplus` mean? The relationship between $\Delta_t$ and A, B

- **`softplus` meaning**: It is an activation function with the formula $\text{softplus}(z) = \log(1 + \exp(z))$. Its function is to map any real number to a **positive number**. Since $\Delta_t$ represents the physical "time step" or "update rate", time cannot be reversed, and the rate cannot be negative. Therefore, a layer of `softplus` must be added to ensure it remains positive.

- **Regarding how A and B are calculated here**:

  In the Mamba architecture, the matrix $A$ is usually a fixed or slowly learned global parameter (to remember history); while $B_t$ is dynamically generated with the current input:

  $$B_t = \text{Linear}_B(x_t)$$

  The discrete $\bar{B}_t$ is determined by $\Delta_t$, continuous $A$, and continuous $B_t$:

  $$\bar{B}_t = (\Delta_t A)^{-1}(\exp(\Delta_t A) - I) \cdot \Delta_t B_t$$

*(When $\Delta_t \to 0$, $\exp(\Delta_t A) \to I$ occurs, leading to $\bar{B}_t \to 0$, which does not absorb new input.)*

------

### 11. What is residual connection? What is the internal mapping of $f_{anomaly}(x_t)$?

- **Residual Connection**: This is $y = f(x) + x$. The idea is that if the complex Mamba path (state compression) loses information, I directly create a “green channel” to add the original information to the final result.
- **Internal mapping of $f_{anomaly}(x_t)$**: It can be a very small and fast-convolving Convolutional Neural Network (CNN) or Multi-layer Perceptron (MLP). Its task is very specific: specifically detecting the sudden increase in **red pixels** in the image or the **instantaneous pulses** from the torque sensor. Once detected, it outputs a large activation value, which directly affects the final output through the residual channel $y_t$, bypassing the slow Mamba memory mechanism.

------

### 12. What is system downgrade? What does six-dimensional torque feedback mean?

- **System Degradation**: Just like an autonomous vehicle encountering heavy rain and fog, making the camera images unreadable, the system automatically degrades from “Advanced Visual Assist Driving (NOA)” to “Adaptive Cruise Control (ACC) relying solely on radar”. Here, it shifts from the complex “visual large model end-to-end control” to the basic “classic industrial robot physical control”.
- **6D Force-Torque Feedback**: You must be familiar with the F/T Sensor on the flange of the Franka robotic arm. In force control mode, the machine no longer relies on images but solely on sensory feedback. “6D” refers to three forces along the $X, Y, Z$ axis and three torques around the $X, Y, Z$ axis. It doesn’t matter if something is broken; as long as the hand feels resistance ahead (encounters an obstacle), it will retreat according to the predefined mechanical rules (such as admittance control).

------

### 13. Why is residual control necessary? What is the theoretical basis for direct addition?

Why is the formula $a_{real} = \pi_{sim}(o_t) + \pi_{res}(o_t, z_{phy}, F_t)$?

- **The theoretical basis for direct addition**: This comes from *Disturbance Rejection* in control theory and *Residual RL* in reinforcement learning.
- **Why not directly modify the model (fine-tuning $\pi_{sim}$)**:
  1. It is impossible to collect hundreds of thousands of failed surgical data points in the real physical world (especially on living organisms) to fine-tune a large model (too low sample efficiency and excessive cost).
  2. $\pi_{sim}$ has already learned a very robust baseline manipulation trajectory (skeleton) in simulation (such as Isaac Sim/Lab).
  3. Differences in the real world (e.g., pork is more slippery than in simulation) are just *high-frequency, low-amplitude physical disturbances*. Letting a small network $\pi_{res}$ only learn to output a tiny compensation amount $\Delta a$ (e.g., “press 1 mm more to the left”) not only speeds up training, but after clipping, it is absolutely safe and will not disrupt the original macro logic of $\pi_{sim}$.

------

### 14. How is $\Delta_t$ calculated? Provide an example

**Calculation formula:**

$$\Delta_t = \text{softplus}(W_\Delta x_t + b_\Delta)$$

Here, $W_\Delta$ is a set of learned weight matrices, and $x_t$ is the current input feature vector.

**Specific examples:**

Suppose you are cutting through tissue:

- **Time $t_1$**: The image $x_{t1}$ shows the scalpel moving suspended above the tissue, with nothing happening around it. The model has learned that this is a redundant action. $W_\Delta x_{t1}$ calculates a very small negative value, which passes through `softplus`, and then goes through $\Delta_{t1} \approx 0.01$. At this point, the system treats this image as a “useless background” and ignores it; the hidden state $h_t$ remains almost unchanged.
- **Time $t_2$**: The image $x_{t2}$ shows the moment when the tip of the scalpel pierces the tissue, with a little bleeding. The model's feature recognition identifies this as a key physical contact point! $W_\Delta x_{t2}$ calculates a very large positive value, which passes through `softplus`, and then goes through $\Delta_{t2} \approx 5.0$. This large step instantly activates the $\bar{B}$ matrix, forcing the critical state of “tissue pierced” into the hidden state $h_t$, guiding the next step to slow down.

------

### 15. How does RHC correct organizational non-linear displacement?

**Specific correction process:**

The diffusion model took a look at the image at $t_0$, found the polyp at coordinate $(10, 10)$, and then generated a perfect trajectory Chunk $\hat{A}_t$ consisting of 20 steps, planning to grab the polyp in 2 seconds.

- **If no RHC (open loop) is performed**: The robotic arm walks these 20 steps with its eyes closed. However, at 0.5 seconds, the tissue undergoes nonlinear displacement, and the polyp slides onto $(12, 8)$. The subsequent 1.5 seconds of movement by the robotic arm result in no contact with the tissue, even causing damage to the tissue.

- **If doing RHC (closed loop)**:

  1. At $t_0$ moment, a 20-step trajectory is generated, and **we only execute the first 2 steps**.

  2. After completing these 2 steps (taking only 0.2 seconds), immediately take another photo (obtain the observation $O_{t+2}$).

  3. In this new photo, it is clearly shown that the polyp has slid. The current coordinates are $(10.5, 9.5)$.

4. The diffusion model uses this **latest image** as a condition, **abandons the old trajectory**, and **generates** a new 20-step Chunk.

  5. Repeat the above process.

    **Summary**: The closed loop correction of the system does not attempt to “modify” the incorrect trajectory. Instead, by rapidly “observing, discarding old plans, and generating new plans,” it continuously updates the displacement (state changes) in the environment to match the latest input conditions, ensuring that the action sequence always tracks the real-time position of the target.



## Follow-up Question 1

1. $\nabla_a \log p(a_k | o)$: A score function predicted by U-Net or Transformer. Its physical meaning is "the steepest direction toward a realistic and reasonable action in the noisy action space at present". Is this thing trained?

2. The $A$ matrix is usually initialized as the HiPPO matrix (a special matrix that optimally remembers historical polynomials). What is this matrix? Please explain it in detail and provide numerical examples to illustrate it. In engineering, the matrix is sometimes treated as a diagonal matrix. Is that the case?

3. How is the gate control variable $\Delta_t$ calculated? Provide a specific calculation example. What is the physical meaning of this variable?

4. **Residual Connection**: This is $y = f(x) + x$. The idea is that if the complex Mamba path (state compression) loses information, I directly create a "green channel" to add the original information to the final result. In this case, residual connection is addition. So why is it called "residual"?



## Question 1 Answer

### 1. Does the score function $\nabla_a \log p(a_k | o)$ need to be trained?

**It must be trained, and it is the core of the Diffusion model training.**

- **Physical meaning and pain points**: $\log p(a|o)$ represents a real, highly complex distribution of physical movements (such as a perfect surgical trajectory). We **do not know the analytical expression for this distribution mathematically**, and naturally, we cannot directly differentiate it to calculate the actual score $\nabla_a \log p(a_k | o)$.

- **Role of the large model**: Since it cannot be calculated, we use a neural network (U-Net or Transformer) to **approximate fit** this score. We define the network as $s_\theta(a_k, k | o)$, and its goal is to approximate the true $\nabla_a \log p(a_k | o)$.

- **Clever equivalence (unification of DDPM and Score-based Model)**: According to the Tweedie formula (Note 1) and Denoising Score Matching theory, the score of the current state is strictly **proportional to the injected noise** in mathematics:

  $$\nabla_a \log p(a_k | o) \propto -\epsilon$$

  This is why, in the engineering implementation, the U-Net you see does not directly output a "score vector", but instead outputs **predicted noise $\epsilon_\theta(a_k, k | o)$**. During training, the MSE loss forces the network to predict noise that approximates the real added noise:

  $$\mathcal{L} = \mathbb{E}_{k, a_0, \epsilon} [ \| \epsilon - \epsilon_\theta(a_k, k | o) \|_2^2 ]$$

Therefore, training a diffusion model is essentially training this score function.

------

### 2. What is the HiPPO matrix for initializing the Markdown matrix $A$? Is it a diagonal matrix in engineering terms?

**HiPPO (High-order Polynomial Projection Operators)** is a mathematical framework that can "optimally remember continuous historical signals".

- **Principle**: If we want a finite-dimensional hidden state $h_t$ to remember an infinitely long history sequence, the optimal approach is not simple decay (like in RNNs), but rather projecting the historical signals onto a set of orthogonal polynomial bases (such as Legendre polynomials). The HiPPO matrix $A$ is the derived state transition matrix.

- **Specific numerical example**: Taking the classic HiPPO-LegS (scaled Legendre) as an example, the mathematical definition of its continuous matrix $A_{nk}$ is:

  When $n > k$, $A_{nk} = -(2n+1)$

  When $n = k$, $A_{nn} = -(n+1)$

  When $n < k$, $A_{nk} = 0$

  If we take a very small $3 \times 3$ hidden state dimension, the matrix $A$ looks like this (a lower triangular matrix):

  $$A = \begin{bmatrix} -1 & 0 & 0 \\ -3 & -2 & 0 \\ -5 & -5 & -3 \end{bmatrix}$$

All its eigenvalues have negative real parts, which ensures the stability of the dynamical system (historical information will decay smoothly rather than explode) (Note 2).

- **Engineering diagonalization**:

  **Your intuition is very sharp; in engineering, this would definitely be treated as a diagonal transformation.**

If you have studied the evolution from S4 to Mamba, you will find that:

  1. The pure HiPPO matrix is a dense lower triangular matrix, and calculating the matrix exponent $\exp(\Delta A)$ and matrix multiplications per step are too slow.
  2. In the S4 paper, the authors used mathematical techniques to diagonalize the HiPPO matrix into $A = V \Lambda V^{-1}$, and only used the complex eigenvalues on the diagonal $\Lambda$ in the complex domain $\mathbb{C}$ for computation.
  3. **More aggressive in the Mamba paper**: The authors found that there is no need to bother initializing a perfect HiPPO matrix and diagonalizing it. Mamba directly **parametrizes $A$ as a real diagonal matrix**. This makes the computation extremely efficient on hardware (GPU SRAM), reduces the complexity of parallel scanning to a minimum, and, combined with data-dependent $\Delta_t$, still achieves or even surpasses the memory performance of the original HiPPO.

------

### 3. How is the gate control variable $\Delta_t$ calculated? What is its physical meaning?

**Calculation Example:**

$\Delta_t$ is a scalar (or vector broadcast) that is entirely calculated by the input $x_t$ at the current moment (e.g., the visual token captured at that time) through a simple linear layer and activation function:

$$\Delta_t = \text{softplus}(\text{Linear}(x_t)) = \log(1 + \exp(W \cdot x_t + b))$$

Assume that the weights of our linear layer $W$ have been trained.

- **Scenario A (useless information)**: The robot looks at the white wall (background redundancy), at this time $W \cdot x_t + b = -5$.

$\Delta_t = \log(1 + e^{-5}) \approx 0.0067$ (a very small value).

- **Scenario B (key information)**: The robot sees the moment when the scalpel touches the tissue, at this time $W \cdot x_t + b = 5$.

$\Delta_t = \log(1 + e^5) \approx 5.0067$ (a very large value).

**Physical Meaning:**

- In continuous-time dynamics, $\Delta_t$ is the **physical sampling step length** (e.g., $0.01$ seconds).

- However, in Mamba's Selective Scan, $\Delta_t$ becomes the **Information Absorption Rate due to data dependencies**.

  According to the discretization formula $\bar{A} = \exp(\Delta_t A)$ (note that $A$ is negative):

- When $\Delta_t \to 0$, $\bar{A} \to I$. The system "freezes" time, and the hidden state remains unchanged ($h_t \approx h_{t-1}$), ignoring all current useless inputs.

- When $\Delta_t$ is large, $\bar{A} \to 0$. The system "accelerates" time, clears the old historical inertia, and significantly absorbs the current key new features into the hidden state.

------

### 4. Why is $y = f(x) + x$ called "Residual"?

This relates back to the mathematical essence of ResNet (Residual Network) when it was introduced.

Assume that at a certain layer in the network, the underlying target mapping function we really want to learn is $H(x)$.

- **Traditional network structure**: It is expected that the non-linear layer $f(x)$ can perfectly fit $H(x)$ directly. However, experiments show that when the network is very deep, directly fitting the identity mapping (where output equals input) is very difficult, leading to gradient vanishing.

- **Residual network concept**: Since it is difficult to directly learn $H(x)$, I will modify the formula to force the nonlinear network layer to learn the **difference** between the target and input, that is:

  $$f(x) \equiv H(x) - x$$

In mathematics and statistics, the difference between an observation value (or target value) and a reference value (or input value) is strictly defined as **residual**.

  Therefore, **the "residual" is what the **nonlinear layer $f(x)$** learns**.

Finally, to obtain the desired target output $y$ (i.e., $H(x)$), we need to add back the reference value $x$:

  $$y = f(x) + x$$

Therefore, it is called residual connection. It is not because addition itself is a residual process, but because that complex network branch $f(x)$ **is specifically responsible for fitting the residuals**. In the Mamba hybrid architecture, this means that the Mamba layer only needs to focus on learning the complex temporal patterns in long sequences, without having to memorize the instantaneous raw spatial features (the raw features are handled by $+x$).



## Follow-up Question 2

1. What are Legendre polynomials? How are they calculated and applied in our field? Can you give an example?

2. In the S4 paper, the author uses mathematical techniques to diagonalize the HiPPO matrix into $A = V \Lambda V^{-1}$, and only uses the complex eigenvalues on the diagonal $\Lambda$ in the complex domain $\mathbb{C}$ for calculations.

What is this V? Won't the calculation be lost in the end?

3. Mamba directly **parametrizes $A$ as a real diagonal matrix.** How is the parametrization done? Are there any numerical requirements?

4. Can you quickly write down the core forward propagation logic in Mamba that includes $\Delta_t$ discretization and parallel scanning using PyTorch pseudocode?





## Question 2 Answer

### 1. What are Legendre polynomials? How are they calculated and used in HiPPO?

**What is it:**

In mathematics, the Legendre polynomials are a set of orthogonal polynomial basis functions on the interval $[-1, 1]$. It can be understood as the "Fourier transform of the polynomial world". The Fourier transform uses sine/cosine waves to fit signals, while the Legendre polynomials use orthogonal combinations of $x, x^2, x^3 \dots$ to fit signals.

The previous levels look like this:

$P_0(x) = 1$

$P_1(x) = x$

$P_2(x) = \frac{1}{2}(3x^2 - 1)$

**Application in SSM/HiPPO:**

In continuous robot control or video streams, the signal $u(t)$ is continuously generated. It is impossible to store all the continuous signals from $t=0$ to the present in memory.

**The core idea of HiPPO is: "orthogonal polynomial projection" for continuous signals.**

We use the previous $N$-th Legendre polynomial to approximate the infinite-length signal in the past. Since $P_n(x)$ is a basis function, the past signal can be expressed as a weighted sum of these $N$ basis functions.

We just need to remember these $N$ coefficients, and compress the infinite-length history into a $N$-dimensional hidden state vector $h(t) \in \mathbb{R}^N$.

**Examples of derivation and calculation:**

Assume we want to track how the $N$ coefficients $h(t)$ change over time $t$. By taking the derivative of the coefficients with respect to time $\frac{d}{dt} h(t)$, through rigorous mathematical derivation (using the derivative recurrence property of Legendre polynomials), we will be surprised to find that it perfectly becomes a linear ordinary differential equation (ODE):

$$\dot{h}(t) = A h(t) + B u(t)$$

The $A$ matrix here is **exactly composed of the natural recurrence coefficients of the Legendre polynomial derivatives!** This is why the HiPPO matrix $A$ has that special lower triangular form (as described earlier in $- (2n+1)$). It is not fabricated randomly, but rather a inevitable mathematical result of the recombination of coefficients after differentiating the orthogonal polynomials.

------

### 2. What is $V$ in S4 diagonalization? Is it discarded last?

**Conclusion: $V$ is indeed discarded (or more precisely, "absorbed"). In the actual neural network forward propagation, there is no need to calculate or retain $V$.**

**Mathematical derivation proof:**

In S4, the continuous state equation is:

$$\dot{h}(t) = A h(t) + B x(t)$$

$$y(t) = C h(t)$$

Perform eigendecomposition on $A$: $A = V \Lambda V^{-1}$. Here, $\Lambda$ is a diagonal matrix with all diagonal elements being complex eigenvalues, and $V$ is the matrix of characteristic vectors.

We substitute $A$ into the original equation and perform a **Basis Transformation**.

Define a new hidden state $\tilde{h}(t) = V^{-1} h(t)$, and multiply both sides by $V^{-1}$:

$$V^{-1} \dot{h}(t) = V^{-1} (V \Lambda V^{-1}) h(t) + V^{-1} B x(t)$$

$$\dot{\tilde{h}}(t) = \Lambda \tilde{h}(t) + (V^{-1} B) x(t)$$

For the output equation:

$$y(t) = C h(t) = C V (V^{-1} h(t)) = (C V) \tilde{h}(t)$$

**Miracle from an Engineering Perspective:**

See, the transformed new system becomes:

$$\dot{\tilde{h}}(t) = \Lambda \tilde{h}(t) + \tilde{B} x(t)$$

$$y(t) = \tilde{C} \tilde{h}(t)$$

Among them, $\tilde{B} = V^{-1} B$, $\tilde{C} = C V$.

In neural networks, $B$ and $C$ are **randomly initialized linear layer weights that are learned end-to-end along with the Loss**. Since they are variables that learn freely, the network can directly learn $\tilde{B}$ and $\tilde{C}$ themselves!

Therefore, the complex $V$ and $V^{-1}$ are directly **physically absorbed** by these two learnable linear layers. In the code, we only retain the complex diagonal matrix $\Lambda$ to participate in the core recursive calculations, completely eliminating the dense matrix multiplication of $O(N^2)$.

------

### 3. Mamba parameterizes $A$ as a real diagonal matrix. How is this done? What are the numerical requirements?

Albert Gu, the author of Mamba, made an extremely drastic simplification in engineering: since computing the complex diagonal matrix on a GPU is still inconvenient, I simply reduced it to a **pure real diagonal matrix**, and made it dependent on data.

**Numerical Requirements:**

The diagonal elements of $A$ must be **strictly negative**.

Why? Because during discretization, $\bar{A} = \exp(\Delta_t A)$.

- If $A > 0$, $\exp(\text{正数}) > 1$, or the hidden state explodes exponentially during the time-step recursion, the network becomes NaN instantly.
- Only $A < 0$, $\exp(\Delta_t A)$ serve as a decay gate between $(0, 1)$, ensuring BIBO stability for the dynamic system (bounded input, bounded output).

**Specific PyTorch parameterized implementation:**

To prevent the gradient from updating $A$ as a positive number during backpropagation, an engineering technique called **Exponential Reparameterization** is used:

In the code, it is actually not initialized at the beginning. $A$ It is essentially a learnable parameter. $P$（`nn.Parameter`）。

$$A = - \exp(P)$$

No matter what real number $P$ becomes, $A$ will always be negative.

In the Mamba source code, the initialization of $A$ usually sets a sequence from $1$ to $N$ along the feature dimension, that is:

$A_{init} = [-1, -2, -3, \dots, -N]$. This allows different channels to have different "historical forgetting decay rates", capturing features at different time scales.

------

### 4. Handwritten Mamba forward propagation core logic (including discretization and scanning)

During the interview, tearing up this pseudo-code directly wins the job. To make you clearly see the internal data flow, I have written a highly clear pure PyTorch concept version and annotated the key points of parallel scanning replacements.



```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaCore(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state # 隐状态维度 N

        # 1. 参数化实数对角阵 A (初始化为 -1, -2... -N)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A_init)) # 学习 log(A)

        # 2. 生成依赖于输入 x_t 的 B, C, Delta 的线性层
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model) # 输出B, C, Delta的原始值
        self.dt_proj = nn.Linear(d_model, d_model) # 专门用于对 Delta 进一步投影

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        B, L, D = x.shape

        # 严格保证 A 为负数对角线
        # A 维度: (d_state,)
        A = -torch.exp(self.A_log)

        # 1. 计算数据依赖的 B, C, Delta
        x_proj = self.x_proj(x) # (B, L, d_state * 2 + d_model)

        # 拆分特征
        delta_raw, B_t, C_t = torch.split(x_proj, [self.d_model, self.d_state, self.d_state], dim=-1)

        # 对 Delta 应用 softplus 保证步长为正
        # Delta 维度: (B, L, d_model)
        Delta_t = F.softplus(self.dt_proj(delta_raw))

        # 2. ZOH 离散化 (Zero-Order Hold)
        # 为了计算高效，Mamba 实际使用广播机制进行元素级乘法，而不是稠密矩阵乘

        # 维度扩展对齐: Delta_t (B, L, D, 1) * A (d_state,) -> (B, L, D, d_state)
        Delta_A = Delta_t.unsqueeze(-1) * A.view(1, 1, 1, -1)

        # A_bar: (B, L, d_model, d_state)
        A_bar = torch.exp(Delta_A)

        # B_bar: Mamba 源码中通常使用简化版的欧拉离散化 B_bar = Delta * B，或者一阶泰勒展开
        # 这里写严格的简化版 ZOH (B_t 形状 B, L, 1, d_state)
        B_bar = Delta_t.unsqueeze(-1) * B_t.unsqueeze(-2)

        # 3. 扫描过程 (Scan)
        # ======= 面试重点 =======
        # 这里用 for 循环展示真实的数学递推过程。
        # 在 Mamba 底层的 CUDA 实现 (selective_scan_fn) 中，
        # 这个 O(L) 的 for 循环被替换成了基于 Blelloch 算法的 Parallel Associative Scan。
        # 它利用了前缀和（Prefix Sum）的思想，在 GPU 的 SRAM 中将串行计算并行化，耗时降为 O(log L)。
        # =======================

        h = torch.zeros(B, self.d_model, self.d_state, device=x.device)
        ys = []

        for t in range(L):
            A_bar_t = A_bar[:, t, :, :] # (B, d_model, d_state)
            B_bar_t = B_bar[:, t, :, :] # (B, d_model, d_state)
            x_t = x[:, t, :].unsqueeze(-1) # (B, d_model, 1)
            C_t_curr = C_t[:, t, :].view(B, 1, self.d_state) # (B, 1, d_state)

            # 状态更新 h_t = A_bar * h_t-1 + B_bar * x_t
            # 注意：A_bar 是对角阵，所以退化为点乘 (element-wise multiplication)
            h = A_bar_t * h + B_bar_t * x_t

            # 输出投影 y_t = C_t * h_t (点乘后求和)
            y_t = torch.sum(h * C_t_curr, dim=-1) # (B, d_model)
            ys.append(y_t)

        y = torch.stack(ys, dim=1) # (B, L, d_model)

        return y
```

Core highlights:

1. **Pure dot product without matrix multiplication**: Since $A$ is parameterized as a real diagonal matrix, `A_bar_t * h` in the code performs an extremely efficient element-wise multiplication, rather than the time-consuming `matmul`.
2. **Physical meaning**: `Delta_t` controls the decay gate of `A_bar`. The larger `Delta_t`, the closer `A_bar` approaches 0, quickly forgetting history; the smaller `Delta_t`, the closer `A_bar` approaches 1, maintaining memory.
3. **Nature of parallel scanning**: The formula for `h_t = A_t * h_{t-1} + B_t x_t` satisfies the **associative property**. As long as the associativity is satisfied, a parallel prefix and algorithm can be used on a GPU for parallel reduction. This is the core secret behind Mamba's ability to overcome the seriality of RNNs and achieve training speeds comparable to those of Transformers.





## Question 3

1. I understand that if the large model does not produce new outputs, the previous output will be used as the input for the diffusion model in the current inference.
2. Is the zero-force drag mode the same as the impedance control mode?
3. What is the specific formula for the score function?
4. Will this hidden state superposition change the mean and variance of the original latent vector? It seems so. Can Mamba still operate properly? Is it necessary to include this conditional vector during training?
5. The last question, which is also the sixth question: This n appears to be a constant, right? A relatively small constant.







## Question 3 Answer

### 1. When there is no new output from VLM, should the previous conditional input be directly reused?

**Completely correct.**

In control theory, this is called the **Zero-Order Hold (ZOH) policy**.

When the VLM (frequency of 1Hz) outputs a target vector $g_{t}$ (such as "suture needle coordinates" or the corresponding semantic Embedding), within the next 1 second, even if the underlying diffusion model (frequency of 50Hz) performs 50 inferences, these 50 conditions $c$ will be "locked" as $g_{t}$. Only when the VLM refreshes $g_{t+1}$ the next second does the condition update. This ensures that during the interval when the VLM is "thinking," the underlying system can still maintain smooth interpolation and obstacle avoidance at high frequencies.

### 2. Is the zero-force drag mode the same as the impedance control mode?

**Different. Although they make the robots appear "soft" on a macroscopic level, the underlying physical equations are completely different.**

- **Zero force drag mode (Gravity Compensation):**

- **Principle**: The system uses the robot's dynamics model to calculate in real time the gravitational force on all joints under the current posture. Then, the motor generates a torque of equal magnitude but opposite direction to counteract gravity.
- **Sensation**: The robot enters a "weightless" state, as if floating in space. You can push it with just one finger, and it stops where you push it (like playdough). It **has no tendency to return to its original position**.

- **Impedance Control Mode:**

- **Principle**: It constructs a virtual "mass-spring-damping" system at the end. The core formula is:

    $$M \ddot{x} + B \dot{x} + K (x - x_d) = F_{ext}$$

    (Where $K$ represents stiffness, $B$ represents damping, $x_d$ represents the target position, and $F_{ext}$ represents external forces).

- **Sensation**: The robot acts like a **spring**. If you push it, it will be pushed back gently (due to damping and stiffness, it is not as hard as a stone); but once you release your hand, it **automatically bounces back** to its original target position $x_d$. During surgery, when encountering unstoppable tissue collisions, impedance control can prevent hard tearing, and the surgery can continue after the collision disappears.

### 3. What is the specific formula for the Score Function?

Yes. In diffusion models (especially continuous-time score-based generative models), the formulas are very elegant.

- **Definition**: The Score function is essentially the log-gradient of the data probability density distribution:

  $$s(x) = \nabla_x \log p(x)$$

- **The role of neural networks**: We do not know the mathematical formulation of the actual surgical action distribution in the physical world $p(x)$. Therefore, we use a neural network $s_\theta(x, t)$ to **approximate** this gradient.

- **Training objective (Score Matching)**:

  Through the noise addition process, we make the target computable. The essence of training is for the neural network to predict the added noise $\epsilon$. Under this derivation, Score is exactly equal to negative noise divided by variance:

  $$\nabla_{x_t} \log p(x_t) \approx s_\theta(x_t, t) = - \frac{\epsilon_\theta(x_t, t)}{\sigma_t}$$

- **Sampling process (Langevin dynamics discretization)**:

  $$x_{t-1} = x_t + \frac{\sigma_t^2}{2} \nabla_x \log p(x_t) + \sigma_t z$$

Just take a half-step in the direction of the gradient, and add some random perturbation $z$ to explore the multi-peaked distribution.

### 4. Does the superposition condition vector affect the mean/variance of the Mamba hidden state? Can it still function properly?

**You hit the core problem! Your intuition is completely correct: if we simply add them up ($h_t + W c$), the mean and variance will explode, and the model will collapse directly (OOD, out-of-distribution deviation).**

- **Must be added during training**: Since it is used in inference, **you must not train without this condition at all**. The conditional injection module must be included in the forward pass during the entire training process, so that the network can adapt to this distribution change in advance.

- **Engineering solution (FiLM mechanism)**:

  In actual large models or video generation models, addition is rarely used to integrate cross-modal conditions. **FiLM (Feature-wise Linear Modulation)** or **Cross-Attention** mechanisms are usually used. If it is FiLM, the formula becomes as follows:

  $$h'_t = \gamma(c_{semantic}) \odot h_t + \beta(c_{semantic})$$

The semantic vector of VLM $c_{semantic}$ is processed by a small MLP, which outputs two scaling and translation parameters $\gamma$ and $\beta$. Through this **affine transformation (scaling + translation)**, combined with subsequent LayerNorm or RMSNorm normalization, the semantic information can be injected into Mamba while strictly controlling the drift of the mean and variance, ensuring proper inference of Mamba.

### 5. $n$ seems to be a relatively small constant, right?

**Yes, absolutely correct.**

- Here, uppercase $N$ It is **Sequence Length**. For example, if you input 10,000 frames of surgical images, then $N = 10000$。
- Lowercase $n$ It is the **dimension of the internal hidden state of Mamba** (State Dimension), or the number of feature channels. In practical applications, this $n$ Usually set to small values such as 16, 64, 128, or 256.
- Since $n$ They are fixed hyperparameters (constants) set by humans. Then, when evaluating with sequence length $N$ When the complexity of growth increases,$\mathcal{O}(N \cdot n)$ It can be regarded as $\mathcal{O}(N)$, which is **linear complexity**. This is exactly why Mamba can "outperform" Transformer $\mathcal{O}(N^2)$ The root cause is that over time, the VRAM usage of Transformer explodes exponentially, while Mamba is linear.



---

**Tweedie Formula** is a magical tool from Bayesian statistics. Its most significant contribution is: **If you observe a sample contaminated by Gaussian noise, as long as you know the "score" (the gradient of the log probability) of this noise distribution, you can derive the optimal estimate of the original graph.**

In the context of diffusion models, it explains why "prediction noise" and "estimated scores" actually do the same thing.

------

### 1. Mathematical expression: Seeking truth from noise

Suppose we have a raw signal $z$ (such as a clean image), and we add Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ to it, resulting in the observed value $y = z + \epsilon$.

According to the Tweedie formula, given the observation $y$, the posterior expectation of the original signal $z$ (i.e., the most reliable denoised estimate) can be expressed as:

$$\mathbb{E}[z | y] = y + \sigma^2 \nabla_y \log p(y)$$

There are two key items here:

- **$y$**: Current observation value (noisy data).
- **$\nabla_y \log p(y)$**: This is the **score**, which represents the gradient direction of the log probability at $y$.

> **Intuitive understanding:** This formula tells you that to retrieve the original signal, you only need to start from the observation point $y$, and take a small step in the direction where the probability increases fastest (the scoring direction).

------

### 2. Why does it unify DDPM and Score-based Models?

In the diffusion model, the noisy state $a_k$ at time $k$ can be written as:

$$a_k = \sqrt{\bar{\alpha}_k} a_0 + \sqrt{1-\bar{\alpha}_k} \epsilon$$

If we apply this expression to the Tweedie formula and perform simple algebraic transformations (treating $a_0$ as $z$, and $a_k$ as $y$), you will find that:

$$\hat{a}_0 = \frac{1}{\sqrt{\bar{\alpha}_k}} \left( a_k + (1-\bar{\alpha}_k) \nabla_{a_k} \log p(a_k) \right)$$

Meanwhile, we know that the reconstruction formula for $a_0$ (starting from the defined noise) is:

$$\hat{a}_0 = \frac{1}{\sqrt{\bar{\alpha}_k}} (a_k - \sqrt{1-\bar{\alpha}_k} \epsilon)$$

**Comparing the two formulas above, the conclusion is obvious:**

$$\nabla_{a_k} \log p(a_k) = - \frac{\epsilon}{\sqrt{1-\bar{\alpha}_k}}$$

This is the $\nabla_a \log p(a_k | o) \propto -\epsilon$ you mentioned.

------

### 3. Core Meaning: Different Paths Lead to the Same Goal

The Tweedie formula played a "bridge" role in the development of diffusion models:

- **Score-based Models（Score-based Matching）**: It is believed that we should learn how to climb back to the high-probability region along the slope (learning $\nabla \log p$).
- **DDPM（Diffusion Model）**: It is believed that we should learn how to "remove" the injected noise (predicting $\epsilon$).

**Tweedie formula proves that:** As long as Gaussian noise exists, **"prediction noise" is essentially modeling the "score function"**.

This is why in the code you see, it clearly reads `loss = MSE(pred_noise, true_noise)`, but in the paper it is said that it performs "score matching". By predicting noise, the network actually implicitly understands the geometric structure of the data distribution.

---

**How to approximate the "original $a_0$" using "observed $a_k$" and "distribution gradient (score)".**

------

### 1. Preparation: Definition of Forward Diffusion

In DDPM, the reparameterization formula for the noisy sample $a_k$ from the clean sample $a_0$ to the step with noise $k$ is:

$$a_k = \sqrt{\bar{\alpha}_k} a_0 + \sqrt{1-\bar{\alpha}_k} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

From this expression, we can inversely solve for $a_0$:

$a_0 = \frac{1}{\sqrt{\bar{\alpha}_k}} \left( a_k - \sqrt{1-\bar{\alpha}_k} \epsilon \right) \quad \text{--- (式 A)}$

------

### 2. Introduce the Tweedie formula

Tweedie's formula states that for data that satisfy an exponential distribution (with Gaussian distribution as a special case), given the observation $y$, the empirical Bayesian estimate of the mean (the ideal noise removal result) is:

$$\mathbb{E}[z | y] = y + \Sigma \nabla_y \log p(y)$$

In the context of diffusion models, we align the variables:

- **Observation value $y$** corresponds to the noisy state $a_k$ at time $k$.
- **Original signal $z$** corresponds to the signal $\sqrt{\bar{\alpha}_k} a_0$ at time $0$ (Note: Here, Tweedie estimates the mean with coefficients).
- **Noise variance $\Sigma$** corresponds to the variance of the injected noise at this time $(1-\bar{\alpha}_k) \mathbf{I}$.

Introduce the Tweedie formula:

$$\mathbb{E}[\sqrt{\bar{\alpha}_k} a_0 | a_k] = a_k + (1-\bar{\alpha}_k) \nabla_{a_k} \log p(a_k)$$

Organize it and calculate the estimate $a_0$: $\hat{a}_0$.

$\hat{a}_0 = \frac{1}{\sqrt{\bar{\alpha}_k}} \left( a_k + (1-\bar{\alpha}_k) \nabla_{a_k} \log p(a_k) \right) \quad \text{--- (式 B)}$

------

### 3. Key Equivalence Transitions

Now, let's compare **(Equation A)** and **(Equation B)**:

- **Starting from the definition (A):** $\hat{a}_0 = \frac{1}{\sqrt{\bar{\alpha}_k}} ( a_k - \sqrt{1-\bar{\alpha}_k} \epsilon )$
- **Starting from Tweedie (B):** $\hat{a}_0 = \frac{1}{\sqrt{\bar{\alpha}_k}} ( a_k + (1-\bar{\alpha}_k) \nabla_{a_k} \log p(a_k) )$

To make these two equations hold, the middle term must be exactly equal:

$$(1-\bar{\alpha}_k) \nabla_{a_k} \log p(a_k) = -\sqrt{1-\bar{\alpha}_k} \epsilon$$

Finally, export the famous equivalence relation:

$$\nabla_{a_k} \log p(a_k) = - \frac{\epsilon}{\sqrt{1-\bar{\alpha}_k}}$$

------

### 4. Conclusion: Why does U-Net output noise?

This formula reveals profound engineering implications:

1. **Mathematical equivalence**: The score function of the data distribution is proportionally equal to the noise in network prediction $\epsilon$.
2. **Engineering convenience**: At $k \to 0$, the score function $\nabla \log p$ tends to infinity (because the data distribution becomes very sharp), which is difficult to train in neural networks. However, the noise $\epsilon$ always follows a standard normal distribution, and its magnitude remains stable.
3. **Prediction noise = finding gradients**: When training the prediction noise of U-Net using MSE, you are actually teaching the network to learn how to move from low-probability regions (noise areas) to high-probability regions (data areas).

------

This equivalence is the cornerstone for the success of diffusion models.



---

Why does the presence of negative real parts for all eigenvalues ensure the stability of the dynamical system?

This is a very classic and profound control theory/dynamic system problem. To understand why "negative real parts of eigenvalues" is equivalent to "system stability," we need to consider it from two perspectives: the **solutions of differential equations** and the **physical meaning on the complex plane**.

In the HiPPO or more general state space model (SSM), the state evolution of the system typically follows a linear ordinary differential equation:

$$\dot{h}(t) = Ah(t) + Bx(t)$$

To analyze stability, we focus mainly on the system's "homogeneous solution" (i.e., the evolution pattern of the system itself without external input $x(t)$): $\dot{h}(t) = Ah(t)$.

------

### 1. Viewing "exponential decay" from the general solution

For the first-order linear differential equation $\dot{h} = Ah$, the form of its general solution is always closely related to the **exponential function**:

$$h(t) = e^{At}h(0)$$

If we perform a feature decomposition on the matrix $A$ (assuming it is diagonalizable, i.e., $A = V \Lambda V^{-1}$), then the solution can be further broken down into multiple feature modes:

$$h(t) = \sum_{i} c_i e^{\lambda_i t} v_i$$

Here, $\lambda_i$ is the **eigenvalue** of the matrix $A$. Now, let's observe the trend of each item $e^{\lambda_i t}$ over time.

Since the eigenvalue $\lambda_i$ is usually complex, it can be written as: $\lambda = \sigma + j\omega$ (where $\sigma$ is the real part and $\omega$ is the imaginary part). According to Euler's formula:

$$|e^{\lambda t}| = |e^{(\sigma + j\omega)t}| = |e^{\sigma t} \cdot e^{j\omega t}| = e^{\sigma t}$$

- **If the real part $\sigma > 0$**: $e^{\sigma t}$ will increase infinitely over time, causing the system to **explode**.
- **If the real part $\sigma = 0$**: The system will continue to oscillate ($\omega \neq 0$) or remain constant, which is called critical stability.
- **If the real part $\sigma < 0$**: $e^{\sigma t}$ will approach $0$ over time. The system will **converge** back to the equilibrium point.

------

### 2. Physical Meaning in the HiPPO Matrix

The core task of HiPPO is **“memory”**. If the eigenvalue has a negative real part, it means:

- **Preventing numerical overflow**: When handling extremely long sequences (such as 10k or even 1M steps), if the system is slightly unstable (positive real part), the error will amplify exponentially, causing the hidden state $h$ to become `NaN`.
- **Smooth decay (forgetting)**: The negative real part ensures that “historical information” gradually decays in a controllable manner. The brilliance of the HiPPO matrix lies in its eigenvalues being distributed in the left half-plane of the complex plane, with a specific distribution, which allows it to capture time correlations at different scales rather than blindly forgetting everything.
- **The “sink” of the system**: On the phase plane, this means that all state trajectories will eventually be attracted toward the origin, rather than diverging away.

### 3. Why is the A matrix of HiPPO all negative?

The HiPPO-LegS matrix you mentioned:

$$A = \begin{bmatrix} -1 & 0 & 0 \\ -3 & -2 & 0 \\ -5 & -5 & -3 \end{bmatrix}$$

This is a lower triangular matrix. For a lower triangular matrix, **the eigenvalues are the elements on the diagonal**.

In this example, the eigenvalues are $\{-1, -2, -3\}$. They are all **negative real numbers**.

This ensures that:

1. **Absolute stability**: No imaginary part, which means the system will not oscillate at all; it will only converge smoothly.
2. **Multi-scale memory**: The eigenvalue $-1, -2, -3$ corresponds to different decay rates. This difference in rates is designed to match the basis functions of different orders in the Legendre polynomial, thereby accurately depicting the "shape" of historical signals.

------

**Summary:**

The real part of the eigenvalues determines whether the system has "gain" or "loss". In HiPPO, the negative real parts act as a **"gravity"**, ensuring that the system remains on a stable track while receiving new information, and does not completely lose control due to the accumulation of information.
