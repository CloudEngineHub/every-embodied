# Supplement 04 Detail Questions

The Rotating Position Encoding (RoPE) is designed in the form of a rotation matrix due to its elegant mathematical derivation and well-thought-out engineering considerations.

The core design intention can be summarized in one sentence: **By injecting manipulation of “absolute position,” the dependence on “relative position” is naturally reflected in the inner product calculation of the attention mechanism.**

Here is the logical flow and engineering considerations for its design derivation:

## 1. Core motivation: Finding the perfect expression for relative positions

In the Transformer's self-attention mechanism, the core computation is the inner product of Query and Key $\langle q_m, k_n \rangle$.

Without any positional encoding, this inner product is only related to the semantics of the token itself and is "blind" to positions.

The author of RoPE (Su Jianlin et al.) proposed a concept: Can we find a positional encoding function $f(\cdot, \cdot)$ such that a Query with a position $m$ and a Key with a position $n$ result in an inner product that **depends solely on the original features of these two tokens and their relative positions $m-n$**?

In mathematical language, it means we want to find a function $f$ that satisfies:

$$\langle f_q(q, m), f_k(k, n) \rangle = g(q, k, m-n)$$

### 2. Mathematical Derivation: The Diminishing-Scale Attack on the Complex Field

To solve the above equation, the author cleverly utilized the **two-dimensional complex plane**.

If we consider the two-dimensional feature vector $(x_0, x_1)$ as the complex number $x = x_0 + i x_1$, then the above inner product can be equivalent to complex multiplication (taking the real part):

$$\langle q, k \rangle = \text{Re}(q \cdot k^*)$$

*(Note: $k^*$ is the conjugate complex of $k$)*

Given our objective, if we define the function $f$ as **multiplying by a rotation factor $e^{im\theta}$ related to the position $m$ on the complex plane**:

$$f_q(q, m) = q e^{im\theta}$$

$$f_k(k, n) = k e^{in\theta}$$

Then, their inner product in the complex domain miraculously simplifies:

$$\langle f_q(q, m), f_k(k, n) \rangle = \text{Re}\left( (q e^{im\theta}) \cdot (k e^{in\theta})^* \right)$$

$$= \text{Re}\left( q e^{im\theta} \cdot k^* e^{-in\theta} \right)$$

$$= \text{Re}\left( q k^* e^{i(m-n)\theta} \right)$$

**Conclusion**: The inner product result perfectly contains and only contains relative position information $m-n$.

Return the operations in the complex domain to the familiar real-number matrix representation. Complex multiplication $e^{im\theta}$ is equivalent to multiplying by a two-dimensional rotation matrix:

$$\begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

This is the origin of the formula in your image.

### 3. High-dimensional Expansion and Design Considerations

In actual large models, the feature dimension $d$ is much greater than 2. The design of RoPE is very sophisticated:

- **Orthogonal subspaces divided into pairs:** RoPE divides the $d$-dimensional feature space into $d/2$ two-dimensional subspaces, and applies the aforementioned rotation matrix to each of them.
- **Exponential decay of frequency (Distance Penalty):** Different rotation fundamental frequencies $\theta_i$ are designed for different two-dimensional planes. Usually, $\theta_i = 10000^{-2i/d}$ is used. This means:
  - The earlier feature dimensions (low-frequency parts) rotate very fast and are highly sensitive to local position changes.
  - The later feature dimensions (high-frequency parts) rotate extremely slowly, capturing global long-distance dependencies.
  - **Remote decay:** This combination of different frequencies causes the expectation value of inner products to decrease as the relative distance $m-n$ increases. This gives the model a natural "distance sense"; the attention weights of tokens farther away tend to decrease.
- **Efficiency in engineering calculations:** Although the formula uses matrix multiplication, in the actual underlying implementation of frameworks like PyTorch, this is actually a **extremely sparse block-diagonal matrix**. We don't need to perform dense matrix multiplications of $O(d^2)$, but instead use the technique of swapping element signs and multiplying (e.g., `[-q1, q0]`), reducing the time complexity to $O(d)$.

RoPE perfectly integrates absolute and relative positions, which is why most mainstream large models today, such as LLaMA and Qwen, almost exclusively use RoPE.

In the academic and industrial fields, the most popular research direction regarding RoPE is actually **Length Extrapolation**, which involves training on short texts and inference on long texts (e.g., based on NTK-Aware Scaling or YaRN rotation angle scaling).