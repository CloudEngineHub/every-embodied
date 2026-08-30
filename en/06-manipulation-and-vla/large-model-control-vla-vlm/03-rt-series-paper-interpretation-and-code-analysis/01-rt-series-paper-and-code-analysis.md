# Robot Transformer Series: Detailed Introduction of RT-1, RT-2 and RT-X

## 1. Introduction: From Expert Model to Generalist Model

In traditional robot learning, due to data scarcity and the complexity of the physical world, we typically train specialized models for specific tasks. However, the success of large language models (LLMs) has inspired the robotics field: Can we build a general **Robot Foundation Model** that understands manipulation in the physical world just as ChatGPT understands language?

The RT series from Google DeepMind is a milestone in this direction:

- **RT-1 (2022)**: Demonstrated the effectiveness of the Transformer architecture in robot control, and introduced the core concept of **Action Tokenization**.
- **RT-2 (2023)**: Proposed the **VLA (Vision-Language-Action)** paradigm, treating robot actions as a “language” to facilitate the transfer of internet knowledge to the robot domain.
- **RT-X (2023)**: Through the **Open X-Embodiment** dataset, proved the feasibility of training using **Cross-Embodiment** across robot forms.

------

## 2. RT-1: The Transformer Base with Robot Manipulation

RT-1 is a end-to-end control model based on Transformer. Its core contribution lies in designing an efficient architecture that can output control commands in real time at a frequency of 3Hz, and it has high anti-interference capabilities.

### 2.1 Model Architecture

RT-1 uses **EfficientNet** as the visual encoder, **FiLM (Feature-wise Linear Modulation)** layer for integrating language instructions, and finally **Token Learner** compresses the visual feature input into a Transformer.

<p align="center">

<img src="../../../../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/03RT系列论文解读与代码分析/assets/image1-1771476766353-16.png" width="90%" />

<b> Figure 1: Schematic diagram of the RT-1 model architecture and FiLM adjustment mechanism </b>

</p>

### 2.2 Core Innovation: Action Tokenization

This is the cornerstone of the RT series. Traditional imitation learning usually focuses on continuous action values (such as speed and position). RT-1 discretizes continuous actions.

- **Discretization of intervals**: The actions in each dimension (such as moving the $x, y, z$ axis, gripper opening and closing, etc.) are evenly divided into 256 intervals (Bins).
- **Tokenization**: Each action dimension becomes a classification problem (integers from 0 to 255).
- **Advantage**: Transformers are highly adept at processing discrete token sequences (similar to predicting the next word), and this design transforms robot control into “predicting the next action word”.

### 2.3 Python Code Analysis: Action Tokenizer

The following code demonstrates how RT-1 converts continuous robotic arm actions into Tokens readable by Transformers.



```Python
import numpy as np
import tensorflow as tf

class RT1ActionTokenizer:
    def __init__(self, action_min=-1.0, action_max=1.0, vocab_size=256):
        self.min = action_min
        self.max = action_max
        self.vocab_size = vocab_size

    def tokenize(self, action_continuous):
        """
        将连续动作 [-1, 1] 转换为离散 Token [0, 255]
        """
        # 1. 归一化到 [0, 1]
        action_norm = (action_continuous - self.min) / (self.max - self.min)
        action_norm = np.clip(action_norm, 0, 1)

        # 2. 映射到 [0, vocab_size - 1]
        action_tokens = np.floor(action_norm * (self.vocab_size - 1)).astype(np.int32)
        return action_tokens

    def detokenize(self, action_tokens):
        """
        将离散 Token 解码回连续动作
        """
        # 1. 映射回 [0, 1]
        action_norm = (action_tokens + 0.5) / self.vocab_size

        # 2. 反归一化到 [min, max]
        action_continuous = action_norm * (self.max - self.min) + self.min
        return action_continuous

# --- 示例 ---
tokenizer = RT1ActionTokenizer()
raw_action = np.array([0.5, -0.2, 0.9]) # x, y, z 速度
tokens = tokenizer.tokenize(raw_action)

print(f"原始动作: {raw_action}")
print(f"Token化结果: {tokens}")
# 输出示例: [191, 102, 242] -> 这些整数直接作为 Transformer 的输入 Target
```

------

## 3. RT-2: Visual-Language-Action (VLA) Model

RT-2 represents a qualitative leap. It no longer trains the robot model from scratch, but instead fine-tunes an already trained **Visual-Language Model (VLM)** (such as PaLI-X or PaLM-E) to enable it to generate actions.

### 3.1 Core Concept: Action is Language

In RT-2, robot actions are encoded as text strings (e.g., numerical tokens). For models, answering "What is in the image?" (output: "Apple") and answering "How to pick up the apple?" (output: "128 128 100...") are essentially the same.

This enables the RT-2 to inherit the **semantic understanding** and **logical reasoning** capabilities learned by VLM from massive internet data.



<p align="center">

<img src="../../../../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/03RT系列论文解读与代码分析/assets/Screenshot-2023-08-04-at-4.37.05-PM.webp" width="90%" />

<b> Figure 2: RT-2 Co-fine-tuning Policy </b>

</p>



### 3.2 Emergent Capabilities

Due to inheriting internet knowledge, RT-2 exhibits remarkable generalization capabilities:

1. **Symbolic reasoning**: The instruction “pick up a nearly extinct animal” allows RT-2 to identify dinosaur toys and grasp them (even though there is no pairing of “extinct animal” and the action in the training data).
2. **Visual semantic generalization**: It can recognize objects and backgrounds that are not seen in the training set.

### 3.3 Training Policy: Co-fine-tuning

RT-2 does not merely fine-tune robot data, but instead **mixes training** robot data with raw Internet Visual Question Answering (VQA) data.

- **Purpose**: To prevent "catastrophic forgetting". If only robot data is used for fine-tuning, the large model will become a dumb robot that only does tasks, losing its original common sense reasoning ability.

------

## 4. RT-X: Large-scale Generalization across Formats

If RT-2 addressed the issue of the "brain," RT-X focuses on addressing the diversity of the "body."

### 4.1 Open X-Embodiment dataset

Google, in collaboration with 33 laboratories, collected data from 22 different robots (such as Franka, UR5, WidowX, etc.) and created the Open X-Embodiment dataset.





<p align="center">

<img src="../../../../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/03RT系列论文解读与代码分析/assets/overview.png" width="90%" />

<b>Figure 3: Robot forms included in the Open X-Embodiment dataset </b>

</p>



### 4.2 Cross-Embodiment Reasoning

Different robots have different numbers of joints, arm lengths, and control frequencies. How can we control them with a single model?

- **Unified action space**: Map the actions of all robots to a standard 7-DoF (position + pose + gripper) space.
- **RT-X model**: Trained on mixed datasets based on the RT-1 and RT-2 architectures. Results show that **training with other robot data can improve the performance of one’s own robot** (Positive Transfer), especially for tasks with limited data.

------

## 5. Summary and Comparative Analysis

The following table summarizes the evolution logic of the RT series, which is crucial for understanding OpenVLA (the open-source reproduction version of RT-2).

| **Features**       | **RT-1**                               | **RT-2**                        | **RT-X**                           |
| -------------- | -------------------------------------- | ------------------------------- | ---------------------------------- |
| **Release Date**   | 2022                                   | 2023                            | 2023                               |
| **Core Architecture** | EfficientNet + Transformer             | VLM (PaLI-X / PaLM-E)           | RT-1 / RT-2 variants                |
| **Model Parameters** | 35M (very small, fast inference)       | 55B (very large, slow inference)   | Multiple specifications               |
| **Input Modes**   | Image + text instruction                  | Image + text                     | Image + text + robot type            |
| **Output Modes**   | Discrete action tokens                    | Text tokens (interpreted as actions) | Discrete action tokens                |
| **Key Contributions** | Verified the feasibility of Transformer + Tokenization | Proposed the VLA paradigm and achieved semantic reasoning | Validated the effectiveness of training on cross-robot form data |
| **Inference Frequency**   | ~3Hz (real-time)                            | <1Hz (requires cloud or quantization) | Depends on the base model              |

### Learning suggestions

For beginners, it is recommended to practice along the following path:

1. **Reproduction of RT-1**: Understand the basic Tokenization and Transformer training process (Google has opened sourced `rt-1` code).
2. **Using OpenVLA**: This is an open-source alternative to RT-2 (based on Llama 2 + Prismatic), and weights can be directly downloaded for fine-tuning (refer to the document in the previous chapter).
3. **Exploring BridgeData V2**: This is one of the most commonly used subsets in the RT-X dataset, suitable for testing cross-domain generalization.