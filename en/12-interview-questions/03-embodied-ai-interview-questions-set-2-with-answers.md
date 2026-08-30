RoboticsX(2024.06)

### ⭐️ Side 1: Mentor Interview (about 35 minutes)

- **Past project experience statement:** Focuses on the candidate's overall introduction and background of the projects they participated in.
- **Q&A session:** Specific tasks they may have been responsible for during their internship.

------

### ⭐️ Second Round: Leader Round (about 1 hour)

The interviewer has a background as a NTU Ph.D. with expertise in traditional control. The questions are very detailed, and they can be broken down by project as follows:

#### 1. Reinforcement Learning Planning Project

- **Data acquisition:** How was the simulation dataset obtained?
- **Algorithm robustness:** Will there be situations where the algorithm gets stuck in local optima (e.g., unable to escape a dead end)?
- **Simulation modeling:** Why is the Differential Model used in the Gazebo simulation?

#### 2. VLN (Visual Language Navigation) Project

- **Technical implementation:** The specific implementation method of 3D remapping.
- **Performance evaluation:** What is the actual performance?
- **Real-machine deployment:** Does the model support operation on real robot hardware? If so, what are the inference/execution speeds?

#### 3. Robot manipulation projects

- **Whole-body control (WBC):** What specific tasks does WBC perform?
- **Coordinate system definition:** Why is a reference coordinate provided for the Base?
- **Solution comparison:** Have pure end-to-end imitation learning methods been attempted?
- **Data collection:** How are the expert trajectories collected?
- **Sensors:** Why are six-dimensional force/torque sensors (F/T Sensors) installed?

#### 4. Mobile Robot Project

- **Computing efficiency:** How can the speed of global path planning be ensured in large-scale real-world scenarios with small grid cell sizes?
- **Model predictive control (MPC):** * What mathematical model is specifically used for MPC?
  - How are the acceleration constraints and dynamics constraints formulated?
  - How are the prediction horizon and step size set?
- **Tracking performance:** * Is angle tracking included during trajectory tracking?
  - Have oscillations or inability to keep up with the target values occurred during tracking?
- **Obstacle avoidance logic:** In a crowded environment (dynamic environment), do “sharp turns” or planning failures occur?

------

### ⭐️ Third round: HR interview (about 15 minutes)

- **Process Communication:** Explain the reasons for HC restoration and process restart.
- **Personal Background:** Discuss personal background and internship interests.
- **Employment Information:** Confirm the start date and duration of the internship.

---

# Answer

------

## 1. How was the simulation dataset obtained?

- **Core principle:** Obtaining simulation datasets essentially involves constructing a Markov Decision Process (MDP) in a physics engine (such as Gazebo or Isaac Sim). By running expert policies (such as A* or DWA) or random exploration policies, complete state transition data of the agent’s interactions with the environment are collected.

- **Mathematical derivation:** The reinforcement learning experience replay dataset $D$ is defined as:

  $$D = \{ (s_t, a_t, r_t, s_{t+1}, d_t) \}_{t=1}^N$$

  Among them, $r_t = \mathcal{R}(s_t, a_t)$ is the instant reward, and $d_t \in \{0, 1\}$ is the round termination flag.

- **Realization pseudocode (Python - simulation collection pipeline):**



  ```Python
  class SimulationEnv:
      def collect_data(self, episodes, steps_per_episode):
          dataset = []
          for ep in range(episodes):
              state = self.reset() # 初始化随机场景
              for step in range(steps_per_episode):
                  action = self.expert_policy(state) # 调用A*或DWA获取最优动作
                  next_state, reward, done, info = self.step(action) # 物理引擎步进
                  dataset.append((state, action, reward, next_state, done))
                  state = next_state
                  if done: break
          return dataset
  ```

### 2. Will there be a situation where the model gets stuck in local optimization (e.g., unable to escape a dead end)?

- **Core principle:** Definitely. In reinforcement learning that relies on current observations or artificial potential field methods (APF), singularities where gravitational and repulsive forces cancel out occur easily within U-shaped obstacles (dead ends), leading to the robot becoming immobilized.

- **Mathematical derivation:** The total force on the robot is composed of the target gravitational gradient and the obstacle repulsive gradient:

  $$F_{total} = -\nabla U_{att}(q) - \nabla U_{rep}(q)$$

When $\nabla U_{att}(q) = -\nabla U_{rep}(q)$ and $F_{total} = 0$, the system falls into a local minimum.

- **Implementation pseudocode (Python - ICM intrinsic curiosity reward mechanism):**

  ```Python
  def compute_intrinsic_reward(state, next_state, action, forward_model):
      # 预测特征与实际特征的MSE误差越大，说明环境越陌生，给予高探索奖励
      pred_next_feat = forward_model(state, action)
      actual_next_feat = feature_extractor(next_state)
      intrinsic_reward = torch.mse_loss(pred_next_feat, actual_next_feat)
      return intrinsic_reward.item() * scale_factor
  ```

### Local optimal (dead end) solution

#### A. At the traditional algorithm level: Improvement of the Artificial Potential Field Method (APF)

When the robot detects $F_{total} \approx 0$ but does not reach the target point, the following mathematical methods can be used:

1. **Introduce Virtual Target Point:**

Set up a virtual gravitational source temporarily near the dead-end exit to change the potential field distribution and guide the robot out of the U-shaped area.

2. **Random Perturbation/Simulated Annealing:**

Apply a short-term random force to the robot's position $F_{noise}$ to break the force balance.

   $$F_{actual} = F_{att} + F_{rep} + \eta \cdot \text{rand}(-1, 1)$$

   $\eta$: Disturbance coefficient.

#### B. Reinforcement learning level: Breaking "greed" and "short-termism"

In addition to the **ICM (Innate Curiosity)** mentioned in the code, there are also the following key methods:

1. **Inherent reward and exploration (Exploration Bonus):**

   - **ICM (Intrinsic Curiosity Module):** As mentioned earlier, "prediction error" is used as a reward. The more the robot feels that its predictions of the stuck states are inaccurate, the more it will try to explore, thus increasing the chance of it getting stuck in a dead end.
   - **RND (Random Network Distillation):** Predict the output of a fixed random network, using the difficulty of prediction to encourage exploration.

2. **Introduce memory-based RL:**

Use **LSTM** or **GRU** to process sequence states.

   - **Principle:** If the robot can remember “I have been rotating in this small area for a long time,” it will reflect this dead loop in its state representation, allowing the policy to learn to take non-repetitive actions.
   - **Mathematical representation:** The state $s_t$ is no longer a single frame of point cloud, but a hidden state $h_t = \text{LSTM}(s_t, h_{t-1})$.

#### C. System Engineering Level: Layered Planning (The Industry Standard)

This is the most commonly used approach in the industry (such as Tencent RoboticsX or Meituan Delivery): **Global Planner + Local Planner.**

- **Core logic:**

  - **Global Planning (Global):** Use A*, JPS, or RRT* to calculate a sequence of waypoints on the global map that theoretically avoids all U-shaped dead ends.
  - **Local Planning (Local):** The RL algorithm or MPC is responsible for tracking these local points only.

- **Mathematical guarantee:**

  The local target point $q_{sub\_goal}$ will be updated continuously to ensure that the gravitational field always points in the "hutout exit direction" rather than at the "target point across the wall."

------

### 🚀 Break down the implementation code: Breaking out of local optimization based on "random perturbation"

If you are asked to write how to improve APF on the spot during an interview, you can present this code with **state detection and perturbation regression** logic:

```Python
import numpy as np

def compute_apf_action(robot_pos, goal_pos, obstacles):
    # 1. 计算标准势场力
    f_att = k_att * (goal_pos - robot_pos)
    f_rep = np.zeros(2)
    for obs in obstacles:
        dist = np.linalg.norm(robot_pos - obs)
        if dist < d_min:
            f_rep += k_rep * (1/dist - 1/d_min) * (1/dist**2) * (robot_pos - obs) / dist

    f_total = f_att + f_rep

    # 2. 局部最优检测 (Local Minima Detection)
    # 如果合力极小但离目标还远，说明陷入了平衡点
    if np.linalg.norm(f_total) < 1e-3 and np.linalg.norm(goal_pos - robot_pos) > threshold:
        print("Detected Local Minima! Applying Perturbation...")
        # 3. 施加切向扰动或随机力
        f_total = np.array([-f_att[1], f_att[0]]) + np.random.normal(0, 0.5, 2)

    return f_total
```

- **`threshold`**: Allowed end point error range.
- **`[-f_att[1], f_att[0]]`**: Rotate gravity by 90 degrees and try to slide along the edge of the obstacle.

------

**Interview Script Summary:**

To solve local optima problems, I usually use a **layered planning architecture**. Through a global planner (such as A*), dead ends at the topological level are avoided, and then **reinforcement learning based on the Curiosity Mechanism (ICM)** or a **local planner with random perturbations** is applied to handle obstacles in dynamic environments. This combination ensures both global correctness and local robustness.



### 3. Why should a differential model be used in Gazebo simulation?

- **Core principle:** The differential model is the actual physical configuration of most indoor mobile chassis (AGV, floor sweepers). It possesses the ability to rotate in place under non-holonomic constraints, and mathematically, it can easily decouple and map the linear/angular velocities output by the high-level planner directly to the speeds of the two side motors.

- **Mathematical derivation:** Global state space update matrix (with spacing $L$ and radius $R$):

  $$\begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{\theta} \end{bmatrix} = \begin{bmatrix} \cos\theta & 0 \\ \sin\theta & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} \frac{R}{2}(\omega_R + \omega_L) \\ \frac{R}{L}(\omega_R - \omega_L) \end{bmatrix}$$

#### 1. What is the principle of the differential model?

The core of differential drive lies in **independent control**. It consists of two coaxial independent drive wheels, and the direction of the robot is changed through the **speed difference (Differential)** between the two wheels.

#### Physical Structure

  - **Dual-wheel coaxial:** Two driving wheels are mounted on the same axis.
  - **Independent drive:** Each wheel is driven by an independent motor, allowing different speeds $\omega_L$ and $\omega_R$.
  - **Support wheel:** Usually equipped with 1-2 caster wheels for balance, without affecting kinematics.

#### Core Kinematics Formulas

  Assume the robot's centerline speed is $v$, angular velocity is $\omega$, wheel radius is $R$, and wheel width is $L$:

  $$v = \frac{v_R + v_L}{2} = \frac{R(\omega_R + \omega_L)}{2}$$

  $$\omega = \frac{v_R - v_L}{L} = \frac{R(\omega_R - \omega_L)}{L}$$

  ------

#### 2. How is "in-place rotation" implemented?

The mathematical essence of Rotation in place is: **the linear velocity is 0, and the angular velocity is not 0**.

#### Mathematical Derivation

  To achieve in-place rotation, according to the above formula, we need to set $v = 0$:

  $$v = \frac{v_R + v_L}{2} = 0 \implies v_R = -v_L$$

This means: **As long as the speeds of the left and right wheels are equal in magnitude and opposite in direction, the robot will rotate around the midpoint of the two wheel axes (i.e., the robot center).**

#### The angular velocity at this moment

  $$\omega = \frac{v_R - (-v_R)}{L} = \frac{2v_R}{L}$$

By adjusting the size of $v_R$, you can precisely control the speed of rotation. This capability allows the differential robot to turn 180° directly when facing a "dead end", without having to perform a three-point turn like a car.

  ------

#### 3. In-depth Understanding: Non-Holonomic Constraint

This is a common trap question in interviews.

#### Why is it a non-complete constraint?

"Incomplete constraint" can be simply said as: **You cannot move in any direction instantaneously.**

  For the differential robot, the constraint formula it faces is:

  $$\dot{x}\sin\theta - \dot{y}\cos\theta = 0$$

This formula means: **At any moment, the robot's lateral (side) speed must be 0.** It cannot move left or right directly like a crab.

#### Where does the "flexibility" of the differential model lie?

  Although it has a non-complete constraint (cannot move horizontally), it possesses the ability of **"small-radius turning"** and even **"zero-radius turning"**.

- **Ordinary vehicle (Ackman steering):** The minimum turning radius is limited by the front wheel angle, and it cannot turn around in place.

- **Differential model:** By controlling $v_L$ and $v_R$, its turning radius $r$ can range from $0$ (in-place rotation) to $\infty$ (straight line driving).

    $$r = \frac{v}{\omega} = \frac{L}{2} \cdot \frac{v_R + v_L}{v_R - v_L}$$

  ------

  ### 🚀 Deep questions from the interviewer: If there are constraints, why is it workable in Gazebo?

  > **You can answer like this:**
  >
  > “The differential model is popular in Gazebo because it achieves a perfect balance between **the simplicity of mathematical expression** and **the flexibility of environmental adaptation**.
  >
  > 1. **Simple decoupling:** Its Jacobian is an analytical solution, and the mapping from the upper-level $(v, \omega)$ to the underlying motor does not involve complex inverse trigonometric functions.
  > 2. **Robust obstacle avoidance:** The ability to rotate in place greatly simplifies the ‘get-out-of-jail’ logic in local path planning. In narrow Gazebo simulation environments, it is less likely to get stuck than four-wheeled models.
  > 3. **Physical robustness:** Compared to the complex steering linkage physics simulation of the Ackermann model, the differential model only needs to simulate two friction discs, resulting in higher simulation stability (convergence of the ODE solver).”

### 4. Specific implementation scheme for 3D Remapping.

- **Core principle:** Using the camera intrinsic parameters, the 2D pixels and depth values of RGB-D are inversely projected into 3D point clouds in the camera coordinate system. Then, these are converted to the global coordinate system using the external parameters (TF tree) provided by SLAM. Finally, a weighted moving average method is used to integrate them into TSDF (truncated symbolic distance field) or voxel maps.

- **Mathematical derivation:** Calculation process for the world coordinate point $P_w$:

  $$P_w = T_{wc} \cdot P_c = \begin{bmatrix} R_{wc} & t_{wc} \\ 0^T & 1 \end{bmatrix} \begin{bmatrix} \frac{(u - c_x)Z}{f_x} \\ \frac{(v - c_y)Z}{f_y} \\ Z \\ 1 \end{bmatrix}$$



### 1. Detailed Analysis of Formula Parameters

This formula can be broken down into two parts: **inverse projection (pixel $\to$ camera)** and **coordinate transformation (camera $\to$ world)**.

#### 1. Left side: Target coordinates

- $P_w$: The 3D point coordinates in the World Frame, usually represented as $[X_w, Y_w, Z_w, 1]^T$. The **1** at the end is used to form a homogeneous coordinate for easier matrix multiplication.

#### 2. Intermediate: Extrinsic Matrix

- $T_{wc}$: The transformation matrix from the **camera coordinate system** to the **world coordinate system**, belonging to the $SE(3)$ group.
- $R_{wc}$: The **rotation matrix** of $3 \times 3$, representing the camera's orientation relative to the world.
- $t_{wc}$: The **translation vector** of $3 \times 1$, indicating the position of the camera focal point in the world coordinate system.
- $0^T$ and $1$: Completion matrices, making them the $4 \times 4$ matrix.

#### 3. Right side: 3D point under the camera coordinate system (inverse projection)

This part of $[X_c, Y_c, Z_c, 1]^T$ is derived from the **Pinhole Model**:

- $Z$: The **depth value** of this pixel (directly provided by the RGB-D camera's depth map).
- $u, v$: The **horizontal and vertical coordinates** of the pixel on the image (unit: pixel).
- $c_x, c_y$: **Principal Point** coordinate, which is the intersection of the camera optical axis with the image plane, usually near the center of the image.
- $f_x, f_y$: **Focal Length**, indicating the number of pixels occupied by a unit distance on the image plane.
- $\frac{(u - c_x)Z}{f_x}$: The $X$ axis coordinate under the camera system calculated using the similar triangle principle.
- $\frac{(v - c_y)Z}{f_y}$: The $Y$ axis coordinate under the camera system calculated using the similar triangle principle.

------

### II. Implementation Process of 3D Remapping and TSDF Integration

In actual robots (such as RoboticsX projects), since the camera is moving, the position of the same object seen in each frame changes. Therefore, the "map construction" needs to be achieved through the following steps:

1. **Point cloud generation:** Traverse each pixel in the depth map and calculate the 3D points in the camera frame using the camera intrinsic parameters.
2. **Coordinate alignment:** Use the real-time pose $T_{wc}$ provided by SLAM to transform all point clouds into a unified world coordinate system.
3. **Voxel fusion (TSDF):** * Divide space into small cubes (Voxels).
   - Calculate the distance (SDF) of each Voxel from the nearest object.
   - **Weighted fusion:** Update the values of Voxels using a moving average method to filter out random noise from the depth sensor.

------

### III. Complete Implementation Demo (Python + NumPy)

This demo shows how to generate a point cloud in world coordinates from a depth map, and perform a simple voxel update simulation.



```Python
import numpy as np

class VoxelMap:
    def __init__(self, resolution=0.05):
        self.res = resolution
        # 简单起见，用字典模拟稀疏体素网格: {(x,y,z): [tsdf_val, weight]}
        self.grid = {}

    def update_tsdf(self, world_points, max_dist=0.1):
        """
        简化版融合：将点云存入体素并更新权重
        """
        for pt in world_points:
            # 空间离散化，找到对应的体素索引
            idx = tuple((pt[:3] / self.res).astype(int))

            # 模拟加权融合更新 (Moving Average)
            if idx not in self.grid:
                self.grid[idx] = [1.0, 1] # [初始TSDF, 初始权重]
            else:
                curr_val, weight = self.grid[idx]
                # 新观测值的融合逻辑 (此处简化演示)
                new_weight = weight + 1
                self.grid[idx] = [curr_val, new_weight]

def remapping_demo():
    # 1. 模拟相机内参 (Intrinsic)
    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5

    # 2. 模拟一帧深度数据 (Depth Image) 640x480
    # 假设所有像素深度都是 2.0 米
    depth_img = np.ones((480, 640)) * 2.0

    # 3. 模拟 SLAM 提供的相机位姿 (Extrinsic T_wc)
    # 假设相机向 X 方向平移了 1 米，旋转为单位阵
    T_wc = np.eye(4)
    T_wc[0, 3] = 1.0

    # --- 开始核心计算 ---

    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(640), np.arange(480))

    # 第一步：逆投影到相机坐标系 P_c
    z = depth_img
    x_c = (u - cx) * z / fx
    y_c = (v - cy) * z / fy

    # 构造齐次坐标 [N, 4]
    points_c = np.stack([x_c, y_c, z, np.ones_like(z)], axis=-1).reshape(-1, 4)

    # 第二步：坐标变换到世界坐标系 P_w = T_wc * P_c
    # 矩阵运算：(4,4) * (4, N) -> 转置后为 (N, 4)
    points_w = (T_wc @ points_c.T).T

    print(f"成功将 {len(points_w)} 个点重映射到世界坐标系")
    print(f"前5个点坐标示例:\n{points_w[:5, :3]}")

    # 第三步：融合进地图
    my_map = VoxelMap()
    my_map.update_tsdf(points_w)
    print(f"地图更新完成，当前活跃体素数量: {len(my_map.grid)}")

if __name__ == "__main__":
    remapping_demo()
```

### Key points addition:

- **Performance optimization:** In real robots, due to the large number of pixels, this process is usually processed in parallel on **GPU (CUDA)** or using the integration functions of **Open3D**.
- **TSDF truncation:** The actual TSDF sets a `trunc_margin` (e.g., 10cm). Only points within this range are updated to save memory and ensure surface accuracy.

- **Implementation Pseudocode (C++ - TSDF voxel weighted fusion):**

This code is divided into three mathematical stages: **computing SDF**, **truncated normalization (TSDF)**, and **weighted fusion update**.

  ------

### 1. Calculate Symbol Distance (SDF)

  **Code:** `float sdf_measured = measured_depth - voxel->depth_from_camera;`

  **Mathematical expression:**

  $$sdf = d_{meas} - d_{vox}$$

  - **$d_{meas}$ (`measured_depth`)**: The true depth value observed by the sensor (such as an RGB-D camera) at the current pixel.
  - **$d_{vox}$ (`depth_from_camera`)**: The Euclidean distance or projection distance along the optical axis from the center of the voxel being processed to the focal point of the camera.
  - **Meaning**:
    - If $sdf > 0$: It indicates that the voxel is **behind** the object surface (the obscured area).
    - If $sdf < 0$: It indicates that the voxel is **in front** of the object surface (the open area).
    - If $sdf \approx 0$: It indicates that the voxel is exactly at the **object surface**.

  ------

### 2. Truncation and Normalization (TSDF)

  **Code:** `float tsdf_new = max(-trunc_margin, min(trunc_margin, sdf_measured)) / trunc_margin;`

**Mathematical expression:**

  $$tsdf_{new} = \frac{clamp(sdf, -\mu, \mu)}{\mu}$$

  - **$\mu$ (`trunc_margin`)**: truncation distance. We only care about a very small range near the surface; data far from the surface (too large positive or negative values) are meaningless for reconstruction.
  - **Normalization**: By dividing by $\mu$, the range of $tsdf_{new}$ is compressed to $[-1, 1]$.
    - $1$: indicates that the voxel is clearly before the surface.
    - $-1$: indicates that the voxel is clearly after the surface (usually no updates occur beyond this value to avoid penetration through objects).

  ------

### 3. Cumulative Weighted Fusion (Weighted Fusion)

This is the most critical **incremental update** part in the code, which is essentially an **online weighted moving average**.

  **Code:** `voxel->tsdf = (voxel->weight * voxel->tsdf + weight_new * tsdf_new) / (voxel->weight + weight_new);`

  **Mathematical expression:**

  $$F_{i}(\mathbf{x}) = \frac{W_{i-1}(\mathbf{x}) F_{i-1}(\mathbf{x}) + w_{i}(\mathbf{x}) f_{i}(\mathbf{x})}{W_{i-1}(\mathbf{x}) + w_{i}(\mathbf{x})}$$

  - **$F_{i}(\mathbf{x})$ (`voxel->tsdf`)**: The final TSDF value of the voxel $\mathbf{x}$ after fusion in frame $i$.
  - **$W_{i-1}(\mathbf{x})$ (`voxel->weight`)**: The cumulative weight over the previous $i-1$ frames.
  - **$f_{i}(\mathbf{x})$ (`tsdf_new`)**: The new TSDF value observed in the current frame $i$.
  - **$w_{i}(\mathbf{x})$ (`weight_new`)**: The observed weight (usually set based on the confidence or distance of the depth value; in simple processing, it is set to $1$).

  Last updated weight:

  $$W_{i}(\mathbf{x}) = W_{i-1}(\mathbf{x}) + w_{i}(\mathbf{x})$$

  ------

  ### 💡 Why do this?

  1. **Noise reduction**: The single-frame depth map contains random noise. Through multi-frame weighted average, positive and negative noise cancel each other out, making the isoperimetric surface of $tsdf = 0$ smoother.
  2. **Incremental computation**: This method does not require storing all the original point clouds; only a voxel grid needs to be maintained. The grid is updated with each new frame of data, and the memory usage remains relatively constant.
  3. **Surface extraction**: After fusion, the **Marching Cubes** algorithm can be used to search for points in $tsdf = 0$, thereby extracting a smooth 3D mesh.

  ```C++
  void updateVoxel(Voxel* voxel, float measured_depth, float weight_new) {
      float sdf_measured = measured_depth - voxel->depth_from_camera;
      float tsdf_new = max(-trunc_margin, min(trunc_margin, sdf_measured)) / trunc_margin;

      // 移动加权平均，滤除传感器噪声
      voxel->tsdf = (voxel->weight * voxel->tsdf + weight_new * tsdf_new) / (voxel->weight + weight_new);
      voxel->weight += weight_new;
  }
  ```

### 5. What is the actual running effect?

- **Core principle:** Rejecting qualitative descriptions, it directly uses the standard quantitative metric SPL from the field of embodied AI. It considers both navigation success rates and path optimality, and severely penalizes policies that “reach the destination but circle wildly in place.”

- **Mathematical derivation:**

  $$SPL = \frac{1}{N} \sum_{i=1}^{N} S_i \frac{l_i}{\max(p_i, l_i)}$$

  Where $S_i \in \{0, 1\}$ indicates success, $l_i$ is the shortest expert distance, and $p_i$ is the actual driving distance.

### 6. Does the model support running on real robot hardware? If so, what are the inference/execution performance?

- **Core principle:** Fully supported. The standard deployment method on real devices involves exporting the PyTorch model to ONNX, and then using the TensorRT engine for INT8 or FP16 quantization inference on edge computing platforms such as Jetson Orin, which can reduce latency from 30ms to under 5ms.

- **Implementation Pseudocode (C++ - TensorRT memory allocation and asynchronous inference):**

  ```C++
  void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
      void* buffers[2];
      cudaMalloc(&buffers[0], batchSize * INPUT_SIZE * sizeof(float)); // 输入显存
      cudaMalloc(&buffers[1], batchSize * OUTPUT_SIZE * sizeof(float)); // 输出显存

      // Host To Device
      cudaMemcpy(buffers[0], input, batchSize * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

      // TensorRT 异步推理
      context.enqueue(batchSize, buffers, stream, nullptr);

      // Device To Host
      cudaMemcpy(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  }
  ```

### 7. What tasks does WBC specifically implement?

- **Core principle:** Whole-body control (WBC) decouples tasks in a multi-degree-of-freedom high-dimensional space through priority and null-space projection. The typical priorities are: joint limit avoidance > center of mass (CoM) balance > end-effector trajectory tracking > posture optimization.

- **Mathematical derivation:** Whether at the kinematic or dynamic level, the core is the projection operator $(I - J_1^{+} J_1)$. Taking kinematic velocity control as an example, the secondary task $\dot{x}_2$ can only be effective within the zero space of the main task:

  $$\dot{q} = J_1^{+} \dot{x}_1 + (I - J_1^{+} J_1) J_{2|1}^{+} (\dot{x}_2 - J_2 J_1^{+} \dot{x}_1)$$



#### 1. What is Null Space Projection?

**Intuitive Understanding:**

Imagine holding a full cup of coffee in your right hand (Task 1: maintain the end pose). Since your arm has 7 degrees of freedom (redundancy), you can move your elbow freely without moving your hand at all (this is movement within the **zero space** of Task 1).

**Mathematical Definition:**

For linear equations $J\dot{q} = \dot{x}$, the **zero space** refers to the set of joint velocities $\dot{q}$ that satisfy $J\dot{q} = 0$.

**Zero-space projection matrix** $N = (I - J^+J)$ serves to project any joint motion command into this space, ensuring that the motion generated by this command **has no impact** on the execution of the main task.

------

#### 2. Detailed Explanation of the Four Levels of WBC Tasks

In WBC, tasks are arranged in priority order. High-level tasks have "absolute dominance" over low-level tasks.

- **Joint Limit Avoidance (Joint Limit Avoidance):**
  - **Meaning:** Protecting hardware. Each motor has a rotation angle limit.
  - **Importance:** **Highest priority**. If the pose is overstretched or hits the limit due to trajectory tracking, it can cause damage to the robot’s hardware or instantaneous instability of the control system.
- **CoM Balance (CoM Balance):**
  - **Meaning:** Controlling the robot’s Center of Mass within a supporting polygon.
  - **Importance:** Ensuring “no collapse”. For quadrupedal or bipedal robots, if the CoM becomes unstable, even if the end-effector grasps accurately, the entire robot will fall.
- **End-effector Tracking (End-effector Tracking):**
  - **Meaning:** Performing actual tasks. For example, the hand should grasp an object, and the foot should move to a specific coordinate.
  - **Importance:** This is the “business logic” of the robot. Under the premise of ensuring safety and stability, it aims to reach the target position accurately as much as possible.
- **Posture Optimization (Posture Optimization):**
  - **Meaning:** Making the robot look good or reducing energy consumption when there are remaining degrees of freedom.
  - **Importance:** **Lowest priority**. It is usually used to avoid abnormal poses or to bring the joints back close to their midpoint, leaving room for subsequent movements.

------

#### 3. In-depth Analysis of the Formula Meaning

$$\dot{q} = J_1^{+} \dot{x}_1 + (I - J_1^{+} J_1) J_{2|1}^{+} (\dot{x}_2 - J_2 J_1^{+} \dot{x}_1)$$

This formula is known as **Hierarchical Inverse Kinematics**. It describes how to calculate the joint velocities that take into account two priorities.

#### Variable Meaning Table:

| **Variable**               | **Meaning**                     | **Description**                                 |
| ---------------------- | ---------------------------- | ---------------------------------------- |
| $\dot{q}$              | **Joint velocity command**       | The control quantity finally output to the motor. |
| $J_1, J_2$             | **Jacobi matrix for tasks 1 and 2** | The "bridge" establishing the speed mapping between joint space and manipulation space. |
| $J_1^{+}$              | **Pseudoinverse of $J_1$** | Mapping the desired speed in manipulation space back to joint space.     |
| $\dot{x}_1, \dot{x}_2$ | **Desired speeds for tasks 1 and 2** | The task objectives defined in Cartesian space.             |
| $I$                    | **Identity matrix**                 | Used to construct the projection operator.                       |
| $(I - J_1^{+} J_1)$    | **Zero-space projection operator**           | **Core component**, ensuring that subsequent motion does not interfere with task 1. |
| $J_{2                  | 1}$                          | **Jacobi matrix for task 2 after projection** |

#### Breakdown of the formula logic:

1. **First item $J_1^{+} \dot{x}_1$**:

    Do our best to complete task 1. This is the minimum norm solution that satisfies task 1.

2. **The bracket content of the second item $(\dot{x}_2 - J_2 J_1^{+} \dot{x}_1)$**:

This is called **compensation items**. The execution of Task 1 has already produced some effects of Task 2 (i.e., $J_2 J_1^{+} \dot{x}_1$). Task 2 only needs to fill in the remaining gap.

3. **Preceding operator $(I - J_1^{+} J_1)$**:

    This is the "filter". It ensures that the joint movements required to compensate for the gap in Task 2 occur only in the zero space of Task 1, **and it will never disrupt the results of Task 1**.



### 8. Why provide reference coordinates for the Base?

- **Core principle:** The Mobile Manipulator has a high degree of redundancy (system degrees of freedom $n > 6$). Providing a base reference coordinate allows for the establishment of a local reference frame, decoupling “where the chassis moves” from “how the robotic arm operates”, preventing the inverse solver from diverging, and maximizing the manipulation space of the robotic arm.

- **Mathematical derivation:** The global end pose is composed of the chassis pose and the local relative pose of the arm as a homogeneous matrix multiplication:

  $$T_{end\_world} = T_{base\_world}(x_b, y_b, \theta_b) \cdot T_{end\_base}(q_{arm})$$

  Fixing or constraining $T_{base\_world}$ can reduce the dimensionality of complex SE(3) coupling problems.

#### 1. Why can a homogeneous matrix be "multiplied directly"?

The secret behind this is that the **Homogeneous Transformation Matrix** is designed in such a way that the two different mathematical operations of **rotation** and **translation** are unified into **matrix multiplication**.

### Chain reaction of coordinate transformation

In the mobile manipulator, we describe the position of the end effector in the world coordinate system, which is essentially a "coordinate system jump":

1. **Step 1**: The position of the end effector relative to the **robot base** (determined by the robotic arm joint angle $q_{arm}$).
2. **Step 2**: The position of the robot base relative to the **world coordinate system** (determined by the chassis pose).

### Mathematical Derivation

Assume that a point on the end effector has coordinates $P_{end}$ in the end coordinate system.

- The coordinates of this point in the base coordinate system are: $P_{base} = T_{end\_base} \cdot P_{end}$
- The coordinates of this point in the world coordinate system are: $P_{world} = T_{base\_world} \cdot P_{base}$

Substitute the first equation into the second equation:

$$P_{world} = T_{base\_world} \cdot (T_{end\_base} \cdot P_{end}) = (T_{base\_world} \cdot T_{end\_base}) \cdot P_{end}$$

It can be seen that the result of multiplying the two transformation matrices is exactly the transformation matrix that jumps directly from the end to the world:

$$T_{end\_world} = T_{base\_world} \cdot T_{end\_base}$$

**In simple terms:** The multiplication of homogeneous matrices is physically equivalent to the **composition of transformations**. As long as the coordinate system chain is continuous, you can continue multiplying indefinitely.

------

#### 2. What is SE(3)? What does it mean?

$SE(3)$ is not a "problem," but a mathematical space, officially known as **Special Euclidean Group**.

### Breakdown of Meaning

- **S (Special)**: Special. This means that the determinant of the matrix must be **1**. This ensures that the transformation only involves rotation and translation, without "mirror reflection" or "scaling", so the object does not become larger or smaller or flip.
- **E (Euclidean)**: Euclidean. This means that the transformation preserves the **Euclidean distance** between points in space. This is what is called a "rigid body transformation" — no matter how the robot moves, the arm itself remains unchanged in length.
- **(3)**: Represents 3-dimensional space.

### Mathematical Formulation

A $SE(3)$ matrix is usually a $4 \times 4$ matrix:

$$T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$$

- **$R \in SO(3)$**: The rotation matrix of $3 \times 3$, with 3 degrees of freedom.
- **$t \in \mathbb{R}^3$**: The translation vector of $3 \times 1$, with 3 degrees of freedom.
- **Total**: The $SE(3)$ space has **6 degrees of freedom** (3 rotations + 3 translations).

------

#### 3. What is the SE(3) coupling problem? Why reduce dimensionality?

When combining a 3-degree-of-freedom chassis ($x, y, \theta$) with a 7-degree-of-freedom robotic arm, the system has a total of **10 degrees of freedom**. However, our target pose $T_{end\_world}$ only requires **6 degrees of freedom** in the $SE(3)$ space.

### The Trouble with Coupling

Due to excessive degrees of freedom (redundancy), there are countless combinations of chassis and arm for the same end goal pose.

- **Mathematical Challenge**: When solving inverse kinematics (IK), the equations are underdetermined (more unknowns than equations). The slight vibrations of the chassis are amplified at the end via multiplication effects, leading to unstable calculations, oscillations, or inability to find solutions. This is known as the **SE(3) coupling problem**.

### The Art of Dimension Reduction

By "fixing or constraining $T_{base\_world}$", we artificially reduce the variable:

1. **Fixed base**: The chassis remains stationary. At this time, $T_{base\_world}$ becomes a constant matrix.
2. **Dimension reduction of the problem**: The inverse kinematics problem is reduced from "finding 6 targets with 10 variables" to "finding 6 targets with 7 variables (arm joints)".

### Why do this?

- **Computing efficiency**: The analytical or numerical solution for single-arm IK is much faster than full-body coupled IK.
- **Manipulation optimization**: Typically, the chassis is first moved to a “comfortable position” ($T_{base\_world}$ reference value). At this position, the robotic arm is far from the singular point, offering maximum flexibility.

------

### 💡 Summary

The multiplication of homogeneous matrices is the **chain rule of pose transformation**. $SE(3)$ is the mathematical language for describing a robot’s **rigid pose**. “Dimension reduction” is a method to find a fast and stable control scheme in complex redundant systems.



### 9. Have you tried pure end-to-end imitation learning methods?

- **Core principle:** Behavior cloning (BC) was attempted. However, pure BC suffers from a fatal Covariate Shift issue—since the recovery policy has not been learned under non-expert distribution conditions, even minor errors can cause the system to crash in the time dimension.

- **Mathematical derivation:** The loss function for BC is usually a negative logarithm of likelihood or MSE:

  $$\mathcal{L}_{BC}(\theta) = \mathbb{E}_{(s, a^*) \sim \mathcal{D}_{expert}} \left[ || \pi_\theta(s) - a^* ||^2_2 \right]$$

### 1. What exactly is the covariant offset?

#### 1. Core Definition

In machine learning, covariate shift refers to the difference in input distributions between the **training set** and the **test set** $P(s)$, while the conditional probability $P(a|s)$ (i.e., policy logic) remains unchanged.

- **During training**: The agent learns the states under the expert distribution $\rho_{\pi^*}$. The experts perform smoothly, and the provided states are "textbook-level".
- **During testing**: Due to the precision limitations of the model, the agent inevitably incurs minor errors $\epsilon$.

#### 2. Error Compounding (Compounding Errors)

This is the most fatal. In a dynamic system, the current action $a_t$ will affect the next state $s_{t+1}$.

1. The robot incurred a minor error, deviating from the expert trajectory.
2. It entered a state that no expert has ever encountered $s'$ (non-expert distribution area).
3. Due to the lack of $s'$ data in the training set, the robot began to make random movements.
4. These random movements caused further deviation, leading it into an entirely unknown domain, ultimately resulting in a system cascade failure (Cascade Failure).

#### 3. Mathematical derivation: the square growth of error

Assume that the worst error of a single-step action is $\epsilon$.

- In the task with $T$ time steps, the Total Variation Distance of BC is $O(\epsilon T^2)$.

- In contrast, if it is running within a distribution, the error is usually only linear $O(\epsilon T)$.

**The square-level growth of this $T^2$ represents the destruction of system stability due to the time-dimensional shift of co-variants.**

------

### II. How should this problem be solved?

There are currently four mainstream approaches in the industrial and academic sectors to "patch" this vulnerability:

#### 1. DAgger algorithm (Dataset Aggregation, the most classic approach)

**Principle**: Since the robot doesn’t know how to return after deviating, let experts tell it.

- **Process**:
  1. Run the current policy $\pi_\theta$ to collect a trajectory.
  2. Present these trajectories to an expert, asking, "If I were in these (deviated) positions, what would I do?"
  3. Add these "error sets" along with the expert’s correction actions to the training set and retrain.
- **Mathematical essence**: Through iteration, the distribution $\rho_{data}$ of the training set gradually approaches the distribution $\rho_{\pi_\theta}$ of the agent during actual operation.

#### 2. Noise injection and data augmentation (such as DART)

**Principle**: When collecting data from experts, deliberately create some confusion for them.

- **Method**: During the expert's operation, random noise is added to the executor, forcing the expert to perform remedial actions in an “ideal not condition”.
- **Effect**: This ensures that the training set contains a large amount of data on “how to recover from deviations”.

#### 3. Change the prediction target (Action Chunking / Diffusion Policy)

**Principle**: Do not just predict the action of the current frame, but predict a sequence of future actions.

- **Logic**: By predicting a trajectory (Action Chunk), the temporal correlation between actions can be utilized to offset the random disturbances of individual steps. The currently popular **Diffusion Policy** models complex action distributions using generative models, thereby significantly improving the tolerance for offsets.

#### 4. Adversarial imitation learning (such as GAIL)

**Principle**: Instead of directly learning actions, one learns "how to act like an expert".

- **Logic**: A discriminator is introduced to determine whether a state-action pair $(s, a)$ is performed by an expert. Through reinforcement learning, the agent aligns its entire trajectory with that of the expert, rather than merely learning the mapping per frame.

------

### III. Summary Comparison

| **Solution** | **Advantages**                        | **Disadvantages**                                        |
| ------------ | ------------------------------- | ----------------------------------------------- |
| **Pure BC**   | Simple, efficient, no environment interaction required. | **Fails due to covariant offset**, cannot handle long-distance tasks.          |
| **DAgger**   | Theoretically capable of solving $T^2$ error issues. | Requires experts to be online throughout (Human-in-the-loop), high cost. |
| **Noise Injection** | Low cost, no need for experts to correct repeatedly. | Difficulty in adjusting noise level; too much noise makes it impossible for experts to operate.            |
| **GAIL**     | High performance ceiling, strong generalization ability. | Extremely unstable training, requires a lot of environment interaction.                |



### 10. How are Expert Trajectories collected?

- **Core principle:** Multi-source heterogeneous collection. Utilize remote manipulation (VR controller/SpaceMouse), kinesthetic teaching (gravity-compensated dragging), or traditional planners based on OMPL. The collected data must include proprioceptive information with high-frequency timestamp alignment, visual streams, and control commands.

- **Implementation Pseudocode (Python - Construction of HDF5 Standard Dataset):**



  ```Python
  import h5py
  import numpy as np

  def save_expert_trajectory(states_obs, actions, joint_positions, file_path="expert.h5"):
      with h5py.File(file_path, 'w') as f:
          f.create_dataset('obs_images', data=np.array(states_obs), compression="gzip")
          f.create_dataset('actions', data=np.array(actions))
          f.create_dataset('proprioception', data=np.array(joint_positions))
  ```

### 11. Why install a six-dimensional force/torque sensor (F/T Sensor)?

- **Core principle:** Relying solely on motor current to estimate the contact force at the end is greatly affected by the friction of the reducer. The F/T Sensor installed on the flange plate can provide high-precision external contact force feedback, which is used for impedance/conductance control. This enables the robot to exhibit a "soft" behavior during rigid contact tasks such as plugging in connectors or cleaning, preventing collisions with objects.

- **Mathematical derivation:** Admittance control converts the contact force into a dynamic correction of the reference trajectory:

  $$M_d (\ddot{x} - \ddot{x}_d) + B_d (\dot{x} - \dot{x}_d) + K_d (x - x_d) = F_{ext}$$

This formula describes how we want the end of the robot to behave like a physical system with a specific mass, spring, and damping when subjected to external forces.

**Formula:**

$$M_d (\ddot{x} - \ddot{x}_d) + B_d (\dot{x} - \dot{x}_d) + K_d (x - x_d) = F_{ext}$$

#### Explanation of variable meanings:

- **$x, \dot{x}, \ddot{x}$**: The **actual** position, velocity, and acceleration of the robot end.
- **$x_d, \dot{x}_d, \ddot{x}_d$**: The **expected (nominal)** trajectory originally planned by the robot.
- **$e = x - x_d$**: Position deviation.
- **$M_d, B_d, K_d$**: **Virtual parameters** (mass, damping, stiffness) manually defined by us.
- **$F_{ext}$**: **External contact force** read by the six-dimensional force sensor.

#### Derivation logic:

1. **Input and Output**: In admittance control, **the input is the force ($F_{ext}$)**, and **the output is the displacement correction ($x$)**.
2. **Physical Simulation**: When the sensor detects $F_{ext}$, the algorithm does not directly control the force. Instead, it calculates: “If my end is an object with a mass of $M_d$, connected to a spring and damper $B_d$ with a stiffness of $K_d$, and subjected to this force $F_{ext}$, how much displacement **should** it produce $x$?”
3. **Trajectory Correction**: Solve this second-order differential equation to obtain a new reference position $x_{ref}$, which is then sent to the underlying position controller (such as PID).

### 3. Impedance Control vs Admittance Control: Essential Differences

This is a frequently asked question in robot interview tests. The difference between the two is **"who is the cause and who is the effect."**

| **Feature**     | **Impedance Control**                   | **Admittance Control**          |
| ------------ | -------------------------------------------------- | ------------------------------------------ |
| **Causal Relationship** | **Input displacement $\to$ Output force**  | **Input force $\to$ Output displacement** |
| **Basic Requirements** | The robot must have **torque control** (direct current/torque control) | The robot can have ordinary **position control** (encoder control) |
| **Mathematical Expression** | $F_{out} = Z(\Delta x)$                            | $\Delta x_{out} = A(F_{ext})$              |
| **Hardware Foundation** | Prefer joints with **high transparency and low friction** (e.g., direct drive motors) | Must have a **six-dimensional force sensor** at the end (F/T Sensor) |
| **Applicable Scenarios** | Emphasizes "like a spring", such as grinding and wiping | Emphasizes "moving smoothly", such as assisting with heavy objects or collaborative dragging |



Impedance Control and admittance control are "two sides of the same coin." In robot dynamics, the core idea of impedance control is: **to simulate the end of the robot as a controlled "spring-damping-mass" system**.

Unlike admittance control (force $\to$ displacement), the logic of impedance control is: **displacement deviation $\to$ outputs torque**.

------

### 1. Target Dynamics Equation

We hope that when the robot end is subjected to external disturbances, its behavior adheres to the characteristics of the following second-order linear system:

$$M_d (\ddot{x} - \ddot{x}_d) + B_d (\dot{x} - \dot{x}_d) + K_d (x - x_d) = f_{ext}$$

To simplify the derivation, we define the error term $e = x - x_d$ (the difference between the actual and expected positions), and the target dynamics equation is:

$$M_d \ddot{e} + B_d \dot{e} + K_d e = f_{ext}$$

#### Meaning of variables:

- **$x, \dot{x}, \ddot{x}$**: The actual pose, velocity, and acceleration of the end point in the task space.
- **$x_d, \dot{x}_d, \ddot{x}_d$**: The preset desired trajectory.
- **$M_d, B_d, K_d$**: Manually set **desired inertia, desired damping, desired stiffness**.
- **$f_{ext}$**: The external force applied by the environment on the end point.

------

### 2. Formula Derivation Based on Feedback Linearization

For a typical $n$-degree-of-freedom robotic arm, the dynamics equation in the joint space is:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau + J^T f_{ext}$$

Among them:

- $M(q)$: Inertia matrix.
- $C(q, \dot{q})$: Centrifugal force and Coriolis force terms.
- $g(q)$: Gravity term.
- $\tau$: Joint driving torque.
- $J$: Jacobian matrix.

#### Step 1: Map to task space

Using the relationship of the Jacobi matrix $\dot{x} = J\dot{q}$, differentiate to obtain $\ddot{x} = J\ddot{q} + \dot{J}\dot{q}$, and solve back for $\ddot{q}$:

$$\ddot{q} = J^{-1} (\ddot{x} - \dot{J}\dot{q})$$

#### Step 2: Calculate control moments $\tau$

To make the robot behave like a target system, we designed the control law $\tau$ to cancel the dynamic nonlinear terms of the robot itself (gravity, Coriolis force, etc.).

Let the desired terminal acceleration be $\ddot{x}_{ref}$. According to feedback linearization, the control law is usually written as:

$$\tau = M(q) J^{-1} (\ddot{x}_{ref} - \dot{J}\dot{q}) + C(q, \dot{q})\dot{q} + g(q) - J^T f_{ext}$$

#### Step 3: Substitute Impedance Characteristics

Substitute the target dynamics equation $M_d \ddot{e} + B_d \dot{e} + K_d e = f_{ext}$ to solve for the reference acceleration $\ddot{x}_{ref}$ we need:

$$\ddot{x} = \ddot{x}_d + M_d^{-1} (f_{ext} - B_d \dot{e} - K_d e)$$

Use this formula as the input for the controller. Ultimately, the end of the robotic arm will exhibit the “softness-hardness” defined by $K_d$ and the “glossiness” defined by $B_d$.

------

### 3. The Physical Nature of Impedance Control

Impedance control essentially involves adjusting the robot's **dynamic stiffness**.

- **Stiffness term $K_d (x - x_d)$**: Generates a spring-like restoring force. The larger $K_d$, the more "rigid" the robot is, and it is harder to push it away from the target position.
- **Damping term $B_d (\dot{x} - \dot{x}_d)$**: Consumes energy and prevents oscillations. The larger $B_d$, the more "sticky" the motion is, and the more stable it is when contacting objects.
- **Inertia term $M_d \ddot{e}$**: Determines how sensitive the robot is to acceleration, i.e., "how heavy it feels".

------

### 💡 "Pitfalls" in Industrial Implementation

1. **Torque control accuracy**: Impedance control relies on precise torque output $\tau$. If the motor's reducer friction is too high, or the current loop response is not fast enough, the effectiveness of impedance control will be greatly reduced. This is why robots with built-in joint torque sensors, such as **Franka Panda**, have much better force control than ordinary industrial arms.
2. **Stability margin**: If $K_d$ is set very high, and the robot control cycle is not fast enough (e.g., only 100Hz), the system will experience **instability jitter** due to discrete errors. In high-performance robotics scenarios, the control frequency typically needs to reach **1000Hz (1ms)** or higher.



### 12. How can the speed of global path planning be ensured in large-scale real-world scenarios where the grid cell size is very small?

- **Core principle:** High-resolution large maps cause the search space of traditional A* algorithms to explode. The three key methods used by industry to improve performance: 1. Introduce Weighted A* with a dilation factor; 2. Multi-resolution pyramid grid; 3. Abandon jump point search (JPS) for symmetric redundant paths.

- **Mathematical derivation:** The weighted A* heuristic cost function (breaks the global optimal guarantee, but offers a DFS-like exploration speed):

  $$f(n) = g(n) + \epsilon \cdot h(n) \quad (\epsilon > 1)$$

This is a very classic **robotic navigation and planning (Path Planning)** interview question. If you work in embodied AI or robotic control, whether it’s controlling a robotic arm to move through a surgical cavity filled with obstacles, or controlling a quadruped robot to find its way in a massive ruin, these are all major engineering challenges that must be overcome.

Let's visualize this "disaster scenario":

Suppose you want a robot to find its way in a huge warehouse of 100 meters x $\times$ 100 meters. To prevent it from hitting the tiny cables on the ground, you set the grid cell precision to 1 centimeter.

This means your map has **100 million grids**!

If the traditional A* algorithm is used, it will diffuse outward in a painful, slow process like ripples, covering tens of millions of cells along the way. By the time it finds a perfect path, even the daylily has withered (and the system memory is completely exhausted).

To address this "search space explosion" problem, the industry has been forced to use these three methods. We will break down their physical meanings and cleverness one by one:

------

#### The First Tool: Introduce the Weighted A* with a Expansion Coefficient (turning "water wave" into "spear")

This is where the core mathematical formula you listed applies. Let's first see how the traditional A* algorithm works.

**1. The Absolute Rationality of Traditional A* (But Also Rigid)**

The formula for traditional A* is:

$$f(n) = g(n) + h(n)$$

- $g(n)$: **Actual cost incurred** (the number of steps actually taken from the starting point to the current cell $n$).
- $h(n)$: **Estimated future cost** (heuristic function, such as the straight-line distance from the current cell to the goal).
- Traditional A* requires these two weights to be 1:1. This makes it extremely "water-like": if you move a little too far forward, $g(n)$ increases, and it feels pain, so it immediately turns around to search for the nearest cell to the starting point. As a result, its search range on the map appears as a **growing, huge circular wave**, searching a lot of meaningless side space.

**2. The "Foolish Philosophy" of Weighted A\***

The solution provided by the industry is extremely straightforward and effective. Simply multiply a expansion factor $\epsilon$ (such as $\epsilon = 5$) greater than 1 in front of $h(n)$:

$$f(n) = g(n) + \epsilon \cdot h(n) \quad (\epsilon > 1)$$

- **A drastic change in physical meaning:** Now, the influence of $h(n)$, which predicts future costs, has been amplified by 5 times!
- **Why delve down like DFS (Depth-First Search)?** When the algorithm reaches a crossroads, although taking two steps to the side ($g$ becomes larger) might be more secure overall, due to $\epsilon$, taking one step toward the destination ($h$ becomes smaller) yields **5 times greater benefits** in the formula!
- This makes the algorithm extremely “greedy” and “reckless”. It no longer probes in all directions like a water wave, but acts like a **spear (or like DFS, going straight to the end)**, desperately digging deep toward the destination!

**3. Costs and Benefits (Engineering Compromises)**

- **What was sacrificed?** It broke the “global optimality guarantee (Admissibility)”. Because it is too greedy, when encountering a U-shaped obstacle, it may get stuck inside and then make a detour. The path found in this case may be 5% longer than the absolute shortest path.
- **What was gained?** The search speed increases exponentially (possibly dozens or even hundreds of times faster). In the real implementation of embodied AI, **spending 10 milliseconds to find a suboptimal path that is 5% longer is definitely better than spending 5 seconds to find an absolutely perfect path.** This is the wisdom of the industrial world.

------

#### Second Axe: Multi-resolution pyramid grid (“First look at the fast path, then the detailed one”)

The spear that was used just now is fast, but blindly stabbing into 100 million 1-cm cells is still very difficult. What to do? **Imitate the way humans look at maps.**

If you are driving from Beijing to Guangzhou, you definitely won't use the highest precision of your mobile map right away to see how each street is laid out.

1. **Top level (low resolution, e.g., 1-meter large cells):** First, divide the map into extremely coarse 1-meter large grids. Running an A* algorithm on this tiny map of several dozen by several dozen units instantly defines a “macro corridor” from the starting point to the end point (e.g.: first take the Beijing-Hong Kong-Macao Expressway, then turn onto the Daguang Expressway).
2. **Bottom level (high resolution, 1-centimeter small cells):** With this “macro corridor” constraint, the high-precision A* algorithm at the bottom level **must not** search beyond the corridor to explore other unrelated areas! It is only allowed to carefully avoid small stones and cables on this extremely narrow restricted area.

Through this pyramid structure, the search space, which was originally a complete 100-meter square, is directly compressed into a very thin tube. The computational power consumption drops dramatically in a geometric progression.

------

#### Third Tool: Jump Point Search JPS (discarding symmetric redundancy of "instant movement")

(Jump Point Search, JPS) This is an optimization algorithm for grid maps that is extremely extreme but highly intelligent.

**1. The "obsessive-compulsive" behavior of traditional A* and symmetry redundancy**

Imagine a very empty corridor with no obstacles in front of you.

From the starting point to the destination, you can first take a step upward and then a step to the right; or you can first go to the right and then upward. In the grid, there are thousands of combinations of this zigzag path, and their distance costs are **exactly the same** (this is called symmetric redundancy).

The extremely rigid traditional A* will trigger OCD-like symptoms, forcing the computation and comparison of thousands of identical paths, wasting a huge amount of computational power.

**2. JPS's "Instant Travel" Rule**

The JPS algorithm is extremely intelligent. It introduces a rule that aligns with human intuition: **If the path ahead is smooth, rush forward with your eyes closed until you encounter a "decision-point turn" where you must make a choice!**

- It emits rays within the grid. If there is nothing in front, it **never** adds the cells along its path to the list for calculation (Open List).
- It will keep "jumping" until the ray hits an obstacle, or a **"Forced Neighbor"** is detected at the edge of the obstacle (in other words, if it doesn't stop and turn here, it will hit the wall or take a longer route).
- Only then does it record this crucial "corner point" as a node.

**Result:** In an open area, traditional A* requires 1,000 calculations for each step, while JPS acts like playing a portal—it instantly jumps from the starting point to the corner at the end of the corridor, skipping all 999 steps in between, thus eliminating the calculation load!

------

**Summarize this line:**

In the desperate scenario of 100 million small cells, industrial robots survive as follows:

Use the **multi-resolution pyramid** to reduce 100 million cells to 10,000 cells; within these 10,000 cells, use **JPS** to skip all the boring calculations on the flat ground; finally, for those complex terrains that require careful consideration, use **Weighted A*** with the expansion factor to brutally pierce a path through them like a spear!

These "three key techniques" can be regarded as a perfect example of balancing the mathematical limits of algorithms with real-world physical constraints.





### 13. What mathematical model does MPC specifically use?

- **Core principle:** A linear time-varying model based on kinematic error (LTV-MPC) is adopted. The nonlinear system is discretized using a Jacobian first-order Taylor expansion at the reference trajectory points, thereby transforming the control problem into a standard quadratic programming (QP) problem.

- **Mathematical derivation:** State space update model:

  $$\Delta x_{k+1} = A_k \Delta x_k + B_k \Delta u_k$$

  Among them, $A_k = \frac{\partial f}{\partial x} \big|_{x_r, u_r}$ and $B_k = \frac{\partial f}{\partial u} \big|_{x_r, u_r}$ are the system Jacobi matrix.

In embodied AI, after you issue a macroscopic path command, it is the MPC that determines exactly how a robotic arm or quadruped robot exerts force every millisecond, how it counteracts inertia, and how it precisely stops at the target point. This **LTV-MPC (Linear Time-Varying MPC)** is the ultimate solution used by industries such as Boston Dynamics and Tesla Optimus to balance "operating speed" and "control accuracy".

We directly break down this seemingly cold mathematical model into its components.

------

#### 1. Why is such complex mathematics necessary? (Prediction by the old driver)

Imagine you are driving:

- **Normal PID control:** The eyes focus only on the area 1 meter in front of the car's front. If it is too left, turn right; if too right, turn left. This is called "feedback control", which often causes the car to move back and forth on the road, and it cannot be driven fast.

- **MPC control:** The eyes are fixed 50 meters ahead. You would think, “There's a curve in the next 3 seconds. I need to slow down and fine-tune the steering wheel in advance.”

**The essence of MPC is to predict the system state over a future period at each moment, and determine the most appropriate action to take at that time.**

------

#### II. Breakdown of the core formula: $\Delta x_{k+1} = A_k \Delta x_k + B_k \Delta u_k$

Robotic arms in the real world (such as surgical robotic arms) are **nonlinear**. For example, when the joint rotates, the relationship between the angle and the end position involves a large number of $\sin$ and $\cos$ factors. Calculating nonlinear control directly is too slow, and a graphics card running at high temperatures would not be able to handle real-time 1000Hz control.

So we use **"linearization"** to save time, which is what you mentioned as **Taylor expansion**:

##### 1. What are $\Delta x$ and $\Delta u$? (Finding the differences)

- **$x_r, u_r$ (reference point):** This is the "ideal path" provided by the planning algorithm (such as A\* just now). For example: "At this moment, you should be at coordinate (1,1) with a speed of 0.5."
- **$\Delta x_k$ (state error):** The reality is harsh; it is the **difference** between your current actual position and the ideal position.
- **$\Delta u_k$ (control increment):** The force or torque you apply additionally to correct that difference.

##### 2. $A_k$ matrix: The "inertia" and "self-evolution" of the system

$$A_k = \frac{\partial f}{\partial x} \big|_{x_r, u_r}$$

- **Meaning:** If I do not manipulate the rod ($u$ remains unchanged), and it is simply because my current position is off ($x$ has an error), what will this deviation be like in the next second?
- **Intuitive Understanding:** It represents the **physical properties** of the robot. For example, a robotic arm will fall under gravity, or it will continue to move forward due to inertia. $A_k$ shows how the deviation **diffuses spontaneously** over time.

##### 3. $B_k$ Matrix: "Sensitivity" of Actions

$$B_k = \frac{\partial f}{\partial u} \big|_{x_r, u_r}$$

- **Meaning:** If I increase the motor current by one unit ($u$), how much will my position and speed ($x$) change?
- **Intuition:** It is the **“sensitivity of the steering wheel”**. Under different positions, the same torque produces different effects. $B_k$ tells you in real time: at this moment, if you push it, how much it will move.

------

#### III. Jacobi first-order Taylor expansion: The art of turning "curves" into "straight lines"

The "Jacobi expansion" you asked about is actually a kind of **"using straight lines instead of curves locally"** concept.

- **Physical Intuition:** The motion laws of the robot are a highly complex curve.
- **Mathematical Manipulation:** We only consider this step (first-order expansion). For a very short time, we assume that this small segment of the curve is a straight line.
- **Result:** This manipulation transforms the terrifying non-linear equation into a simple **matrix multiplication**. Matrix multiplication runs extremely fast on computers, which is why MPC can achieve millisecond-level responses.

------

#### IV. Final Form: Converted into a Quadratic Programming (QP) Problem

The big boss (control target) ultimately wants to achieve: **small errors ($\Delta x \to 0$) and minimal effort ($\Delta u$ — not too much, otherwise the motor won't handle it).**

Thus, MPC turned the problem into finding an optimal $\Delta u$ to minimize the following cost function:

$J = \sum (\text{误差}^2 \times Q + \text{控制力}^2 \times R)$

- **Why is it called "Quadratic Programming (QP)"?** Because there are quadratic terms in the formula.
- **Why is it easy to compute?** Because QP problems are mathematically **convex optimization** problems. They have a unique and definite global optimal solution, and the industry has very mature algorithms (such as OSQP, qpOASES) that can compute the answer in just a few hundred microseconds.

------

#### Summary:

1. **Reference point $(x_r, u_r)$:** The “ideal scenario” provided by the target navigation.
2. **Jacobi matrix $A_k, B_k$:** A “snapshot” of the robot’s physical properties (inertia, sensitivity) at this exact moment.
3. **Predictive model:** Using the $A, B$ matrix, predict where I will be 10 steps ahead if I exert force now.
4. **Quadratic programming:** Among dozens of predicted possibilities, select the sequence of movements that are most effortless and closest to the ideal scenario.
5. **Rolling optimization:** Execute only the **first step** from the calculated sequence of movements, and then take a new image, make a new prediction, and recalculate immediately (this is what is known as rolling time-domain control).





### 14. How are acceleration constraints and dynamic constraints constructed respectively?

- **Core principle:** In the MPC solver (such as OSQP), the capability limit of the actuator is transformed into a strict linear inequality constraint matrix. The acceleration constraint essentially limits the rate of change of control variables between adjacent time steps.

- **Mathematical derivation:**

  $$\Delta u_{min} \le \Delta u_k \le \Delta u_{max}$$

  Expanded into specific linear and angular acceleration limits:

  $$a_{min} \le \frac{v_k - v_{k-1}}{\Delta t} \le a_{max}, \quad -\alpha_{max} \le \frac{\omega_k - \omega_{k-1}}{\Delta t} \le \alpha_{max}$$

### 15. How are the Prediction Horizon and Step Size set?

- **Core principle:** This is a trade-off between computing power and control safety. The step size $\Delta t$ determines the control frequency (usually synchronized with the chassis at 0.05s~0.1s), and the prediction time domain $N_p$ determines the foresight distance. The core principle is: the total prediction time $T_p = N_p \times \Delta t$ must be greater than the maximum braking time of the robot at its highest speed.

- **Mathematical derivation:** In the cost function, the front term integrates the step state error of $N_p$, and the back term penalizes the control increment of $N_c$ steps:

  $$J = \sum_{i=1}^{N_p} \| \mathbf{x}_{k+i|k} - \mathbf{x}_{ref} \|_Q^2 + \sum_{i=0}^{N_c-1} \| \Delta \mathbf{u}_{k+i|k} \|_R^2$$



**The same mathematical variable plays a completely different role at different positions in the algorithm.**

$\Delta u$** Mathematically, they are the same thing (control increment $\Delta u$), but in the optimizer, their meaning and handling are completely different.**

We can imagine MPC as a "chained race". $\Delta u$ here is both the **"chains"** and the **"points deduction"**.

------

#### 1. In question 14: $\Delta u$ represents "hard constraints" (physical limits).

In the context of acceleration constraints, $\Delta u$ represents the **physical limits of the actuator (motor, hydraulic rod).**

- **Its meaning:** “How much current can my motor increase at most in one millisecond?” or “What is the maximum speed of my servo?”

- **How to construct:** In solvers such as OSQP, it is written as **inequality constraints (Constraints)**.

- **Mathematical Intuition:**

  $$\Delta u_{min} \le u_k - u_{k-1} \le \Delta u_{max}$$

It is like installing a "limiter" on the robot. If the calculated $\Delta u$ exceeds this range, the solver will **forcefully** push it back to the boundary.

- **Physical consequences:** If you do not set this constraint, MPC may issue a "instantaneous superhuman" command in order to achieve the goal as quickly as possible. As a result, the motor may burn out, the gears may break, or the system may lose control due to the fact that the acceleration cannot be achieved physically.

------

#### 2. In question 15: $\Delta u$ is the "penalty score" (pursuit of smoothness)

In the cost function $J$, $\Delta u$ is used to make the robot's movements **"elegant and smooth"**.

- **Its meaning:** “Although you can step on the accelerator hard, for each step, I deduct points from your total score.”

- **How it is constructed:** It is the **quadratic term (Quadratic Cost)** in the cost function, and its weight is controlled by the matrix $R$.

- **Mathematical Intuition:**

  $$J = \dots + \sum \| \Delta u_k \|_R^2$$

- If $R$ is set very high, the robot will become extremely "passive", with extremely slow and smooth movements.
- If $R$ is set very low, the robot will become very "aggressive". As long as it does not reach physical limits (the constraints of Problem 14), it will frequently and significantly change its movements.

- **Physical consequences:** Without this penalty term, even with physical constraints, the robot's movements may behave like "spasms," repeatedly jumping across the edge of limits (this phenomenon is called high-frequency oscillation). This not only consumes more power but also causes severe wear and tear on the mechanical structure.

------

#### 3. Summary Comparison: Two Faces of the Same $\Delta u$

| **Dimension**       | **Δu in Problem 14**                    | **Δu in Problem 15**                  |
| -------------- | -------------------------------------- | ------------------------------------ |
| **Role**       | **Hard Boundary (Fence)**                | **Cost/Tax (Tax)**                   |
| **Mathematical Form**   | Linear inequality constraints (Box Constraints) | Quadratic penalty term                 |
| **Purpose**       | **Survival**: Prevent physical damage, ensure dynamics are feasible | **Optimization**: Seek stability, power saving, comfort |
| **If violated** | Solver error or forced truncation (physically infeasible) | Only the total score decreases (physically feasible, but not optimal) |

------

#### 4. Additional explanations on the prediction time domain ($N_p$) and step size ($\Delta t$)

The $N_p$ and $\Delta t$ in question 15 you mentioned actually determine how far this "little brain" can see.

- **Step size $\Delta t$ (sampling frequency):**

  If $\Delta t$ is too large (low frequency), the robot responds slowly and is prone to hitting walls; if $\Delta t$ is too small (high frequency), the computational load will explode. In embodied AI, the frequency usually needs to match that of the underlying driver (e.g., 20Hz - 100Hz).

- **Prediction time domain $N_p$ (forward depth):**

  This is why it is mentioned as **"the time required for the maximum braking force"**.

- **Example:** If a quadruped robot runs at 2 m/s, it takes 1.5 seconds to stop completely. If the $N_p \times \Delta t$ you set has only a 0.5-second duration, then the robot is like a "myopic person with only half a meter of vision." By the time it sees the cliff, it is physically unable to stop within the remaining distance.

- **Core trade-offs:**

You need to find a balance between "seeing far ahead but unable to calculate accurately" and "calculating very quickly but being short-sighted". Currently, the industry practice is usually: **The control time domain $N_c$ is set to be short (e.g., 2–5 steps), while the prediction time domain $N_p$ is set to be longer (e.g., 20–50 steps)**. This ensures long-term safety while reducing the number of variables that need optimization.

### 16. Does angle tracking be included in the trajectory tracking process?

- **Core principle:** Must be included. The differential model operates in the SE(2) space; if only $(x, y)$ is tracked, the chassis will slide laterally. The key is not to use the absolute error of the world coordinate system directly, but to rotate it into the robot's local coordinate system.

- **Mathematical derivation:** Error transformation matrix:

  $$e_k = \begin{bmatrix} e_x \\ e_y \\ e_\theta \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_{ref} - x \\ y_{ref} - y \\ \theta_{ref} - \theta \end{bmatrix}$$

**The errors you see on the map are not those that the robot can directly handle.** Imagine this: you (the robot) have your eyes covered, and I command from behind you. I say, “Move 1 meter east!” (world coordinate system error). But you don’t know which side is east; all you know is your **“front, back, left, right”** (local vehicle coordinate system). Therefore, I must convert “1 meter east” into “move 0.7 meters forward and 0.7 meters to the right” from your perspective.

This matrix is that **"translator"**.

------

#### 1. Preparation: Defining Two Coordinate Systems

Before deriving, we need to draw two diagrams in our mind first:

1. **World coordinate system $\{W\}$ (Global Frame):** Fixed on the ground $X_W, Y_W$. The reference trajectory point $(x_{ref}, y_{ref}, \theta_{ref})$ and the robot's current position $(x, y, \theta)$ are both defined here.
2. **Robot coordinate system $\{R\}$ (Local/Body Frame):** Attached to the center of the robot. Its $x_R$ axis always points straight ahead, and the $y_R$ axis points to the left.

**Why convert?**

Because the control quantity of the differential chassis is **linear velocity $v$** (along $x_R$) and **angular velocity $\omega$** (around the center). If we directly use the error in the world coordinate system for control, the robot will act like a drunk person who can't distinguish left from right, resulting in severe slipping.

------

#### II. Step 1: Calculate the original error under "World Perspective"

This step is simple; it involves simply subtracting two coordinate points:

$$\Delta x_W = x_{ref} - x$$

$$\Delta y_W = y_{ref} - y$$

$$\Delta \theta = \theta_{ref} - \theta$$

This indicates how far the reference point is from you when viewed on the map. However, this direction is relative to the $X$ axis of the map, not relative to the robot's front.

------

#### III. Step 2: Two-dimensional Coordinate Rotation (Core Derivation)

We need to**rotate** this displacement vector $\begin{bmatrix} \Delta x_W \\ \Delta y_W \end{bmatrix}$ from the world coordinate system to the robot's current pose.

1. **Review the standard rotation matrix:**

    If you want to transform a point from the robot coordinate system $\{R\}$ to the world coordinate system $\{W\}$, the rotation matrix $R(\theta)$ is:

   $$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

(This matrix means: rotate the local coordinate axis by $\theta$ degrees to map it to the global coordinate system.)

2. **Reverse manipulation (global $\to$ local):**

   Now we have the global error, and want to find the local error. Therefore, what we need is the **inverse matrix** of $R(\theta)$ (for a rotation matrix, the inverse matrix is equal to the transpose matrix $R^T$):

   $$R(\theta)^{-1} = R(\theta)^T = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}$$

3. **Decomposition of physical meaning:**

  - **First line $(\cos\theta, \sin\theta)$:** Project the global error onto the robot's **front** ($x_R$ axis).
   - **Second line $(-\sin\theta, \cos\theta)$:** Project the global error onto the robot's **side** ($y_R$ axis).

------

#### IV. Step 3: Final Matrix Assembly

For the angle error $\Delta \theta$, since we are discussing a two-dimensional plane (SE(2) space), the change in angle does not vary with translation. Its projection is always 1:1.

By combining the 2x2 rotation matrix and the angle 1, you get the 3x3 transformation matrix you see:

$\begin{bmatrix} e_x \\ e_y \\ e_\theta \end{bmatrix} = \underbrace{\begin{bmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}}_{\text{旋转映射}} \begin{bmatrix} x_{ref} - x \\ y_{ref} - y \\ \theta_{ref} - \theta \end{bmatrix}$

------

#### 5. Why do this? (Summary of physical intuition)

- **$e_x$ (longitudinal error):** Indicates how much further forward (or backward) the robot needs to move. This error is mainly eliminated through **linear velocity $v$**.
- **$e_y$ (lateral error):** Indicates whether the robot has deviated to the left or right of the intended trajectory. **Key point:** A differential robot does not have a lateral motor, so it cannot move sideways like a crab.
- **How is $e_y$ eliminated?** The robot must rotate by a certain angle through **angular velocity $\omega$**, converting the lateral error $e_y$ into a longitudinal error $e_x$, and then move forward.

**If this matrix is not present:**

Your MPC or PID solver attempts to apply a "lateral force" to eliminate $e_y$, but the differential chassis physically cannot generate this force. As a result, the system experiences severe oscillations or spins in place.









### 17. Was there any oscillation or difficulty in keeping up with the target value during tracking?

- **Core principle:** It has occurred. **Oscillations** are mostly due to an unreasonable MPC weight allocation (too small $R$ resulting in overly aggressive movements), or the communication latency of the underlying system not being compensated in the state model. **Lag** occurs because the curvature of the reference trajectory exceeds the maximum physical acceleration capability of the chassis.

- **Implementation Pseudocode (C++ - System Delay Compensation Queue):**



  ```C++
  // 利用运动学模型，将发送但还未被执行的历史控制指令提前向前积分，抵消通信Delay
  State predictDelayCompensatedState(State current_state, deque<ControlCommand>& past_cmds) {
      State virtual_state = current_state;
      for (const auto& cmd : past_cmds) {
          virtual_state.x += cmd.v * cos(virtual_state.theta) * dt;
          virtual_state.y += cmd.v * sin(virtual_state.theta) * dt;
          virtual_state.theta += cmd.omega * dt;
      }
      return virtual_state; // 将补偿后的状态喂给MPC
  }
  ```

#### 1. Phenomenon Analysis: What are oscillation and delay?

1. **Oscillation —— “Drunk Driver”**

The robot behaves like a drunk person, repeatedly jumping sideways along the path, unable to stop.

   - **Reason A (too small weight $R$):** Remember we discussed the cost function of MPC before? $R$ is a penalty for “intensity of movement.” If $R$ is too small, the controller thinks “no effort is wasted,” resulting in extremely harsh commands. The robot turns sharply to the left, realizes it has gone too far, then turns sharply to the right, ultimately leading to oscillation.     * **Reason B (system delay):** This is the most insidious reason. The sensor tells you: “You are 1 cm to the left.” But due to communication delay, this refers to you **50ms ago**. You turn left based on this old information, but by then you have already returned to the center line. It’s like a game with high latency—you click the mouse, and the character moves only half a second later, resulting in repeated mistakes.

2. **Lagging behind (Lag) —— "The slow fat guy"**

The reference trajectory indicates 2 meters per second, but the robot can only move 1.5 meters; or the curve is too sharp for the robot to turn around.

- **Core reason:** Physical limits. Your algorithm calculates a perfect circle mathematically, but the torque of the motor or friction on the ground does not support this centripetal acceleration. This is often referred to as **"Dynamic Feasibility Not Met"** in interviews.

------

#### II. Core Algorithm: System Delay Compensation (Code Section)

The C++ code in the image provides an industrial-level solution: **Predictive Delay Compensation**.

##### 1. Physical logic: Since there is a delay, I "predict" your prediction.

Imagine you are targeting a moving target. You can't aim at its current position, but at the position it will be in **the next second**.

What this code does is: **“I know there is a delay in the data transmission, so based on the commands I sent in the past, I can calculate your real physical location right now.”**

##### 2. Translate the code line by line:



```C++
State predictDelayCompensatedState(State current_state, deque<ControlCommand>& past_cmds) {
    State virtual_state = current_state; // 1. 拿到“过时的”传感器位置
    for (const auto& cmd : past_cmds) {  // 2. 翻开“记事本”，看看在延迟期间我发了哪些指令
        // 3. 模拟物理模型，把这些指令“叠加”到旧状态上
        virtual_state.x += cmd.v * cos(virtual_state.theta) * dt;
        virtual_state.y += cmd.v * sin(virtual_state.theta) * dt;
        virtual_state.theta += cmd.omega * dt;
    }
    return virtual_state; // 4. 返回一个“虚拟的、当下的”状态喂给 MPC
}
```

- **Its ingenuity:** It performs a “simulated journey” in the computer. Since sensor data is delayed, we use the **kinematic model** (i.e., those addition operations) to complete this period in advance.
- **Mathematical basis:** This is the basic **dead reckoning**. By using the known linear velocity $v$, angular velocity $\omega$, and time $dt$, the coordinates are continuously accumulated.





### 18. Will there be "sharp turns" or planning failures in a crowded (dynamic environment)?

- **Core principle:** The static obstacle expansion algorithm fails when dealing with dynamic pedestrians. Since people are in motion, if applied using a static logic, the free space would be compressed to zero instantly, causing the algorithm to throw an exception (Freezing Robot) or generate a highly convoluted trajectory.

- **Mathematical derivation:** In the cost function of local planners such as DWA, a TTC (Time to Collision) penalty term for the relative speed of dynamic obstacles is introduced:

  $$Cost(v, \omega) = \dots + \delta \cdot \frac{1}{\text{TTC}(v, \omega)}$$

  Among them is $\text{TTC} = \frac{||\Delta P||}{||\Delta V|| \cos(\theta_{rel})}$. In this way, the controller will actively reduce its speed when predicting a possible collision in the future, rather than blindly turning the steering wheel to avoid it.
