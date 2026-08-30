# Guide to Integrating GEN72-EG2 Robot with Stable PPO Version

This document describes how to use the GEN72-EG2 robot in the ManiSkill environment and train it using the stable PPO algorithm.

## 1. Introduction to the GEN72-EG2 Robot

GEN72-EG2 is a combination of a 7-degree-of-freedom robotic arm and an EG2 gripper. Its main features include:

- 7-degree-of-freedom robotic arm, with flexible motion
- Equipped with EG2 double-gripper, capable of grasping objects
- Physical parameters and controller optimized for ManiSkill environment
- Multiple initial poses pre-set for various tasks

## II. File Description

This integration includes the following files:

1. `register_gen72_robot.py` - Register GEN72-EG2 in the ManiSkill environment
2. `run_gen72_ppo.sh` - Scripts for training and evaluation
3. `ppo_my.py` - Stable PPO implementation (located in the upper directory)

URDF file location: `$HOME/17robo/ManiSkill/urdf_01/GEN72-EG2.urdf`

## III. Usage Steps

### 1. Register Robot

First, you need to register the GEN72-EG2 robot in the ManiSkill environment:

```bash
cd $HOME/17robo/ManiSkill/examples/baselines/ppo_my
python register_gen72_robot.py
```

### 2. Train and evaluate using Shell scripts

We provide convenient Shell scripts for training and evaluation:

```bash
# 赋予脚本执行权限
chmod +x run_gen72_ppo.sh

# 训练PushCube任务（稳定配置）
./run_gen72_ppo.sh train PushCube-v1 stable

# 预设的快速训练PushCube任务
./run_gen72_ppo.sh push-cube

# 预设的PickCube任务（稳定配置）
./run_gen72_ppo.sh pick-cube

# 评估训练好的模型
./run_gen72_ppo.sh evaluate PushCube-v1 ./runs/PushCube-v1__ppo_my__1__1234567890/final_ckpt.pt
```

### 3. Use the Python script directly

If more custom options are needed, you can directly use a Python script:

```bash
# 确保先注册了机器人
python register_gen72_robot.py

# 使用稳定版PPO训练
python ppo_my.py \
    --robot_uids="gen72_eg2_robot" \
    --env_id="PushCube-v1" \
    --control_mode="pd_joint_delta_pos" \
    --num_envs=128 \
    --total_timesteps=10000000 \
    --learning_rate=5e-5 \
    --max_grad_norm=0.25 \
    --update_epochs=4 \
    --num_minibatches=4 \
    --eval_freq=10
```

## IV. Training Configuration Instructions

Four preset training configurations are provided:

1. **Default configuration** (`default`)
   - Suitable for general training
   - Balanced performance and stability

2. **Stable configuration** (`stable`)
   - Improve numerical stability
   - Smaller learning rate and gradient clipping
   - Suitable to avoid NaN issues

3. **Ultra-stable configuration** (`ultra-stable`)
   - Extremely conservative parameter settings
   - Very small learning rate and gradient clipping
   - Efficiently runs on a single GPU
   - Suitable for severe numerical instability issues

4. **Quick Configuration** (`fast`)
   - Prioritize training speed
   - A large number of parallel environments
   - Suitable for rapid prototype validation

## 5. Physical Parameter Configuration

The physical parameters of GEN72-EG2 have been optimized, with the following key aspects:

- **Gripper friction**: A higher static and dynamic friction coefficient (2.0) is set to facilitate grasping of objects
- **Elastic coefficient**: Set to 0 to reduce the bounce effect
- **Control parameters**:
  - Robotic arm stiffness: 1e3
  - Robotic arm damping: 1e2
  - Robotic arm force limit: 100

## VI. Control Mode

Multiple control modes are supported:

1. `pd_joint_delta_pos` - Joint position increment control (recommended for RL)
2. `pd_joint_pos` - Joint position control
3. `pd_ee_delta_pos` - End effector position increment control
4. `pd_ee_delta_pose` - End effector pose increment control

## VII. Common Questions

1. **NaN issue**: If NaN occurs during training, try using the `stable` or `ultra-stable` configuration.
2. **Performance issue**: Adjust the `num_envs` parameter to adapt to GPU memory.
3. **Gripper control**: The gripper is controlled via simulated joints, with values ranging from 0.0 (closed) to 0.04 (open).

## VIII. Reference Resources

- ManiSkill document: https://maniskill.readthedocs.io/
- Original PPO paper: https://arxiv.org/abs/1707.06347