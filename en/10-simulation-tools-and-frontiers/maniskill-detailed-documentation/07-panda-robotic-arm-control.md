# Panda Robotic Arm Control

Read file: $HOME/17robo/ManiSkill/mani_skill/agents/controllers/__init__.py

Read file: $HOME/17robo/ManiSkill/mani_skill/agents/controllers/pd_joint_pos.py
The control method for the Panda robot gripper is as follows:

1. **Gripper joint name**:

   ```python
   gripper_joint_names = [
       "panda_finger_joint1",
       "panda_finger_joint2",
   ]
   ```

The Panda robot has two gripper joints, which are `panda_finger_joint1` and `panda_finger_joint2`.

2. **Controller configuration**:

   ```python
   gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
       self.gripper_joint_names,
       lower=-0.01,  # 一个技巧，当物体很薄时可以施加力
       upper=0.04,   # 完全打开的位置
       stiffness=self.gripper_stiffness,
       damping=self.gripper_damping,
       force_limit=self.gripper_force_limit,
   )
   ```

3. **Control method**:

- Panda uses `PDJointPosMimicController`, which is a Mimic controller
- This means both finger joints move synchronously; inputting a value controls both joints simultaneously
- The control range is from -0.01 (slightly closed) to 0.04 (fully open)
- In the initial posture, the gripper position is set to 0.04 (fully open):

     ```python
     keyframes = dict(
         rest=Keyframe(
             qpos=np.array([
                 0.0, np.pi/8, 0, -np.pi*5/8, 0, np.pi*3/4, np.pi/4,
                 0.04, 0.04,  # 夹爪两个关节，都设为完全打开
             ]),
             pose=sapien.Pose(),
         )
     )
     ```

4. **Object grasping detection**:
   The Panda robot uses `is_grasping` method to detect whether the object has been successfully grasped. This method is based on the contact force and angle.

Compared to GEN72-EG2 robot:

- Panda has only 2 gripper joints, while GEN72-EG2 has 4 gripper joints
- Both use the same `PDJointPosMimicController` controller, allowing the gripper joints to move synchronously
- The control range is similar: Panda ranges from -0.01 to 0.04, while GEN72-EG2 ranges from 0.0 to 0.04
- When in use, only one value needs to be entered to control all gripper joints simultaneously

Summary: During training, the grippers of both robots can be controlled to open and close by inputting a value within a single range of 0.0–0.04. 0.0 indicates full closure, while 0.04 indicates full opening.
