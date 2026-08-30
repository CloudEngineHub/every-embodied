# Gripper Symmetric Control

In reinforcement learning training, the left-right symmetric opening and closing of the Franka Panda gripper are ensured by `PDJointPosMimicController` (imitation controller). The working principle of this special controller is as follows:

1. **Single control parameter**:
   Although Panda has two gripper joints `panda_finger_joint1` and `panda_finger_joint2`, the mimic controller allows both joints to share the same control signal. A single value is sufficient to control the opening and closing of the entire gripper.

2. **Code implementation**:

   ```python
   class PDJointPosMimicController(PDJointPosController):
       def _get_joint_limits(self):
           joint_limits = super()._get_joint_limits()
           diff = joint_limits[0:-1] - joint_limits[1:]
           assert np.allclose(diff, 0), "Mimic joints should have the same limit"
           return joint_limits[0:1]
   ```

This code indicates that the imitation controller only needs to know the constraints of the first joint, and then the same control signal will be applied to all joints.

3. **Controller configuration**:

   ```python
   gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
       self.gripper_joint_names,  # 包含两个夹爪关节
       lower=-0.01,  # 最小位置（稍微闭合）
       upper=0.04,   # 最大位置（完全打开）
       ...
   )
   ```

4. **Simplified action space**:

   - Although two joints are physically controlled, only one dimension is required in the action space.
   - For example, when input is 0.04, both joints will move to the 0.04 position (fully open).
   - When input is 0.0, both joints will move to the 0.0 position (fully closed).

Advantages of this design:

- **Simplified learning**: Reinforcement learning algorithms only need to learn to control one parameter, rather than controlling two fingers separately
- **Ensuring symmetry**: No matter what value is input, the gripper always opens and closes symmetrically on both sides
- **Accommodating physical reality**: The real Franka Panda gripper is designed to move synchronously and symmetrically in terms of its physical structure

In the GEN72-EG2 robot, the same control method is used. Although there are 4 gripper joints, they can maintain synchronized symmetric motion through `PDJointPosMimicController`.
