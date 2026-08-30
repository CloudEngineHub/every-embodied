# 4. Environmental Camera and Debugging View

## 4.1 Can it be added?

It can be added, but only as a third-party perspective for local debugging and screen recording. It is suitable for answering:

- Whether the robotic arm has reached the search area.
- Whether the robotic arm collides with a wall or trash can.
- Whether the timing of the robotic arm’s movement conflicts with the base motion.
- Whether the target is outside the robotic arm’s view.
- Whether the failure occurs during searching, navigation, approaching, or placement.

## 4.2 How to Use It Is Not Allowed

Environmental cameras cannot:

- Formal observations written into the policy.
- The world coordinates of the target directly, for use by the policy.
- As additional input available during runtime in the training label.
- As a substitute for calibration of the head or end camera.
- As proof of success when there is no evidence from an official evaluator.

The judgment criterion is not whether "a camera is created in the code", but whether "the pixels of the camera affect the actual action". Debugging videos are allowed; as long as the camera image does not enter the calculation chain from observation to action, the policy input boundaries remain unchanged.

## 4.3 Recommended Three-Channel Video

1. **Head view**: Observe the long-distance search scene that the robot can actually see.
2. **End-effector view**: Observe the close-range relationship between the target and the gripper.
3. **Environment view**: Perform global troubleshooting for third parties only, without providing it to the policy.

The three-way video should share the same `episode_id`, step, or simulation timestamp to avoid mixing videos from different rounds into a successful case. When publishing, the non-policy attributes from the environmental perspective should also be indicated.

## 4.4 Minimum Verification

Perform an A/B check:

1. Execute the policy and record the action sequence `A`.
2. Turn off the environment camera while keeping the official observation unchanged, then run the same seed to record the action sequence `B`.
3. Compare `A` and `B`. If there are changes in the actions, it indicates that the environment camera has leaked into the policy through some path, and further investigation is required.

This is more reliable than stating in the video that "the environmental camera is just for observation."
