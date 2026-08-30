# Evaluate Evidence Boundaries

## What can be proven

- The official task definitions and observation boundaries can be verified by the official repository.
- The public reference scheme indeed includes documentation on search, navigation, end-locking, and classification for Task B, which can be verified via the public GitHub link.
- Local third-party environment cameras are suitable for observing overall behavior, but they are not part of the policy input.
- Indicators for proximity, touch, and contact can be used for fault diagnosis.

## There is no proof possible based on these alone

- A screenshot of the environmental perspective cannot prove that only legal sensors were used in the policy.
- The robot approaching the object cannot prove successful grasping.
- The gripper closing cannot prove the object is held.
- The temporary movement of the object cannot prove it has been placed in the trash can.
- The presence of demo code in the external warehouse cannot prove that the final result or a complete closed loop has been confirmed by the official tester.

## Minimum evidence of complete success

1. The official runner and evaluator versions can be traced back.
2. The policy inputs in the running logs come only from valid observations.
3. After the gripper closes, the object is lifted by the end and held.
4. The carrying object reaches the trash can and is released.
5. After release, the object meets the official placement criteria.
6. Save the original video, step-by-step log, seed, and official score.
