# 5. Comparison of Excellent External Solutions

The public fork network of the competition repository is the first entry for retrieving competing implementations: [ATEC official fork network](https://github.com/atecup/ATEC2026_Simulation_Challenge/network/members). Only the public repositories are referenced here; no external code is copied into this repository.

## 5.1 The closest solution to a complete Task B: SLAM fork

- Repository: [cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge](https://github.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge)
- Task B demo: [demo](https://github.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge/tree/main/demo)
- Task B documentation: [doc](https://github.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge/tree/main/doc)
- End-point camera locking: [task_b_ee_lock_pipeline.md](https://raw.githubusercontent.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge/main/doc/task_b_ee_lock_pipeline.md)
- Rapid movement training: [task_b_fast_locomotion_training.md](https://raw.githubusercontent.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge/main/doc/task_b_fast_locomotion_training.md)
- Object classifier: [task_b_object_classifier.md](https://raw.githubusercontent.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge/main/doc/task_b_object_classifier.md)

It exposes multiple segments of Task B scripts, including navigation, approach, robotic arm calibration, squatting, and end locking. Structurally, it is closest to a complete implementation for competition use. The parts that are worth migrating are:

| Module | Migratable ideas | Must be recreated before migration |
| --- | --- | --- |
| Search | Segmented waypoint, 360-degree head scan at each point | Map range, number of targets, speed, and termination conditions |
| Approach | Switch from the head camera to the end camera for locking after finding the target | Camera external parameters, depth units, and B2-Piper end coordinates |
| Close-range control | Visual alignment plus base fine-tuning, then crouch/extend arms after approaching | Gripper working space, ground height, and collision safety |
| Classification | Lightweight target category judgment for RGB crop | Target categories, data distribution, and official evaluation inputs |
| Movement | Separate speed policies for straight movement and approach | Training environment and official scoring behaviors |
| Debugging | Smoke test, data inspection, and single-target validation thresholds | Local GPU, seed, runner, and log directory |

Important limitation: The public documents and navigation scripts of this repository cannot automatically prove that the actual "grasping and putting into the trash can" has been completed. One default navigation route includes processing logic in the touch/service style, so during reproduction, it is necessary to verify gripping, lifting, and placement separately, rather than considering "touching the target" as a complete success.

## 5.2 Clear fork: Reference for motion and squat calibration

- Repository: [StevenLiudw/Clear_ATEC2026_Simulation_Challenge](https://github.com/StevenLiudw/Clear_ATEC2026_Simulation_Challenge)
- Main value: Locomotion, crouch, robotic arm actions, and policy export call examples for B2-Piper.

It is more suitable as a reference for underlying control and action mapping, and should not be described as a complete implementation of target searching, grasping, handling, and placement. During migration, verify the checkpoint path, joint order, action dimensions, and control frequency first.

## 5.3 Lhy fork: G1/GR00T adaptation reference

- Repository: [Lhy6900/ATEC2026_Simulation_Challenge](https://github.com/Lhy6900/ATEC2026_Simulation_Challenge)
- Main values: G1 robot action mapping, GR00T policy adapter, blind walking and speed command interface.

It is not the complete solution for B2-Piper Task B. The organizational approach of the policy adapter can be adopted, but the 29-dimensional action mapping of G1 cannot be directly replaced by the action interface of B2-Piper.

## 5.4 Judgment of Other Forks

Many repositories in the official fork network are identical to the official main branch or contain only experimental modifications. When searching, prioritize checking custom commits, Task B-specific documents, `demo/solution.py`, testing logs, and licenses, rather than just the repository name.

## 5.5 Licenses and Acquisition Methods

The official repository, Clear fork, and SLAM fork currently have a public LICENSE in MIT style. This article does not copy large files or complete code; it only retains the attribution, links, and download scripts. Run it when local research is required:

```bash
bash ../code/fetch_reference_repos.sh "${TMPDIR:-$HOME/.cache/atec2026-reference}"
```

The script will place the source code in a designated directory outside the repository, preventing external code from being accidentally submitted to the Datawhale tutorial repository.
