# ATEC2026 Task B Chinese Workspace Memory

Update time: 2026-07-31

## Purpose

This document contains the facts and boundaries of work for maintaining the Task B public tutorial later. It does not store private machine paths, credentials, model weights, or original competition logs.

## Confirmed Facts

- Task B is Garbage Collection, using the B2-Piper platform.
- The closed loop of this task includes navigation, as well as target search, approaching, robotic arm and gripper manipulation, grasping verification, handling, trash bin placement, and official testing confirmation.
- Official public observations mainly include proprioception, LiDAR, head RGB-D, and end RGB-D data.
- There is no evidence that a global third-party environment camera constitutes the default formal policy observation.
- The environment camera can record local videos, perform global troubleshooting, and debug alignment, but it cannot provide runtime target coordinates.
- The real pose of objects, object IDs, simulation internal object list, and tester internal state can only be used for debug-only checks.
- The approaching, touching, and contact agent metrics cannot replace physical grasping, lifting, and placement verification.

## The most valuable public solution for reference

- SLAM fork: [cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge](https://github.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge). It discloses navigation for Task B, end-effector camera locking, robotic arm calibration, classifier, and training documentation. This is the closest publicly available solution to a complete Task B structure.
- Clear fork: [StevenLiudw/Clear_ATEC2026_Simulation_Challenge](https://github.com/StevenLiudw/Clear_ATEC2026_Simulation_Challenge). It is more suitable as a reference for B2-Piper locomotion, squatting, and action mapping.
- Lhy fork: [Lhy6900/ATEC2026_Simulation_Challenge](https://github.com/Lhy6900/ATEC2026_Simulation_Challenge). The focus is on the G1/GR00T adapter; it cannot directly replace the B2-Piper Task B control interface.
- The links for ZSN2024 and yma867 have been registered in the higher-level [ public solution index](../references/README.md), respectively used for training skeletons and RGB-D perception references.

## Content that cannot be written as a conclusion

- The existence of a public demo does not equal the official final score.
- Self-reported rankings do not equal official leaderboards.
- “Touching” in the video does not mean the object is trapped.
- The environment camera seeing an object does not mean the policy legally sees the object.
- The joint order, camera parameters, action dimensions, and reward thresholds of other robots cannot be directly transferred.

## Subsequent Maintenance Order

1. Re-check the observation key, action space, runner, and scorer using the current checkout from the official repository.
2. Perform legal observation smoke tests in a single environment, followed by head search and end locking.
3. Validate separately the grasped relative pose, lifting height, and trash bin placement criteria.
4. Save commit, seed, environment version, official score, video, and failed stages for each experiment.
5. Check the LICENSE before adding external code; prioritize linked and fixed versions instead of directly copying the large repository.
6. Run `code/scan_public_release.sh` and `code/build_public_manifest.py` before submission.

## Release Boundaries

GitHub hosts tutorials, lightweight scripts, configurations, result reports, and external links; Datawhale HF contains a small public archive similar to the tutorials. Official assets, data, checkpoints, original videos, and private logs can only be placed in the public repository after obtaining explicit permission for redistribution.

## 2026-07-31 Public Archiving and Cleanup Status

- Datawhale HF: `Datawhale/atec2026-task-b-reproducibility`, the public branch includes the Task B document, scripts, media list, 5 videos, 4 screenshots, and official source code snapshots.
- Official source code snapshots: `source/ATEC2026_Simulation_Challenge_20260518.zip`, with a size of `431182199` bytes, and a SHA-256 value of `84ccfdc3903e4e03a5de8a7dedd90314b15ed09382c136bbeee6a858dae802d9`.
- The videos and screenshots are masked debug/evaluation previews; the top-down/global views use privileged trace, which is only used to explain behavior, not as formal policy inputs or official success proofs.
- The official player guide is archived in HF `official_docs/`; the historical Task E XSA plugin is archived in HF `historical_task_e/`.
- The HF archives can reproduction public tutorials, source code snapshots, and evidence readings; a complete simulation still requires the official matching version of Isaac Sim/Isaac Lab, GPU drivers, and Python environment. It should not be claimed as “just running directly after downloading HF”.
- The original ATEC working directory, official compressed package, guide copy, XSA copy, and local Task B video are deleted after remote SHA/size verification is passed; the original Workspace Memory is not uploaded, and this file is retained as a masked alternative.
