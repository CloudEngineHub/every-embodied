# ATEC2026 Online Competition: Review and Reproduction Guide

This directory serves as the unified entry for ATEC2026 in Chapter 15 of Every Embodied, containing task definitions, L0 task practices, Task B garbage collection review, Task D public solution analysis, Task E desktop organization reproduction, lightweight code, and public resource descriptions.

This guide organizes a real embodied AI competition into task definitions, policy training, evaluation diagnostics, and reproducible release materials. Each result states its evidence source and intended scope.

## Directory Structure

```text
15-Challenge竞赛/ATEC2026/
├── README.md
├── L0-机器人徒步/                 # Task A / B2Piper locomotion 复现与评估
├── L0-桌面整理TaskE/              # Task E / Piper 桌面抓取与 ACT 复盘
├── TaskB-B2Piper/                 # Task B / B2-Piper 垃圾收集专题
├── docs/                          # Task A/B/D/E 共性边界与发布文档
├── code/                          # 轻量 adapter、参考仓库审计和检查脚本
└── references/                    # 外部公开方案、固定 commit 与许可证索引
```

## Task Overview

| Task | Robot/Ability | Entry Point of This Directory | Public Conclusion Boundaries |
|---|---|---|---|
| Task A / L0 | B2Piper Off-road Hiking | [`L0-机器人徒步/`](../../../15-Challenge竞赛/ATEC2026/L0-机器人徒步/) | Official baseline, local evaluation, trajectory tracking, and checkpoint selection |
| Task B | B2-Piper Garbage Collection | [`TaskB-B2Piper/`](../../../15-Challenge竞赛/ATEC2026/TaskB-B2Piper) | Public complete debugging and observation boundaries; no fully verified final policy |
| Task D | Box Pushing Over Obstacles | [`docs/03_公开方案与Logic-TARS.md`](docs/03-public-scheme-and-logic-tars.md) | Focus on the public Logic-TARS state machine, LiDAR, and action adaptation |
| Task E / L0 | Piper Desk Organization | [`L0-桌面整理TaskE/`](../../../15-Challenge竞赛/ATEC2026/L0-桌面整理TaskE/) | ACT seed1 best reached an online return of `15.00`, with training and evaluation notes |

## Recommended Reading

1. [ Task definition and observation boundaries ](docs/01-task-definition-and-observation-boundaries.md)
2. [ Problem list and solutions ](docs/02-problem-list-and-solutions.md)
3. [ Public solution and Logic-TARS interpretation ](docs/03-public-scheme-and-logic-tars.md)
4. [ Reproduction, HF, and GitHub release ](docs/04-reproduction-hf-and-github-release.md)
5. [ Archiving reconciliation and local cleanup ](docs/05-arrangement-of-reconciliation-and-local-cleanup.md)
6. [ Task B special archive ](TaskB-B2Piper/README.md)

## Public Resources

- Official simulation repository: [atecup/ATEC2026_Simulation_Challenge](https://github.com/atecup/ATEC2026_Simulation_Challenge)
- B2Piper L0 model package: [Datawhale/atec2026-b2piper-l0](https://huggingface.co/Datawhale/atec2026-b2piper-l0)
- Task B public reproduction archive: [Datawhale/atec2026-task-b-reproducibility](https://huggingface.co/datasets/Datawhale/atec2026-task-b-reproducibility)
- Task E model: [Datawhale/atec2026-task-e-act-seed1-best](https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best)
- Task E data and logs: [Datawhale/atec2026-task-e-reproducibility](https://huggingface.co/datasets/Datawhale/atec2026-task-e-reproducibility)

## Reproduction Agreement

- GitHub only stores tutorials, lightweight code, experimental boundaries, fixed external commits, and masked Workspace Memory.
- Large datasets, model weights, official source snapshots, videos, and logs are stored on Hugging Face according to their licenses.
- Complete simulation still requires matching Isaac Sim/Isaac Lab, GPU, Python dependencies, official runner, and evaluator; GitHub tutorials cannot be used as a one-click environment.
- Environment cameras, truth values, oracle prefix, script grippers, and phase state machines can be used for diagnosis, but protocol boundaries must be indicated; they cannot be used as formal policy inputs or complete success evidence.
- Raw training caches, private logs, tokens, server credentials, and unauthorized SDKs do not enter the public repository.
