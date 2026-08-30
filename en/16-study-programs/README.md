# Focused Study Programs

This chapter decomposes complete projects into focused study programs. Each program defines learning objectives, chapter order, practical entry points, and acceptance criteria for courses, study groups, and project reproduction.

## Program Index

| Program | Focus | Starting Point | Completion Evidence |
| --- | --- | --- | --- |
| [Damo Academy Study Program](./01-damo-academy-team-learning/README.md) | Embodied AI trends, technical routes, and demonstrations | Embodied AI overview | Explain the principal technical routes and complete a platform demonstration. |
| [D-Robotics Desktop Companion](./03-d-robotics-desktop-companion-tutorial/README.md) | Board access, peripherals, vision, speech, and device control | Linux command line and basic networking | Complete device login, peripheral verification, and one interactive function. |
| [OpenClaw Household Inventory Assistant](./02-openclaw-family-supply-assistant/README.md) | Household inventory, grasp simulation, notifications, and front-end presentation | Desktop companion or MuJoCo foundations | Run inventory entry, a grasp demonstration, and message synchronization. |
| [AMD ROCm Policy Reproduction](./04-amd-rocm-policy-replication-topic/README.md) | Policy training, evaluation, JAX migration, and simulation benchmarks on AMD hardware | LeRobot, MuJoCo, and imitation learning | Download, train, evaluate, and export video for one model. |

## Recommended Routes

For hardware applications, follow the desktop companion with the OpenClaw household assistant: connect the device first, then combine speech, vision, grasping, and notifications. For robot learning, complete the policy and simulation foundations before entering the AMD ROCm program. The Damo Academy program provides a broad technical overview and works well as an introduction.

The programs share concepts but not large runtime environments. Create an isolated environment for each program and keep model weights, datasets, caches, and batch video outside the textbook repository. The repository retains prose, lightweight scripts, configurations, metric summaries, and essential media.

## Common Acceptance Criteria

Preserve at least four outputs when completing a program:

1. Environment and device information, including key software versions and the compute device.
2. A repeatable entry command with input and output directories.
3. A readable result summary such as an evaluation table, training curve, or task log.
4. A video, screenshot, or interactive page that explains system behavior.
