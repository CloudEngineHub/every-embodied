# Robot Simulation Platform Selection and Directory Navigation

This page is designed to help readers first determine “which simulation platform to learn” before proceeding to the corresponding installation or practice tutorials. Previously, there were separate installation notes for DISCOVERSE mixed in, which might lead readers to assume that the basic configuration of this chapter applies only to one project. Now, it has been changed to a comparison of simulation platforms and a guide for learning paths.

## Comparison of Common Simulation Platforms

| Platform | Primary Focus | Suitable Tasks | Position in This Project |
| --- | --- | --- | --- |
| MuJoCo | Lightweight, efficient, easy to control | Control, robotic arm, imitation learning, small closed loop experiments | [examples/README.md](../../examples/README.md), [README.md](../../README.md) |
| Robosuite + LIBERO | Combination of manipulation and VLA research under MuJoCo framework | manipulation, VLA, lifelong learning benchmark | [SmolVLA-LIBERO](../06-manipulation-and-vla/large-model-control-vla-vlm/01SmolVLA-LIBERO/01SmolVLA-libero.md), [LIBERO benchmark](../09-data-and-benchmarks/01-libero.md) |
| Isaac Sim | High-fidelity rendering and complex robot simulation | synthetic data, ROS2, Sim2Real, complex scenarios | [Isaac Sim, Isaac Lab and GR00T deployment guide](01-isaac-deployment-and-gr00t-practices/00-isaac-deployment-navigation.md), [Isaac Sim local and cloud configuration tutorial](01-isaac-deployment-and-gr00t-practices/01-isaac-sim-local-and-cloud-configuration.md) |
| Isaac Lab | Main RL/IL line under the Isaac ecosystem | RL, IL, policy training, task construction | [Isaac Sim local and cloud configuration tutorial](01-isaac-deployment-and-gr00t-practices/01-isaac-sim-local-and-cloud-configuration.md), [Alibaba Cloud deployment of Isaac Lab + GR00T full tutorial](01-isaac-deployment-and-gr00t-practices/02-complete-tutorial-on-deploying-isaac-lab-gr00t-in-alibaba-cloud.md) |
| Habitat-Sim / Habitat-Lab | Indoor navigation and VLN platform | PointNav, ObjectNav, VLN | [Habitat navigation environment](../../08-具身导航及VLN/02仿真环境基础/habitat导航环境) |
| ManiSkill | GPU parallel manipulation and RL platform | manipulation, PPO, data collection, task construction | [ManiSkill environment simulation configuration](02-maniskill-environment-simulation-configuration.md), [ManiSkill detailed documentation](../../10-具身智能其他仿真工具及仿真前沿/Maniskill详细文档) |
| GenieSim / GenieSim3 | High-fidelity manipulation simulation and benchmark | Pi0, policy evaluation, high-fidelity manipulation tasks | [GenieSim configuration](07-geniesim-configuration.md), [GenieSim3 configuration](08-geniesim3-configuration.md) |
| OmniGibson / BEHAVIOR | Long-distance household tasks and complex interactions | household, mobile manipulation, long-distance tasks | [household robot environment configuration](04-household-robot-environment-configuration.md) |
| Genesis | Emerging high-speed physics simulation platform | high-speed simulation, multi-physics fields, research exploration | [Genesis environment configuration](genesis-simulation-environment-configuration/01-environment-configuration-and-testing.md), [Genesis visualization and rendering](genesis-simulation-environment-configuration/02-visualization-and-rendering.md) |
| Gazebo | ROS/ROS2 system engineering verification | sensor simulation, mobile robot system integration | Not the main line in this repository, but still common in engineering pipelines |
| AirSim | Historical commonly used platform for drones | drones, visual navigation demo | [drone multi-modal large model practice](../../13-其他前沿项目复现/无人机大模型%2BGroundingdino实践/无人机多模态大模型.md) |
| DISCOVERSE | MuJoCo + 3DGS high-fidelity rendering + Real2Sim2Real | high-fidelity rendering, real scene reconstruction, Real2Sim2Real research | Suitable as an extended platform; project homepage: [DISCOVERSE](https://air-discoverse.github.io/) |
| SIM1 | Dual-armed soft body manipulation and data generation | fabric manipulation, teleoperation, replay, diffusion trajectory generation | [SIM1 soft body simulation and data generation](09-sim1-soft-body-simulation-and-data-generation/01-sim1-environment-configuration-and-operation.md) |

## Selection Recommendations

If the goal is to quickly understand robotic arm control, grasping, and basic policy reproduction, it is recommended to start with MuJoCo, Robosuite, LIBERO, or ManiSkill. These platforms have relatively low entry costs and are suitable for teaching and algorithm validation.

If the goal is high-fidelity rendering, synthetic data, ROS2 integration, or complex robot systems, it is recommended to learn Isaac Sim and Isaac Lab first. They require higher demands on graphics cards, drivers, and environment consistency, but they are closer to real-world engineering deployment.

If the goal is indoor navigation, language navigation, and path planning, focus on learning Habitat-Sim / Habitat-Lab. Drone navigation also falls into this category, as the core issues are similar: environmental understanding, target positioning, and path planning.

If the goal is a household robot, long-distance tasks, multi-object interaction, and home scenarios, focus on OmniGibson / BEHAVIOR or the household robot environment configuration.

If the goal is to research new platforms or explore cutting-edge technologies, you can continue to learn about extended platforms such as Genesis, DISCOVERSE, SIM1, RoboTwin, and MetaSim/RoboVerse.

## Recommended Learning Path

1. First, read this page to determine whether you want to focus on manipulation, navigation, system deployment, or research on cutting-edge platforms.
2. For the manipulation and reinforcement learning direction: Start with the simulation configuration of the [ManiSkill environment ](02-maniskill-environment-simulation-configuration.md), and then review the detailed documentation of [ManiSkill ](../../10-具身智能其他仿真工具及仿真前沿/Maniskill详细文档).
3. For the high-fidelity and system integration direction: Prioritize reading the deployment guide for [Isaac Sim, Isaac Lab, and GR00T ](01-isaac-deployment-and-gr00t-practices/00-isaac-deployment-navigation.md), and then choose the appropriate route based on whether you are using a local Windows/Linux workstation, cloud server, or GR00T to reproduce the target.
4. For the navigation direction: Focus first on [Habitat navigation environment ](../../08-具身导航及VLN/02仿真环境基础/habitat导航环境) and [ETPNav code reproduction ](../08-navigation-and-vln/03-frontier-vln-reproduction/01VLNCE/02-etpnav-code-reproduction.md).
5. For application practice directions: Select [home robot environment configuration ](04-household-robot-environment-configuration.md), [multi-modal large model practice for drones ](../../13-其他前沿项目复现/无人机大模型%2BGroundingdino实践/无人机多模态大模型.md), or [SIM1 soft body simulation and data generation ](09-sim1-soft-body-simulation-and-data-generation/01-sim1-environment-configuration-and-operation.md) according to the task.

## Writing Conventions

The subsequent documents in this chapter will try to follow the following conventions to avoid numbering errors in the web directory:

- Each Markdown file should contain only one first-level heading, which represents the theme of this article.
- File names determine the chapter order, such as `01...md` and `02...md`; the web sidebar should use the file name to generate the document title.
- Local numbering can be used within the text, such as `1. 安装` and `2. 快速入门`, but do not hardcode cross-file chapter numbers into the text titles.
- Platform comparisons and selection instructions are placed on this page; specific installation commands are included in the configuration documents of each platform.
- External projects like DISCOVERSE are only described in the comparison table; they are not included in the chapter entry page unless a complete tutorial is added later.
