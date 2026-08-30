<div align="center">
  <img src="assets/main.png" width="100%" alt="Every Embodied open textbook cover" />

  # Every Embodied: An Open Textbook for Embodied AI

  Learn robotics foundations, perception, control, simulation, data, policy training, evaluation, and system integration in one structured curriculum.

  [Read Online](https://datawhalechina.github.io/every-embodied/zh-cn/) · [Book Map](./BOOK_MAP.en.md) · [Minimal Example](./examples/README.md) · [简体中文](./README.md)
</div>

## About This Textbook

Every Embodied is designed for developers, students, and researchers who want a systematic path into embodied AI. The curriculum follows a concept–implementation–verification loop: establish robotics and perception foundations, study policy learning and navigation, build simulation and evaluation workflows, and finally connect the pieces through complete projects.

The repository contains four types of material:

- **Core textbook chapters** on robotics, perception, learning, manipulation, navigation, data, and evaluation.
- **Laboratory manuals** for hardware integration, simulation, data collection, training, evaluation, and deployment.
- **Project case studies** from competitions, study programs, world models, drones, and robot design.
- **Community archives** for previous study schedules, events, and publication material.

See the [Book Map](./BOOK_MAP.en.md) for prerequisites, recommended order, and role-specific learning paths.

## Quick Start

The minimal MuJoCo example creates a robot manipulation scene and executes a grasp trajectory. It is a practical check for Python, graphics, and physics-engine setup.

```bash
git clone --depth 1 https://github.com/datawhalechina/every-embodied.git
cd every-embodied

conda create -n embodied python=3.10 -y
conda activate embodied
pip install mujoco ruckig

python examples/01_hello_every_embodied_mujoco.py
```

The robot should approach, grasp, lift, and place the object. See the [minimal example guide](./examples/README.md) for command details and headless execution.

## Four-Volume Curriculum

| Volume | Goal | Topics |
| --- | --- | --- |
| I. Embodied AI Foundations | Build foundations in embodied systems, coordinate transforms, kinematics, control, vision, and 3D perception | [01](./en/01-embodied-ai-overview/README.md), [02](./en/02-robot-basics-control-and-hand-eye-coordination/README.md), [04](./en/04-computer-vision-and-3d-reconstruction/README.md) |
| II. Learning and Decision Making | Study deep learning, reinforcement learning, manipulation, VLA policies, navigation, data, and evaluation | [05](./en/05-deep-and-reinforcement-learning/README.md), [06](./en/06-manipulation-and-vla/README.md), [07](./en/07-robot-operation-and-motion-control/README.md), [08](./en/08-navigation-and-vln/README.md), [09](./en/09-data-and-benchmarks/README.md) |
| III. Systems and Simulation | Integrate hardware, configure simulators, collect data, train policies, evaluate systems, and use engineering tools | [03](./en/03-robot-hardware-lerobot-and-rdk-x5/README.md), [10](./en/10-simulation-tools-and-frontiers/README.md), [11](./en/11-auxiliary-tools/README.md), [21](./en/21-robot-design/README.md) |
| IV. Frontiers and Projects | Apply the complete workflow to competitions, focused studies, world models, drones, and integrated projects | [13](./en/13-frontier-project-reproduction/README.md), [15](./en/15-challenges/README.md), [16](./en/16-study-programs/README.md), [17](./en/17-world-models/README.md), [18](./en/18-drones/README.md) |

Interview review and references are collected in Topics [12](./en/12-interview-questions/README.md) and [14](./en/14-references/README.md). Study-program and publication archives are kept in Topics [19](./en/19-monthly-team-learning/README.md) and [20](./en/20-wechat-articles/README.md).

## Recommended Paths

### Beginner

`01 Overview → 02 Robotics Foundations → Minimal Example → 04 Perception → 07 Manipulation → 10 Simulation`

This path establishes observations, states, actions, control loops, and a first runnable manipulation task.

### Policy Learning

`02 Robotics Foundations → 05 Deep and Reinforcement Learning → 06 VLA Policies → 09 Data and Evaluation → 16 Focused Labs`

This path covers dataset structure, imitation learning, action chunking, training, closed-loop evaluation, and result analysis.

### Navigation and Mobile Manipulation

`02 Coordinates and Control → 04 Perception and Mapping → 08 Navigation and VLN → 09 Benchmarks → 10 Simulation`

This path connects localization, mapping, planning, language-conditioned navigation, and mobile manipulation.

### Systems Engineering

`03 Hardware and LeRobot → 10 Simulation → 11 Engineering Tools → 21 Robot Design → 15/16 Integrated Projects`

This path is intended for readers building hardware systems, collecting robot data, or porting complete open-source projects.

## Environment Guidance

Do not install every project into a single Python environment. Read the environment table in each chapter and create an isolated environment for each simulator or model stack.

Common combinations include:

- Python 3.10 and MuJoCo for robotics foundations and lightweight simulation.
- Isaac Sim and Isaac Lab for high-fidelity simulation and parallel training.
- LeRobot for data collection, policy training, and hardware interfaces.
- ROCm or CUDA according to the GPU requirements documented by each experiment.

Keep model weights, datasets, caches, and bulk video outputs outside the source repository. Pass their locations through environment variables.

## Contributing

Each laboratory chapter should state the working directory, environment, inputs, commands, expected outputs, and common failure modes. New material should follow the [tutorial style guide](./.github/TUTORIAL_STYLE_GUIDE.md) and include a reproducible minimal example whenever practical.

Please report broken links, commands, and technical issues through [GitHub Issues](https://github.com/datawhalechina/every-embodied/issues). The complete contributor history is available on the [contributors page](https://github.com/datawhalechina/every-embodied/graphs/contributors).

## License

The textbook and documentation are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You may share and adapt the material with attribution.
