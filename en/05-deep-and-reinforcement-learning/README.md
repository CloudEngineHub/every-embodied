# Depth and Reinforcement Learning in Embodied Scenarios

This directory collects tutorial materials for the reproduction of reinforcement learning, imitation learning, and deep learning in embodied scenarios. It is recommended that you start by understanding the task boundaries and simulation platforms, and then move on to specific code and training commands.

## Chapter Entry

- [ Reinforcement learning for multi-robot furniture handling reproduction ](01-reinforcement-learning-for-multi-robot-furniture-handling.md): An introduction to the CooHOI collaborative humanoid robot handling task.
- [ HIMLoco quadruped robot motion control reproduction on Isaac Lab ](02-himloco-isaaclab-reproduction/README.md): Understanding from the paper, troubleshooting the old Isaac Gym stack, to the new Isaac Lab stack smoke test.
- [ AGILE humanoid robot Loco-Manipulation on Isaac Lab reproduction ](03-agile-humanoid-robot-loco-manipulation-reproduction/README.md): Understanding the AGILE humanoid robot RL workflow, official task boundaries, local Isaac Sim 5.1 replica video, and evaluation export process.
- [ Introduction to reward alignment for Robots That Know What to Ask ](04-robots-that-know-what-to-ask-reward-alignment-introduction/README.md): Analyzing how ASQ identifies under-specification reward features during teaching, and explains in natural language how to guide humans to provide corrective demonstrations.

## Learning Recommendations

If you are new to reinforcement learning for robots, it is recommended to first read the HIMLoco section to understand the basic concepts of proprioception, privileged critic, and sim-to-real; then read the AGILE section to learn about more comprehensive humanoid robot task configurations, teacher-student distillation, evaluation reports, and the Sim2MuJoCo review process. Students interested in VLA, world model post-training, or human demonstration data can continue with Robots That Know What to Ask, focusing on understanding reward misalignment, under-specified features, and human-in-the-loop reward correction.
