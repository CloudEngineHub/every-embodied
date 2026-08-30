# Isaac Sim, Isaac Lab, and GR00T Deployment Guides

This directory consolidates all related content about Isaac into a single entry, avoiding the need for users to jump between documents such as "Local Installation", "Cloud Server Deployment", "Isaac Lab", and "GR00T". In actual replication, the key differences between a local workstation, a school server, and Alibaba Cloud GPU instances do not lie in the Isaac commands themselves, but rather in GPU drivers, memory, remote desktop, Docker permissions, and disk space.

If you just want to run it on your own Windows computer, first refer to the local and cloud configuration tutorials in this directory; if you already have a cloud server and your goal is to reproduction GR00T, then proceed to the original Alibaba Cloud practice documents.

## Document Entry

| Learning Objectives | Recommended Documents |
| --- | --- |
| Install Isaac Sim locally on Windows, open the GUI, and run a robot scenario | [Isaac Sim Local and Cloud Configuration Guide](01-isaac-sim-local-and-cloud-configuration.md) |
| Install Isaac Sim on a Linux workstation or cloud server, and configure Docker / remote desktop / Isaac Lab | [Isaac Sim Local and Cloud Configuration Guide](01-isaac-sim-local-and-cloud-configuration.md) |
| Manage Isaac Sim / Isaac Lab environment using micromamba, venv, or pip | [Isaac Sim Local and Cloud Configuration Guide](01-isaac-sim-local-and-cloud-configuration.md) |
| Reproduction of Isaac Lab + GR00T old version links on Alibaba Cloud A10 instance | [Complete Tutorial for Deploying Isaac Lab + GR00T on Alibaba Cloud](02-complete-tutorial-on-deploying-isaac-lab-gr00t-in-alibaba-cloud.md) |
| Directly read GR00T code, data format, inference, and fine-tuning instructions | [Isaac-GR00T Project Description](../Isaac-GR00T/README.md) |

## Recommended Order

Students who are new to Isaac should not try to install Isaac Sim, Isaac Lab, ROS2, GR00T, and Docker all at once from the start. It is recommended to first achieve a minimal closed loop: ensure that `nvidia-smi` can access the NVIDIA GPU, then launch Isaac Sim using Workstation or pip, and create a Simple Room and a Franka robotic arm in the GUI. This closed loop proves that the driver, graphical interface, Isaac extensions, and basic asset links are working properly.

After Isaac Sim starts up smoothly, choose the subsequent path based on the goal. Those engaging in reinforcement learning or large-scale parallel tasks should continue to install Isaac Lab; those conducting ROS2 system integration should configure ROS2 Bridge; and those working on GR00T should proceed with the original cloud tutorial or GR00T project directory. This way, when problems occur at each step, the scope of troubleshooting is smaller.

## Version Relationship

This document uses the official installation methods of Isaac Sim 5.1 and Isaac Lab 2.3.x for reproduction. The Alibaba Cloud GR00T tutorial in the repository is locked to Isaac Sim 4.2.0, Isaac Lab v1.4.1, GR00T N1.6, PyTorch 2.5.1, and CUDA 12.1 for historical reproduction. These two installation methods should not be mixed in the same Python environment.

In principle, Isaac Sim, Isaac Lab, PyTorch, and GR00T should have their environments separated by project. NVIDIA drivers, Docker image layers, pip/mamba caches, model weights, and data directories can be shared, but do not mix Python packages from multiple Isaac versions into the same `site-packages`.
