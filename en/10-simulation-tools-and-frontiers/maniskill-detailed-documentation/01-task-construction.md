# Building a Custom ManiSkill Task

A ManiSkill task combines environment registration, scene construction, episode initialization, success evaluation, and optional dense rewards. These interfaces should be clear before collecting demonstrations or starting reinforcement learning.

## Environment Lifecycle

| Interface | Purpose |
| --- | --- |
| `@register_env` | Register the environment name and default horizon |
| `_load_scene` | Build objects and sensors that are not part of the robot |
| `_initialize_episode` | Reset selected parallel environments and sample initial states |
| `evaluate` | Return success and diagnostic stages |
| `_get_obs_extra` | Add task-specific state observations |
| `compute_dense_reward` | Provide an optional continuous training signal |

## Minimal Skeleton

```python
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env


@register_env("MyPickTask-v1", max_episode_steps=200)
class MyPickTaskEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]

    def _load_scene(self, options: dict):
        # Build the table, target object, and goal marker.
        raise NotImplementedError

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Reset only the parallel environments selected by env_idx.
        raise NotImplementedError

    def evaluate(self):
        reached = self._reached_target()
        grasped = self.agent.is_grasping(self.target)
        lifted = self.target.pose.p[:, 2] > self.lift_height
        return {
            "reached": reached,
            "grasped": grasped,
            "lifted": lifted,
            "success": grasped & lifted,
        }
```

Return intermediate stages as well as `success`. Stage metrics reveal whether a policy failed during approach, contact, grasp, or lift, and they can be reused by a dense reward.

## Scene and Episode Initialization

Create geometry in `_load_scene`, then update existing object states in `_initialize_episode`. Rebuilding geometry on every reset harms parallel-environment performance and invalidates object indices. Use the environment random-number generator and respect `env_idx` so that fixed seeds and partial resets remain reproducible.

## Registration Check

```python
import gymnasium as gym
import my_pick_task

env = gym.make("MyPickTask-v1", num_envs=1, obs_mode="state")
obs, info = env.reset(seed=0)
for _ in range(20):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

Confirm that repeated resets with the same seed reproduce the initial state, `info` contains each stage metric, and every metric has one value per parallel environment.

## References

- [ManiSkill custom task tutorial](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/intro.html)
- [ManiSkill quick start](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)
