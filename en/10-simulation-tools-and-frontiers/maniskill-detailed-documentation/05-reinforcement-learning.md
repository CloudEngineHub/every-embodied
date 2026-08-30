# Reinforcement Learning

ManiSkill supports various reinforcement learning methods through a unified API, and provides multiple ready-to-use, tested baselines for use/comparison. The following page explains how to set up a reinforcement learning environment and how to use RL baselines. All baseline results are published on our [ public wandb page ](https://wandb.ai/stonet2000/ManiSkill). On this page, you can filter by algorithm used, environment type, etc. We are still conducting all experiments, so not all results have been uploaded yet.

## Settings

This page records the key points to be aware of when setting up the ManiSkill environment for reinforcement learning, including:

- How to convert the ManiSkill environment into a compatible gymnasium API environment, including [ single ](#gym-environment-api) and [ vectorized ](#gym-vectorized-environment-api) API.
- How to [**correctly** fairly evaluate RL policies ](#evaluation)
- [ useful wrappers ](#useful-wrappers)

The ManiSkill environment is created by the `make` function of gymnasium. By default, the result is a "batch" environment, where each input and output is a batch. Note that this is not the standard gymnasium API. If you want a standard gymnasium environment/vectorized environment API, please refer to the next section.

```python
import mani_skill.envs
import gymnasium as gym
N = 4
env = gym.make("PickCube-v1", num_envs=N)
env.action_space # shape (N, D)
env.observation_space # shape (N, ...)
env.reset()
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# obs (N, ...), rew (N, ), terminated (N, ), truncated (N, )
```

## Gym Environment API

If you want to use a CPU simulator or a single environment, apply `CPUGymWrapper`. Essentially, it cancels all batches and converts everything into numpy, so the behavior of the environment is like that of a regular gym environment. For detailed information on the API of gym environments, please refer to [ and its documentation ](https://gymnasium.farama.org/api/env/).

```python
import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
N = 1
env = gym.make("PickCube-v1", num_envs=N)
env = CPUGymWrapper(env)
env.action_space # shape (D, )
env.observation_space # shape (...)
env.reset()
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# obs (...), rew (float), terminated (bool), truncated (bool)
```

## Gym Vectorized Environment API

We also adopted the gymnasium `VectorEnv` (also known as `AsyncVectorEnv`) interface, which can be implemented through a single wrapper, so that the algorithm for the `VectorEnv` interface can work seamlessly. For detailed information on the API of the vectorized gym environment, please refer to [ and its documentation ](https://gymnasium.farama.org/api/vector/).

```python
import mani_skill.envs
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
N = 4
env = gym.make("PickCube-v1", num_envs=N)
env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
env.action_space # shape (N, D)
env.single_action_space # shape (D, )
env.observation_space # shape (N, ...)
env.single_observation_space # shape (...)
env.reset()
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# obs (N, ...), rew (N, ), terminated (N, ), truncated (N, )
```

You may also notice that there are two additional options when creating a vector environment. The `auto_reset` parameter controls whether the environment is automatically reset when the parallel environment is terminated or truncated. This depends on the algorithm. The `ignore_terminations` parameter controls whether the environment is reset when it is set to `terminated` as True. Similar to the gymnasium vector environment, some resets may occur, with some parallel environments being reset and others not.

Note that, for efficiency, all content returned by the environment will be batch torch tensors on the GPU, rather than batch numpy arrays on the CPU. This may be the only difference that needs to be considered between the ManiSkill vectorized environment and the gymnasium vectorized environment.

## Evaluation

Considering different types of environments, algorithms, and evaluation methods, we describe below a consistent and standardized approach to fairly evaluate all types of policies in ManiSkill. In summary, the following settings are required for fair evaluation:

- Some resets are disabled, and the environment does not reset during success/failure/termination (`ignore_terminations=True`). Instead, we record various types of success/failure metrics.
- All parallel environments are reconfigured upon reset (`reconfiguration_freq=1`), and if the task involves object randomization, the object geometry is randomized as well.

The following code demonstrates how to fairly evaluate policies and record standard metrics in ManiSkill. For GPU vectorized environments, the recommended code for evaluating policies by environment ID is:

```python
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
env_id = "PushCube-v1"
num_eval_envs = 64
env_kwargs = dict(obs_mode="state") # modify your env_kwargs here
eval_envs = gym.make(env_id, num_envs=num_eval_envs, reconfiguration_freq=1, **env_kwargs)
# add any other wrappers here
eval_envs = ManiSkillVectorEnv(eval_envs, ignore_terminations=True, record_metrics=True)

# evaluation loop, which will record metrics for complete episodes only
obs, _ = eval_envs.reset(seed=0)
eval_metrics = defaultdict(list)
for _ in range(400):
    action = eval_envs.action_space.sample() # replace with your policy action
    obs, rew, terminated, truncated, info = eval_envs.step(action)
    # note as there are no partial resets, truncated is True for all environments at the same time
    if truncated.any():
        for k, v in info["final_info"]["episode"].items():
            eval_metrics[k].append(v.float())
for k in eval_metrics.keys():
    print(f"{k}_mean: {torch.mean(torch.stack(eval_metrics[k])).item()}")
```

For CPU vectorized environments, it is recommended to use the following code for evaluation:

```python
import gymnasium as gym
from mani_skill.utils.wrappers import CPUGymWrapper
env_id = "PickCube-v1"
num_eval_envs = 8
env_kwargs = dict(obs_mode="state") # modify your env_kwargs here
def cpu_make_env(env_id, env_kwargs = dict()):
    def thunk():
        env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        # add any other wrappers here
        return env
    return thunk
vector_cls = gym.vector.SyncVectorEnv if num_eval_envs == 1 else lambda x : gym.vector.AsyncVectorEnv(x, context="forkserver")
eval_envs = vector_cls([cpu_make_env(env_id, env_kwargs) for _ in range(num_eval_envs)])

# evaluation loop, which will record metrics for complete episodes only
obs, _ = eval_envs.reset(seed=0)
eval_metrics = defaultdict(list)
for _ in range(400):
    action = eval_envs.action_space.sample() # replace with your policy action
    obs, rew, terminated, truncated, info = eval_envs.step(action)
    # note as there are no partial resets, truncated is True for all environments at the same time
    if truncated.any():
        for final_info in info["final_info"]:
            for k, v in final_info["episode"].items():
                eval_metrics[k].append(v)
for k in eval_metrics.keys():
    print(f"{k}_mean: {np.mean(eval_metrics[k])}")
```

The following metrics are recorded and explained below:

- `success_once`: Whether the task is successful at any point in the episode.
- `success_at_end`: Whether the task is successful on the last step of the episode.
- `fail_once/fail_at_end`: Similar to the above two metrics, but used for failures. Note that not all tasks have success/failure criteria.
- `return`: The total reward accumulated in the episode.

## Useful wrappers

RL practitioners often use wrappers to modify and enhance the environment. These are recorded in the wrappers section. Some commonly used ones include:

- RecordEpisode is used to record the video/transactional trajectory of rollouts.
- FlattenRGBDObservations is used to flatten the `obs_mode="rgbd"` or `obs_mode="rgb+depth"` observation results into a simple dictionary, which only contains the combined `rgbd` tensor and `state` tensor.

## Common Errors / Precautions

In old environments/benchmarks, people often used `env.render(mode="rgb_array")` or `env.render()` to obtain the image input for the RL agent. This is incorrect, as the image observations are directly returned by `env.reset()` and `env.step()`, and `env.render` is only used for visualization/video recording in ManiSkill.

For robot tasks, observation results typically consist of state information (such as robot joint angles) and image observations (such as camera images). When `obs_mode` is not `state` or `state_dict` (such as the posture of the ground object), all tasks in ManiSkill will specifically remove certain privileged state information from the observation results. Additionally, the image observation results returned by `env.reset()` and `env.step()` usually come from cameras located at specific positions to provide a good view for the task, enabling it to be solved.

## Baseline

We provided various benchmark lines and learned from rewards through online reinforcement learning.

As part of these benchmarks, we have established standardized reinforcement learning benchmarks that cover a wide range of difficulty levels (easy to solve for validation, but not saturated) and diversity in robot task types, including but not limited to classical control, dexterous manipulation, desktop manipulation, mobile manipulation, etc.

## Online Reinforcement Learning Benchmarks

List of online reinforcement learning benchmarks that have been implemented and tested. The result links will take you to the corresponding wandb page to view the outcomes. You can change the filters/views in the wandb workspace to view results with different settings (such as state-based or RGB-based training). Note that there are also benchmarks for reinforcement learning using demonstrations (offline RL, online imitation learning).

| Reference Line               | Code                                                                           | Result                                                   | Paper                                     |
| ------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------- |
| Proximal Policy Optimization (PPO) | [ link ](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo) | [ link ](https://api.wandb.ai/links/stonet2000/k6lz966q) | [ link ](http://arxiv.org/abs/1707.06347)  |
| Soft Actor-Critic (SAC)     | [ link ](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/sac) | WIP                                                  | [ link ](https://arxiv.org/abs/1801.01290) |
| Time-Difference Learning for Model Predictive Control (TD-MPC2) | WIP                                                                          | WIP                                                  | [ link ](https://arxiv.org/abs/2310.16828) |

## Standard Benchmark

The reinforcement learning standard benchmarks in ManiSkill consist of two sets: one with a small collection of 8 tasks, and the other with a large collection of 50 tasks. Both sets feature state-based and visual-based settings. All standard benchmark tasks include normalized dense reward functions. A recommended small collection has been created so that researchers without extensive computing resources can still perform reasonable benchmarking/comparison on their work. The large collection is still under development and testing.

These tasks cover a wide range of issues in reinforcement learning, such as high-dimensional observations/motions, large initial state distributions, manipulation of jointed objects, generalizable manipulation, movement manipulation, and motion.

**Small collection environment ID**:

PushCube-v1, PickCube-v1, PegInsertionSide-v1, PushT-v1, HumanoidPlaceAppleInBowl-v1, AnymalC-Reach-v1, OpenCabinetDrawer-v1

## Evaluation

To correctly evaluate the reinforcement learning policy, please refer to the evaluation section in the [ Reinforcement Learning Settings page ](#evaluation) to understand how this code is configured. All results reported in the results from the above link follow the same evaluation settings.
