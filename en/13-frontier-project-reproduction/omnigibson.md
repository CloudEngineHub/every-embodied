# OmniGibson Household Simulation Quick Start

OmniGibson is a robot simulation environment built on Isaac Sim and used by the BEHAVIOR-1K household-activity benchmark. This guide introduces its configuration structure and a minimal environment loop. Follow the [official installation guide](https://stanfordvl.github.io/BEHAVIOR-1K/omnigibson/getting_started/installation.html) for current system and asset requirements.

## Learning Goals

- Understand scene, object, robot, and task configuration blocks.
- Create an environment containing a Fetch robot and a primitive object.
- Step the environment and release simulator resources correctly.
- Identify when BEHAVIOR-1K scene and object assets are required.

## Repository and Environment

```bash
export PROJECT_ROOT=$HOME/projects/BEHAVIOR-1K
export DATA_ROOT=$HOME/datasets/behavior-1k

git clone https://github.com/StanfordVL/BEHAVIOR-1K.git "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
```

Complete the official installation and asset setup, then verify the import:

```bash
python -c "import omnigibson as og; print(og.__file__)"
```

## Configuration Structure

```python
import omnigibson as og

cfg = {
    "scene": {
        "type": "Scene",
        "floor_plane_visible": True,
    },
    "objects": [
        {
            "type": "PrimitiveObject",
            "name": "target_box",
            "primitive_type": "Cube",
            "rgba": [0.2, 0.7, 1.0, 1.0],
            "scale": [0.25, 0.25, 0.1],
            "fixed_base": True,
            "position": [1.0, 0.0, 0.1],
        }
    ],
    "robots": [
        {
            "type": "Fetch",
            "name": "fetch",
            "obs_modalities": ["rgb", "depth"],
        }
    ],
    "task": {
        "type": "DummyTask",
        "termination_config": {},
        "reward_config": {},
    },
}
```

The `scene` block selects the spatial environment, `objects` defines interactive assets and poses, `robots` selects the embodiment and sensors, and `task` defines rewards and termination.

## Create and Step the Environment

```python
env = og.Environment(cfg)

for _ in range(240):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

og.shutdown()
```

The environment should load a floor, the target object, and the Fetch robot. Random actions validate the observation, action, and physics interfaces; task evaluation requires an appropriate controller or policy.

## Asset-Backed Objects

`DatasetObject` resolves categories and models from BEHAVIOR-1K assets. Install the corresponding asset package before using it. Custom USD assets can be loaded with `USDObject` and `usd_path`.

## Verification Checklist

1. `import omnigibson` succeeds.
2. Environment creation reports no missing scene or object assets.
3. The sampled action shape matches the robot action space.
4. `env.step()` returns observations, reward, termination flags, and an information dictionary.
5. The program calls `og.shutdown()` before exiting.

## References

- [OmniGibson installation guide](https://stanfordvl.github.io/BEHAVIOR-1K/omnigibson/getting_started/installation.html)
- [OmniGibson quick start](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/docs/getting_started/quickstart.md)
- [BEHAVIOR-1K repository](https://github.com/StanfordVL/BEHAVIOR-1K)
