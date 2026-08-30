# RoboTwin Setup, Task Execution, and Evaluation Diagnostics

RoboTwin is a benchmark for dual-arm manipulation, demonstration generation, and policy evaluation. This chapter provides a reproducible path from environment checks and camera configuration to closed-loop evaluation and stage-level failure diagnosis.

- Repository: [RoboTwin](https://github.com/TianxingChen/RoboTwin)
- Audience: readers who already have a working Python environment and graphics driver
- Deliverables: one repeatable task launch, one fixed evaluation command, and one stage-level result record

## 1. Environment and directories

Represent source, asset, and output locations with environment variables rather than machine-specific paths:

```bash
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/RoboTwin}"
export ROBOTWIN_ASSETS="${ROBOTWIN_ASSETS:-$ROBOTWIN_ROOT/assets}"
export ROBOTWIN_OUTPUT="${ROBOTWIN_OUTPUT:-$ROBOTWIN_ROOT/outputs}"

cd "$ROBOTWIN_ROOT"
mkdir -p "$ROBOTWIN_OUTPUT"
```

After installing the dependencies from the upstream instructions, verify the core packages before launching a task:

```bash
python - <<'PY'
import importlib

for name in ("numpy", "torch"):
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'unknown')}")
PY
```

Run any environment check or task-listing utility provided by the current repository revision before data collection or evaluation. This separates dependency failures from policy failures.

## 2. Camera parameters

The `fovy` field is the vertical field of view in degrees. A larger value covers more of the workspace but reduces the number of pixels occupied by the target.

| Camera example | `fovy` | Typical effect |
| --- | ---: | --- |
| L515 | 45 | Wider workspace coverage |
| D435 | 37 | Larger target appearance at close range |

When adjusting cameras, verify that:

1. both arms, grippers, and target objects are visible in the initial frame;
2. no critical contact region is hidden during grasping, lifting, or placement;
3. training and evaluation use the same camera names, resolution, and field of view;
4. image preprocessing does not crop away the target or contact region.

## 3. Task execution sequence

Entry-point names may change between repository revisions. Inspect the current command help and build the evaluation path incrementally:

```bash
# 1. Inspect the current interface
python <task_entry.py> --help

# 2. Run one episode with a fixed task and seed
python <task_entry.py> \
  --task <task_name> \
  --seed 0 \
  --episodes 1 \
  --output_dir "$ROBOTWIN_OUTPUT/smoke"

# 3. Expand the denominator only after the single episode is stable
python <task_entry.py> \
  --task <task_name> \
  --seed 0 \
  --episodes 50 \
  --output_dir "$ROBOTWIN_OUTPUT/eval_seed0"
```

Replace `<task_entry.py>` and `<task_name>` with interfaces from the checked-out revision.

## 4. Evaluation records

A final success rate alone cannot identify where a dual-arm task failed. Store stage-level results for every episode:

```json
{
  "task": "task_name",
  "seed": 0,
  "episode": 0,
  "success": false,
  "stage": {
    "approach": true,
    "contact": true,
    "grasp": false,
    "lift": false,
    "place": false
  },
  "termination": "grasp_missed",
  "video": "episode_000.mp4"
}
```

A formal evaluation should preserve the configuration, per-episode results, and video index together.

## 5. Common failures

### 5.1 The gripper closes without holding the object

Check the gripper direction and range, command duration, end-effector coordinate frame, collision geometry, friction, and object mass.

### 5.2 The arm circles around the target

This often indicates action-normalization, control-period, or observation-timing errors. Record the raw action, denormalized action, joint target, and end-effector pose to locate the mismatch.

### 5.3 Replay works but closed-loop evaluation fails

Compare camera order, state-vector order, language instruction, action dimensions, and normalization statistics between training and evaluation.

## 6. Completion checklist

At the end of this chapter, you should be able to state:

- the task entry point, configuration file, and random seed;
- the camera names, resolution, and `fovy` values;
- the action mapping for both arms and grippers;
- the stage at which each failed episode terminated;
- the command that regenerates the evaluation results and videos.
