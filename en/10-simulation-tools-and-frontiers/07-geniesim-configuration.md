# Genie Sim Setup, Benchmark Evaluation, and Trajectory Replay

Genie Sim Benchmark is AgiBot World's simulation and evaluation stack for embodied manipulation. This chapter organizes asset download, container launch, benchmark execution, policy integration, teleoperation, and replay into one reproducible workflow.

- Documentation: [Genie Sim Benchmark](https://agibot-world.com/sim-evaluation/docs/)
- Assets: [GenieSimAssets](https://huggingface.co/datasets/agibot-world/GenieSimAssets)
- Policy repository: [AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World)
- Issue tracker: [AgibotTech/genie_sim issues](https://github.com/AgibotTech/genie_sim/issues)

> The current upstream stack depends on Isaac Sim and NVIDIA Container Toolkit. Check the official hardware and driver requirements before installation.

## 1. Directory convention

Use environment variables for source, assets, and outputs:

```bash
export GENIESIM_ROOT="${GENIESIM_ROOT:-/path/to/genie_sim}"
export SIM_ASSETS="${SIM_ASSETS:-$HOME/GenieSimAssets}"
export GENIESIM_OUTPUT="${GENIESIM_OUTPUT:-$GENIESIM_ROOT/output}"

mkdir -p "$SIM_ASSETS" "$GENIESIM_OUTPUT"
cd "$GENIESIM_ROOT"
```

## 2. Download assets

Genie Sim stores large scenes and objects with Git LFS:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install

git clone https://huggingface.co/datasets/agibot-world/GenieSimAssets "$SIM_ASSETS"
cd "$SIM_ASSETS"
git lfs pull

find "$SIM_ASSETS" -type f | wc -l
du -sh "$SIM_ASSETS"
```

For an existing checkout, run `git pull` and `git lfs pull` in place instead of cloning a duplicate.

## 3. Container runtime

Follow the [Isaac Sim container guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html) to install Docker and NVIDIA Container Toolkit. Verify device access before building Genie Sim:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

Build the image from the repository root:

```bash
cd "$GENIESIM_ROOT"
docker build \
  -f ./scripts/dockerfile \
  -t registry.agibot.com/genie-sim/open_source:latest \
  .
```

If the build requires a proxy, scope it to that build command:

```bash
export DOCKER_HOST_IP="${DOCKER_HOST_IP:-172.17.0.1}"

docker build \
  --add-host="host.docker.internal:$DOCKER_HOST_IP" \
  --build-arg http_proxy="http://host.docker.internal:7890" \
  --build-arg https_proxy="http://host.docker.internal:7890" \
  --build-arg no_proxy="localhost,127.0.0.1" \
  -f ./scripts/dockerfile \
  -t registry.agibot.com/genie-sim/open_source:latest \
  .
```

Do not retain proxy variables that intercept local server-client traffic during simulation.

## 4. Start the server and a minimal task

Start the graphical container:

```bash
cd "$GENIESIM_ROOT"
SIM_ASSETS="$SIM_ASSETS" ./scripts/start_gui.sh
./scripts/into.sh
```

Inside the container, start the simulator server:

```bash
omni_python server/source/genie.sim.lab/raise_standalone_sim.py \
  --enable_curobo True
```

After the server reports that the application is ready, enter the container from a second terminal and run the stock-restocking example:

```bash
omni_python benchmark/task_benchmark.py \
  --task_name curobo_restock_supermarket_items \
  --env_class DemoEnv
```

Some challenge tasks require a policy or teleoperation client. Scene loading and time-step output alone do not imply that a task controller is active.

## 5. Unified task entry point

The `autorun.sh` wrapper provides the main operating modes:

```bash
./scripts/autorun.sh <TASK_NAME>
./scripts/autorun.sh <TASK_NAME> keyboard
./scripts/autorun.sh <TASK_NAME> pico <HOST_IP>
./scripts/autorun.sh <TASK_NAME> replay <STATE_FILE_PATH>
./scripts/autorun.sh <TASK_NAME> infer
./scripts/autorun.sh clean
```

Representative task names include:

| Setting | Task |
| --- | --- |
| Cafe | `genie_task_cafe_espresso` |
| Cafe | `genie_task_cafe_toast` |
| Home | `genie_task_home_clean_desktop` |
| Home | `genie_task_home_collect_toy` |
| Home | `genie_task_home_microwave_food` |
| Home | `genie_task_home_open_drawer` |
| Home | `genie_task_home_pass_water` |
| Home | `genie_task_home_pour_water` |
| Home | `genie_task_home_wipe_dirt` |
| Supermarket | `genie_task_supermarket_cashier_packing` |
| Supermarket | `genie_task_supermarket_stock_shelf` |
| Supermarket | `genie_task_supermarket_pack_fruit` |

Press `q` or `Q` to stop a task cleanly. Periodically review recordings and communication logs under the output directory.

## 6. Benchmark evaluation

Run a single task before expanding the evaluation set:

```bash
cd "$GENIESIM_ROOT"
SIM_ASSETS="$SIM_ASSETS" ./scripts/start_gui.sh
./scripts/autorun.sh genie_task_home_pour_water infer
```

ADER expresses task progress as ordered actions and termination conditions. A compact example is:

```json
{
  "Acts": [
    {
      "ActionList": [
        {
          "ActionSetWaitAny": [
            {"Follow": "beverage_bottle_002|[0.2,0.2,0.2]|right"},
            {"Timeout": 120},
            {"Onfloor": "beverage_bottle_002|0.0"}
          ]
        },
        {
          "ActionSetWaitAny": [
            {"PickUpOnGripper": "beverage_bottle_002|right"},
            {"Timeout": 120}
          ]
        },
        {
          "ActionSetWaitAny": [
            {"Inside": "beverage_bottle_002|handbag_000|1"},
            {"StepOut": 1000}
          ]
        }
      ]
    }
  ],
  "Problem": "pack_in_the_supermarket"
}
```

Common conditions are:

| Condition | Purpose |
| --- | --- |
| `ActionList` | Execute child actions in order |
| `ActionSetWaitAny` | Finish when any child condition is met |
| `ActionSetWaitAll` | Finish when all child conditions are met |
| `Timeout` | Terminate by elapsed time |
| `StepOut` | Terminate by simulation steps |
| `Follow` | Check whether a gripper enters a target region |
| `PickUpOnGripper` | Check whether the object is grasped |
| `Inside` | Check whether an object is inside a container |
| `Ontop` | Check whether an object is above another object |
| `PushPull` | Check an articulated joint interval |
| `Onfloor` | Detect an object falling below a reference height |

Preserve the task name, identifier, timestamps, exit code, step count, stage progress, and score for every episode.

## 7. Integrate a custom policy

Keep the repository's inference entry point and communication contract. A minimal package layout is:

```text
main/
├── model/
│   └── demo_infer.py
├── infer.py
├── genie_sim_ros.py
└── requirements.txt
```

The policy exchanges observations and commands through ROS 2:

| Topic | Direction | Type | Payload |
| --- | --- | --- | --- |
| `/joint_command` | policy → simulator | `JointState` | target joints |
| `/joint_states` | simulator → policy | `JointState` | current joints |
| `/sim/head_img` | simulator → policy | `CompressedImage` | head RGB image |
| `/sim/left_wrist_img` | simulator → policy | `CompressedImage` | left-wrist RGB image |
| `/sim/right_wrist_img` | simulator → policy | `CompressedImage` | right-wrist RGB image |
| `/sim/head_depth_img` | simulator → policy | `CompressedImage` | head depth image |
| `/sim/left_wrist_depth_img` | simulator → policy | `CompressedImage` | left-wrist depth image |
| `/sim/right_wrist_depth_img` | simulator → policy | `CompressedImage` | right-wrist depth image |

Validate joint order, camera order, image encoding, control rate, and gripper range before a formal run.

```bash
SIM_ASSETS="$SIM_ASSETS" ./scripts/start_gui.sh
./scripts/autorun.sh <TASK_NAME> infer
```

Place additional Python dependencies in a model-specific `requirements.txt` and verify imports in an isolated environment.

## 8. Teleoperation

Start the server in one container terminal:

```bash
omni_python server/source/genie.sim.lab/raise_standalone_sim.py
```

Start keyboard teleoperation in another terminal:

```bash
omni_python teleop/teleop.py \
  --task_name genie_task_home_microwave_food \
  --mode keyboard
```

The keyboard interface controls end-effector translation and rotation, base motion, torso, head, active arm, and gripper. Validate each direction before recording demonstrations.

For PICO, connect the headset and host to the same local network, configure the host address in AIDEA Vision App, and run:

```bash
omni_python teleop/teleop.py \
  --task_name genie_task_home_microwave_food \
  --mode pico \
  --host_ip <HOST_IP>
```

## 9. Record and replay trajectories

Record a teleoperated trajectory:

```bash
omni_python teleop/teleop.py \
  --task_name genie_task_home_pour_water \
  --mode keyboard \
  --record
```

The state trace is written to:

```text
output/recording_data/<TASK_NAME>/state.json
```

Start the server in replay-rendering mode:

```bash
omni_python server/source/genie.sim.lab/raise_standalone_sim.py \
  --disable_physics \
  --record_img \
  --record_video
```

Replay the state trace:

```bash
export TASK_NAME="genie_task_home_pour_water"

omni_python teleop/replay_state.py \
  --task_file "teleop/tasks/${TASK_NAME}.json" \
  --state_file "output/recording_data/${TASK_NAME}/state.json" \
  --record
```

Rendered images and videos are stored under `output/recording_data/<TASK_NAME>/<INDEX>/`. Separating interaction from rendering allows the same trajectory to be exported from multiple views.

## 10. Create a custom task

A task configuration contains six components:

1. a unique task name;
2. fixed, random, and non-interactive objects;
3. recording cameras;
4. robot configuration and initial base pose;
5. scene files and functional regions;
6. stage definitions with actions, active objects, and passive objects.

Use these locations:

```text
benchmark/bddl/eval_tasks/<TASK_NAME>.json
benchmark/bddl/task_to_preselected_scenes.json
benchmark/bddl/task_definitions/<TASK_NAME>/problem0.bddl
```

A custom policy implements initialization, episode reset, and action inference:

```python
class YourPolicy(BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        # Load configuration, normalization statistics, and weights.

    def reset(self) -> None:
        # Clear temporal state before each episode.
        pass

    def act(self, observations, **kwargs):
        # Return target joints in the Genie Sim interface order.
        raise NotImplementedError
```

Run it with:

```bash
python3 benchmark/task_benchmark.py \
  --task_name <TASK_NAME> \
  --policy_class <POLICY_NAME> \
  --env_class OmniEnv
```

## 11. Troubleshooting

### Server-client connection

Check the server and port 50051:

```bash
ss -ltnp | grep 50051
```

If a firewall is enabled, open only the required TCP port on a trusted network:

```bash
sudo ufw allow 50051/tcp
```

Identify the process holding the port before stopping it.

### Scene loading stalls

Check `SIM_ASSETS`, Git LFS completion, source-version compatibility, proxy variables, GPU memory, and disk activity. Relevant upstream discussions include [issue 11](https://github.com/AgibotTech/genie_sim/issues/11), [issue 29](https://github.com/AgibotTech/genie_sim/issues/29), and [issue 34](https://github.com/AgibotTech/genie_sim/issues/34).

### The policy moves but does not complete the task

Record observation keys, joint order, action range, control rate, evaluation-stage state, and termination reason. Compare them with the ADER rule for the task.

### Unstable collisions

Prefer simple convex hulls or convex decompositions, avoid dense triangle meshes near grasp regions, and verify mass, friction, and collision geometry.

## 12. Completion checklist

The completed workflow should produce:

- a container runtime that passes the device check;
- a complete GenieSimAssets checkout;
- one stable official task launch;
- one per-episode evaluation record;
- one custom-policy communication path;
- one replayable `state.json` trace;
- images or video rendered from that trace.
