# ATEC2026 L0 Task E: Rules, Physical Configuration, and Current Grasping Issues Explanation

Update time: 2026-06-04
Working directory: `$PROJECT_ROOT`

## 1. Conclusion first

1. Currently, we are debugging the `ATEC-TaskE-Piper` desktop grasping task within the ATEC2026 official simulation package, not a L1 foot/leg task, nor a custom robotic arm task.
2. The local physical configuration primarily comes from the official repository: Isaac Lab v2.3.2, AgileX Piper, three YCB-style objects, a desktop, a basket, fixed RGB-D observation, and end-effector RGB-D observation. The friction between Piper/objects/ground and solver parameters can be found in the source code.
3. “Grabbing and sliding off” is not a simple bug that can be simply handed over to Isaac Sim. More precisely: the official physical configuration is sensitive to the friction grip of two-finger grippers, and our current GraspGen-style/PCA submit-side primitive still relies on friction for non-convex curved objects like bananas. If there are deviations in finger-center, closing height, gripper opening, transport height, or speed, a brief lifting followed by sliding occurs.
4. It is not possible to submit by modifying the physical parameters of the competition evaluation environment. Increasing solver iteration, friction, and contact parameters can serve as local diagnostics, but the final submission must succeed under the official judge environment using `demo/solution.py` observations and action outputs.
5. The current strongest deployable baseline is still the ACT/XSA scheme in `demo/solution.py`; `demo/solution_pca.py` is an experimental branch and should not be used for deployment, unless the multi-object multi-seed continuous evaluation clearly exceeds ACT/XSA.

## 2. Official materials downloaded/organized

The official GitHub README snapshot has been downloaded this time:

- Local snapshot: `docs/sources/ATEC2026_Simulation_Challenge_official_readme_20260604.md`
- Official source link: <https://github.com/atecup/ATEC2026_Simulation_Challenge>
- Official raw README: <https://raw.githubusercontent.com/atecup/ATEC2026_Simulation_Challenge/main/readme.md>

2026-06-04 Further official fresh clone control group, temporary directory:

- `/tmp/ATEC2026_Simulation_Challenge_official_check`
- Official HEAD: `dbe7c251f680b02f357a6db67430b18d3ba45ea1`
- Latest commit: `dbe7c25 2026-06-04 10:10:16 +0800 Merge pull request #8 from atecup/fix/tron2a_joint_configuration`

Comparison conclusion:

- Files closely related to the physics of Task E’s grasping, with identical local and official latest version SHA:
  - `source/atec_rl_lab/atec_rl_lab/assets/robots/piper.py`
  - `source/atec_rl_lab/atec_rl_lab/assets/objects/task_b/object.py`
  - `source/atec_rl_lab/atec_rl_lab/tasks/task_e/terrain.py`
  - `source/atec_rl_lab/atec_rl_lab/tasks/task_e/mdp/rewards.py`
  - `source/atec_rl_lab/atec_rl_lab/tasks/task_e/mdp/terminations.py`
- `source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py` has only a difference in the action group field name: the official latest version uses `joint_leg` / `joint_wheel` / `joint_arm`, while the local version uses `joint_pos_leg` / `joint_vel_wheel` / `joint_pos_arm`; the objects, baskets, cameras, random Y-band, reward/termination, and physics material logic in Task E are consistent.
- `source/atec_rl_lab/atec_rl_lab/tasks/task_base/envs_base_cfg.py` has two types of differences: a difference in the action group field name, and the base reset joint randomization has changed from `(1.0, 1.0)` in the official version to `(0.5, 1.5)` in the local version; in `TaskEEnvPiperCfg`, Task E Piper has disabled the related reset events, so this is not the cause of the banana slipping.
- `demo/server.py` introduces the `/get_action_spec` interface in the official latest version; the local server does not have this interface yet. This only affects whether players can customize action mode/scale/clip in the submission service, without changing the official simulation physics.
- `demo/Dockerfile` has been changed to the ACT/XSA submission template in the local version, while the official default remains the `solution_zero.py` example. This is our own submission image selection, not a difference in the physical environment.

On the same day, the official updates and issues/PRs were checked again:

- After `git fetch origin`, `origin/main` remains `dbe7c251f680b02f357a6db67430b18d3ba45ea1`, and there is no commit updated after 2026-06-04 10:10:16.
- No public issues on GitHub issues/PR pages mention the Task E / Piper / grasping / physics / friction / banana slipping issue.
- `Piper` targeted is PR #2: the eye-in-hand camera configuration for B2Piper/B2WPiper/G1, not the fixed Piper of Task E, nor the gripper slipping.
- PR #5 is a participant action spec; PR #8 is a Tron2A joint configuration reset bug, which has nothing to do with the grasping physics of Task E.
- A unique low-risk compatible fix has been synchronized: add the official `/get_action_spec` endpoint to the local `demo/server.py`. If the solution does not implement this method, it returns `{}`, i.e., using the default action configuration.

The local `readme.md` has minor differences from the official raw README: The new official README adds an optional interface description for `AlgSolution.get_action_spec()`; the local file can still be used to run the current code, but you should check whether the official version enables this new interface before submitting it.

Additionally, the event description provided by the user previously includes:

- L0 Task E: Three different types of objects are randomly placed on the desktop. A desktop robotic arm is used to identify, grasping, and placing them in the designated area.
- Core process: identification, planning, grasping, placement.
- Code review: The highest-scoring image corresponding code must be reproducible and compliant; technical reports need to describe the implementation approach, code structure, and execution process. When using open-source models, a download link must be provided; for additional training models, training code and data sources should be presented.
- During the prediction phase, only the code and model from the uploaded image can be used; utilizing test sets or external information is prohibited.

## 3. Key requirements of the official repository for Task E

The official README specifies that this repository provides simulation assets, task definitions, and submission scripts, and the environment is based on Isaac Lab. Task E corresponds to `ATEC-TaskE-Piper` in the environment matrix. Refer to:

- `readme.md` / Official README: Task E uses `ATEC-TaskE-Piper`.
- `readme.md`: The repository development/test version is Isaac Lab v2.3.2.
- `readme.md`: Participants must implement `demo/solution.py`, with the class name `AlgSolution`, and the core function being `predicts(obs, current_score)`, returning `{"action": action, "giveup": False}`.

Current commit layout of the local Dockerfile:

- `demo/Dockerfile` will copy `solution_act.py` into `solution.py` within the mirror, along with `policy_act.pt`.
- This indicates that the current default commit mirror still uses the ACT/XSA route, not the PCA/AnyGrasp route.
- If you want to switch to GraspGen-style/PCA in the future, you must first change the Dockerfile copy target and complete multi-seed stability evaluation.

## 4. Task E Local Scoring Logic

Source code locations: `source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py` and `source/atec_rl_lab/atec_rl_lab/tasks/task_e/mdp/rewards.py`.

There are two types of rewards for the local Task E:

1. `grasped_objects_once`: The object is approached by the end effector and lifted to a certain height above the table surface. A single object earns approximately 3 points at once.
2. `objects_in_basket`: The object enters the successful area of the basket. A single object earns approximately 3 points at once.

Therefore, a full score for the three-object theory usually corresponds to:

- 3 objects are "picked up/raised" once each: approximately 9 points.
- 3 objects are finally scored into the basket: approximately 9 points.
- Total of approximately 18 points.

The basket success area is defined in the source code as:

- Success Center: `BASKET_SUCCESS_CENTER = (1.08, -0.30, TABLE_TOP_Z + 0.15)`
- XY half-width: `half_x = 0.20`, `half_y = 0.11`
- Z range: `TABLE_TOP_Z <= object_z <= TABLE_TOP_Z + 0.15`

This explains an important phenomenon: what seems to have been grasped in the video only yields grasping/lifting points; to truly make it onto the list, all three objects must land steadily within the basket’s successful area.

## 5. Official/Local Physical Configuration Evidence

### 5.1 Piper robotic arm configuration

Source code: `source/atec_rl_lab/atec_rl_lab/assets/robots/piper.py`

Key parameters:

- Robot gripper: `robot/piper/piper.usd`
- Joints: `joint1` to `joint8`, where `joint7/joint8` is a two-finger gripper.
- Contact sensor: `activate_contact_sensors=True`
- Implicit actuators for joints: `effort_limit=100.0`, `velocity_limit=100.0`, `stiffness=800.0`, `damping=80.0`
- Articulation solver: `solver_position_iteration_count=4`, `solver_velocity_iteration_count=0`

Source code: `source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py`

Initial state of Task E Piper:

- robotic arm fixed on the right side of the table: `pos=(TABLE_CENTER_X + TABLE_HALF_X, TABLE_CENTER_Y, TABLE_TOP_Z)`
- `piper_cfg.spawn.rigid_props.disable_gravity = True` in Task E
- Initial opening of gripper: `joint7=0.035`, `joint8=-0.035`

The solver iteration here is low, especially for `velocity_iteration_count=0`, which is not very friendly to contact stability. It was not modified from our grasping code, but rather the default configuration of Piper in the repository.

### 5.2 Three Object Configuration

Source code: `source/atec_rl_lab/atec_rl_lab/assets/objects/task_b/object.py`

Three objects:

- `object_1`：`004_sugar_box.usd`
- `object_2`：`006_mustard_bottle.usd`
- `object_3`：`011_banana.usd`

Common physical parameters:

- `rigid_body_enabled=True`
- `kinematic_enabled=False`
- `linear_damping=2.0`
- `angular_damping=4.0`
- `max_depenetration_velocity=0.5`
- `mass=0.5`
- `contact_offset=0.01`
- `rest_offset=0.0`

Quality consistency: 0.5kg is too heavy for small bananas/small boxes, which will increase the difficulty of the two-finger gripper clamping through friction.

### 5.3 Physical Materials for Ground/Desktop

Source code: `source/atec_rl_lab/atec_rl_lab/tasks/task_e/terrain.py`

Task E terrain physical materials:

- `friction_combine_mode="multiply"`
- `restitution_combine_mode="multiply"`
- `static_friction=1.0`
- `dynamic_friction=1.0`
- `restitution=0.0`

Source code: `source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py`

Task E is executed in `__post_init__()`:

- `self.scene.terrain = TASK_E_TERRAIN_CFG`
- `self.sim.physics_material = self.scene.terrain.physics_material`
- `self.events.physics_material = None`

This indicates that Task E has disabled the default physics material randomization and uses a fixed physical material. For submission, this is beneficial: the random friction is not applied every time; however, for the contact geometry of a banana, a fixed friction does not guarantee stable grip.

### 5.4 Random Placement of Objects

Source code: `source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py`

Three objects are randomly distributed across different Y-bands on the desktop:

- `object_1`：`Y in [0.25, 0.29]`
- `object_2`：`Y in [0.14, 0.20]`
- `object_3`：`Y in [0.03, 0.09]`
- `X in [0.90, 1.10]`

This is consistent with our subsequent work on RGB-D band mask and object region segmentation. However, the final submission cannot read the object root; it can only estimate the position based on observations.

## 6. Why does a banana "get stuck and then slide off"?

The current most reasonable judgment is not that “there is no grasping model at all,” nor that “Isaac Sim is completely broken,” but rather the combination of the following factors:

1. Banana is a curved, non-convex object with a narrow local contact area. If the Piper two-finger gripper only contacts the local curve, rolling/sliding contact can easily occur.
2. The current object mass is uniformly 0.5 kg, and the banana is set to the same weight as the box, which is very unfavorable for friction-only pinch.
3. The Piper solver parameter `position=4, velocity=0` is not strong enough for stable contact; discussions about “the two-finger gripper drops the object, requiring adjustment of contact/solver/friction or control method” are also common in NVIDIA/Isaac Sim documentation and forums.
4. Our submit-style PCA controller can only use RGB-D and proprio data, without reading the true object root; slight deviations in the grasp center/yaw/closing height estimated from a single perspective point cloud may lead to the situation where “it appears that the grip is deep, but in reality the finger contact normal is insufficient.”
5. The old runner/collector sometimes works because it more closely matches the official scripted primitive through accurate object state, dynamic servoing, and true finger-center tracking. That capability is lost when moving to the observation-only control in `demo/solution_pca.py`.

In a nutshell: It’s not just a matter of “insufficient gripper force,” but a combination of contact geometry, closing height, gripper opening, trajectory during transportation, and official PhysX contact parameters.

## 7. Can the physical configuration be changed?

### What can be done

Local diagnosis can be performed:

- Temporarily increase the Piper articulation solver iteration, such as raising the position from 4 to 16 and the velocity from 0 to 4.
- Temporarily increase the friction of the object/gripper and check the contact offset/rest offset.
- Use these experiments to determine whether "the sliding issue is mainly due to the sensitivity of PhysX contact solving".

### Not recommended as a submission dependency

The final submission should not rely on modifying the official evaluation physics:

- The code review requires that the highest-scoring reproduction is possible on the environment set by the organizing committee.
- If we rely on local modifications to the physics to ensure stability, online judges may not load these changes.
- The compliant approach should be: under the official physical configuration, completing tasks through more stable motion primitives, low-position handling, under-scoop, slow stabilization, and releasing above the basket center.

## 8. Current Solution Status

### 8.1 ACT/XSA

The current default deployable baseline is still ACT/XSA:

- The Dockerfile currently copies `solution_act.py` to submit `solution.py`.
- In historical evaluations, XSA final has achieved high scores on seeds 11/12/13, but there have also been non-deterministic fluctuations.
- This is the most mature image route currently, but it cannot be said to be completely stable and scoring full points.

### 8.2 pi0.5

pi0.5 has tried the 20-demo and 100-demo branches, but the current performance is poor:

- The seeds of the 20-demo fast branch on 11/12/13 are all 0.
- The seeds of the 100-demo branch on 11/12/13 are approximately `0/0/6`, with a mean value of about 2.
- The main issue is more related to the adaptation layer/data representation being incorrect, rather than the weakness of the pi0.5 model itself.

### 8.3 AnyGrasp

AnyGrasp SDK has been integrated and supports inference, but its license and distribution restrictions still apply:

- It is very valuable as a local candidate generator.
- However, if the final image submitted contains non-distributable SDKs or weights, the code review risk is high.
- We can draw inspiration from its grasp candidate approach, but it is best to develop a reproductionable and distributable geometric/PCA/GraspNet-style method, or provide a full explanation of the legal source and license in the technical report.

### 8.4 GraspGen-style PCA/AABB

This is the actual grasping route for the user's priority sprint plan:

- `object_1` has evidence of a single object's success with seed11/12/13 in `demo/solution_pca.py`.
- `object_2` has evidence of a single object's success in runner, but the submit-style version has not yet achieved stable migration.
- `object_3` can currently perform a short strong lift in some scenarios, but it will slide during the mid/transport/release stages, and stability for submission is still not achieved.

It cannot be claimed that PCA/AABB has achieved a full score. It is one of the correct directions, but the issue of banana transportation and the alignment with the object_2 submit-side still need to be completely resolved.

## 9. Recommended subsequent technical approach

1. Do not revert to the "only training ACT/pi0.5" black-box approach; continue using ACT/XSA as the submission baseline, while advancing GraspGen-style/PCA as the ranking strategy.
2. Create two branches for object_3:
   - Local physical enhancement diagnosis: Increase solver iterations to see if there is immediate stability and confirm the sensitivity of contact solving.
   - Official physical submission fixes: Low-position transfer, under-scoop, shortened pure vertical lift at high altitude, slow contact movement along the back rim direction, and stable operation after the basket center is above the edges before opening the claws.
3. Migrate the runner success parameters for object_2 separately; do not use the compensation from object_1/object_3.
4. All evidence of success must be recorded across multiple seeds: at least seeds 11/12/13, and preferably 21/22/23 as well.
5. Before switching to the Dockerfile in the final stage, the following conditions must be met:
   - The score of the three objects in the same episode is stable and close to or equal to 18.
   - There are video evidence.
   - No reading of env root/object states.
   - Not relying on AnyGrasp SDK for non-distributable weights, or having completed compliance documentation.
   - The technical report should explain the model/algorithm/open-source links/training or no-training process.

## 10. Reference Links

- ATEC2026 Simulation Challenge official repository: <https://github.com/atecup/ATEC2026_Simulation_Challenge>
- Official README raw snapshot source: <https://raw.githubusercontent.com/atecup/ATEC2026_Simulation_Challenge/main/readme.md>
- Isaac Lab pip installation documentation: <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>
- Isaac Sim Physics Simulation Fundamentals: <https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html>
- NVIDIA forum, Object Gripping and picking: <https://forums.developer.nvidia.com/t/object-gripping-and-picking/291963>
- NVIDIA forum, Stable grasping in Isaac Sim: <https://forums.developer.nvidia.com/t/stable-grasping-in-isaac-sim/354866>

## 11. Handover Reminder for the Team

1. Do not treat `demo/solution_pca.py` as the final submission; it is just an experimental controller.
2. Do not directly integrate the AnyGrasp SDK into the final image unless the license, technical report, and reproduction check confirm that there are no risks.
3. Do not claim that online reproduction is possible after modifying official physical parameters; physical modifications are only used to identify issues.
4. To truly make it on the rankings, the sequence “Observation Estimation Center -> finger-center control -> closure -> low-position transport -> stable release above the basket center” must be arranged into a single stable trajectory.
5. The current priority is to fix the sliding issue during the banana transport phase of object_3, rather than replacing the large model.
