"""Probe a kinematic footstep teacher against the physical V3 ladder.

This is a diagnostic, not a deployment controller.  It uses MuJoCo's site
Jacobian to turn a planned swing-foot arc into joint-position targets, then
checks whether the real contact-gated foothold state advances.  The result is
used to decide whether PPO needs a teacher/imitation term or only more data.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch


def _find_named_id(model: mujoco.MjModel, obj: mujoco.mjtObj, suffix: str) -> int:
    for index in range(int(model.nsite if obj == mujoco.mjtObj.mjOBJ_SITE else model.njnt)):
        name = mujoco.mj_id2name(model, obj, index)
        if name and (name == suffix or name.endswith("/" + suffix)):
            return index
    raise KeyError(f"cannot find {obj} named {suffix!r}")


def _site_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    site_id: int,
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    dofs = []
    for name in joint_names:
        joint_id = _find_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dofs.append(int(model.jnt_dofadr[joint_id]))
    return data.site_xpos[site_id].copy(), jacp[:, dofs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--bootstrap-state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-rung", type=int, default=1)
    parser.add_argument(
        "--swing-foot",
        choices=("left", "right"),
        default="left",
        help="foot whose task-space target is moved; the other foot is held",
    )
    parser.add_argument("--swing-time", type=float, default=1.35)
    parser.add_argument("--lift-height", type=float, default=0.045)
    parser.add_argument("--teacher-blend", type=float, default=0.25)
    parser.add_argument(
        "--body-force-threshold",
        type=float,
        default=None,
        help="diagnostic-only body contact threshold; omit for the strict task value",
    )
    parser.add_argument(
        "--teacher-ramp",
        type=float,
        default=0.0,
        help="seconds over which the teacher residual ramps from zero",
    )
    parser.add_argument(
        "--command-x",
        type=float,
        default=None,
        help="diagnostic-only forward command applied to the walking policy",
    )
    parser.add_argument(
        "--teacher-all",
        action="store_true",
        help=(
            "apply the kinematic teacher target to every actuated joint; "
            "use this to isolate foothold dynamics from residual-policy drift"
        ),
    )
    args = parser.parse_args()

    # The task config is registered during import.  Set bootstrap variables
    # before importing the task module so load_env_cfg includes the reset
    # event instead of silently starting from the floor.
    # Keep a caller-provided horizon for diagnostics that continue beyond the
    # first physical rung; the old unconditional assignment silently reduced
    # every run to the entry stage and made the success termination fire at
    # target plan index 2.
    os.environ.setdefault("MICRODUCK_LADDER_ACTIVE_RUNGS", "2")
    os.environ["MICRODUCK_LADDER_BOOTSTRAP_STATE"] = str(args.bootstrap_state)
    os.environ.setdefault("MICRODUCK_LADDER_BOOTSTRAP_JOINT_VELOCITY_SCALE", "0.5")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab_microduck.tasks import mdp as microduck_mdp
    from mjlab_microduck.tasks.microduck_video_ladder_footstep_env_cfg import (
        RUNG_COUNT,
        STAGE_SCHEDULE,
    )
    from rsl_rl.runners import OnPolicyRunner

    task_id = "Mjlab-Video-Ladder-Footstep-MicroDuck"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    if args.body_force_threshold is not None:
        env_cfg.terminations["ladder_body_collision"].params[
            "force_threshold"
        ] = float(args.body_force_threshold)
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.observations["critic"].enable_corruption = False
    agent_cfg = load_rl_cfg(task_id)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(args.checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    try:
        base_env.reset(seed=0)
        robot = base_env.scene["robot"]
        action_term = base_env.action_manager.get_term("joint_pos")
        model = base_env.sim.mj_model
        kin_data = mujoco.MjData(model)
        joint_names = tuple(robot.joint_names)
        left_names = tuple(name for name in joint_names if name.startswith("left_"))
        right_names = tuple(name for name in joint_names if name.startswith("right_"))
        joint_qpos_adr = []
        for name in joint_names:
            joint_id = _find_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            joint_qpos_adr.append(int(model.jnt_qposadr[joint_id]))
        left_site = _find_named_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        right_site = _find_named_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        site_ids, _ = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
        site_ids = list(site_ids)
        swing_site = left_site if args.swing_foot == "left" else right_site
        swing_site_index = 0 if args.swing_foot == "left" else 1
        swing_names = left_names if args.swing_foot == "left" else right_names
        swing_indices = (
            [joint_names.index(name) for name in left_names]
            if args.swing_foot == "left"
            else [joint_names.index(name) for name in right_names]
        )
        start = robot.data.site_pos_w[0, site_ids[swing_site_index]].detach().cpu().numpy().copy()
        # The public CLI names a physical rung.  The environment planner has
        # two lower-timber placements before rung 0, so convert once here and
        # keep the rest of the diagnostic in plan-index space.
        target_index = torch.tensor(
            [microduck_mdp.LADDER_ENTRY_FOOTSTEPS + args.target_rung],
            device=base_env.device,
        )
        goal = microduck_mdp._ladder_target_world(
            base_env,
            target_index,
            RUNG_COUNT,
            0.18,
            0.035,
            0.72,
            0.67,
            0.065,
            0.012,
        )[0].detach().cpu().numpy()
        qpos = base_env.sim.data.qpos[0].detach().cpu().numpy().copy()
        teacher_indices = list(range(len(joint_names))) if args.teacher_all else swing_indices
        ik_qpos = qpos.copy()
        hold_target = robot.data.joint_pos[0].detach().cpu().numpy().copy()
        _, initial_jac = _site_jacobian(model, kin_data, ik_qpos, swing_site, swing_names)
        print("INITIAL_LEFT_JAC", initial_jac.round(5).tolist())
        kin_data.qpos[:] = ik_qpos
        mujoco.mj_forward(model, kin_data)
        print("KIN_LEFT", kin_data.site_xpos[left_site].round(6).tolist())
        print("JOINT_NAMES", joint_names)
        print("SWING_FOOT", args.swing_foot)
        print("SWING_JOINTS", swing_names)
        print("RIGHT_JOINTS", right_names)
        print("QPOS_JOINTS", qpos[np.asarray(joint_qpos_adr)].tolist())
        print("ROBOT_JOINTS", robot.data.joint_pos[0].detach().cpu().numpy().tolist())
        print("START_LEFT", start.tolist())
        print("TARGET_LEFT", goal.tolist())

        obs = env.get_observations()
        for step in range(args.steps):
            if args.command_x is not None:
                command = base_env.command_manager.get_command("twist")
                command[:, 0] = float(args.command_x)
                command[:, 1:] = 0.0
            with torch.inference_mode():
                policy_raw = policy(obs)
            raw = policy_raw.detach().clone()
            qpos = base_env.sim.data.qpos[0].detach().cpu().numpy().copy()
            # Re-linearize around the *measured* actuator state on every
            # control cycle.  Keeping old joint qpos here makes the Jacobian
            # operate on a fictitious configuration; the residual then
            # accumulates at the hip while the real foot barely moves.
            ik_qpos[:] = qpos
            current = robot.data.site_pos_w[0, site_ids[swing_site_index]].detach().cpu().numpy()
            alpha = min((step + 1) * base_env.step_dt / max(args.swing_time, 1e-4), 1.0)
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            desired = start + smooth * (goal - start)
            desired[2] += args.lift_height * np.sin(np.pi * alpha)
            _, jac = _site_jacobian(model, kin_data, ik_qpos, swing_site, swing_names)
            # The vectorized mjlab scene and the standalone MuJoCo kinematic
            # view can differ by a per-environment world origin.  Use the
            # live simulator's foot position for the absolute error; keep the
            # standalone model only for the local Jacobian.
            error = desired - current
            damping = 0.0025
            delta = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), error)
            delta = np.clip(delta, -0.10, 0.10)
            swing_qpos_adr = np.asarray([joint_qpos_adr[i] for i in swing_indices])
            ik_qpos[swing_qpos_adr] += 0.90 * delta
            target = hold_target.copy()
            target[swing_indices] = ik_qpos[swing_qpos_adr]
            # Hold the support leg and head at their measured positions.  The
            # action is relative to the configured default-pose offset.
            leg_raw = torch.as_tensor(target, device=base_env.device, dtype=torch.float32)[None, :]
            offset = torch.as_tensor(
                action_term._offset, device=base_env.device, dtype=leg_raw.dtype
            )
            if offset.ndim == 2:
                offset = offset[0]
            scale = torch.as_tensor(
                action_term._scale, device=base_env.device, dtype=leg_raw.dtype
            )
            if scale.ndim == 2:
                scale = scale[0]
            if scale.ndim == 0:
                scale = scale.expand_as(offset)
            leg_raw = (leg_raw - offset[None, :]) / scale.clamp_min(1e-6)[None, :]
            if step == 0:
                print("FIRST_TARGET", target.tolist())
                print("FIRST_RAW", leg_raw[0].detach().cpu().tolist())
            blend = float(np.clip(args.teacher_blend, 0.0, 1.0))
            if args.teacher_ramp > 0.0:
                blend *= float(
                    np.clip((step + 1) * base_env.step_dt / args.teacher_ramp, 0.0, 1.0)
                )
            raw[0, teacher_indices] = (
                (1.0 - blend) * raw[0, teacher_indices]
                + blend * leg_raw[0, teacher_indices]
            )
            obs, _, dones, extras = env.step(raw)
            if step < 10 or (step + 1) % 25 == 0:
                feet = robot.data.site_pos_w[0, site_ids].detach().cpu().numpy()
                actual_joints = robot.data.joint_pos[0].detach().cpu().numpy()
                kin_data.qpos[:] = ik_qpos
                mujoco.mj_forward(model, kin_data)
                target_state = int(getattr(base_env, "_ladder_target_rung", torch.zeros(1))[0].item())
                print(
                    "STEP",
                    step + 1,
                    "target_state",
                    target_state,
                    "left",
                    feet[0].round(4).tolist(),
                    "right",
                    feet[1].round(4).tolist(),
                    "swing_q",
                    actual_joints[swing_indices].round(3).tolist(),
                    "err",
                    float(np.linalg.norm(error)),
                    "desired",
                    desired.round(4).tolist(),
                    "ik_site",
                    kin_data.site_xpos[left_site].round(4).tolist(),
                    "done",
                    bool(dones[0].item()),
                )
            if bool(dones[0].item()):
                print("TERMINATED", step + 1, extras)
                break

        feet = robot.data.site_pos_w[0, site_ids].detach().cpu().numpy()
        found = base_env.scene.sensors["ladder_foot_contact"].data.found[0].detach().cpu().tolist()
        target_state = int(getattr(base_env, "_ladder_target_rung", torch.zeros(1))[0].item())
        print("FINAL_TARGET_STATE", target_state)
        print("FINAL_FEET", feet.tolist())
        print("FINAL_CONTACT", found)
    finally:
        env.close()


if __name__ == "__main__":
    main()
