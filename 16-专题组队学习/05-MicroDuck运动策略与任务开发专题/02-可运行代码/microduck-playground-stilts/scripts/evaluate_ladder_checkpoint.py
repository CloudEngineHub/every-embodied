"""Deterministic physical audit for a MicroDuck ladder checkpoint.

This is intentionally separate from training metrics.  A vectorized PPO
``ladder_footstep_success`` count can include many short stochastic episodes;
this script checks one reset with the actor mean and reports whether the
foothold state machine actually advances on real rung contact.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.tasks import mdp as ladder_mdp
from rsl_rl.runners import OnPolicyRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--active-rungs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug-every", type=int, default=0)
    parser.add_argument(
        "--save-transition",
        type=Path,
        default=None,
        help="Save the first real target transition state for a bridge curriculum.",
    )
    parser.add_argument(
        "--save-transition-delay",
        type=int,
        default=0,
        help="Wait this many control steps after target 1 is reached before saving.",
    )
    parser.add_argument(
        "--hold-bootstrap-action",
        action="store_true",
        help="Replay the bootstrap position action instead of the learned policy.",
    )
    args = parser.parse_args()

    os.environ["MICRODUCK_LADDER_ACTIVE_RUNGS"] = str(args.active_rungs)
    double_support = os.environ.get("MICRODUCK_LADDER_DOUBLE_SUPPORT", "1") != "0"
    entry_target_count = ladder_mdp.LADDER_ENTRY_FOOTSTEPS
    rung_target_stride = 2 if double_support else 1
    task_id = "Mjlab-Video-Ladder-Footstep-MicroDuck"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.observations["critic"].enable_corruption = False
    agent_cfg = load_rl_cfg(task_id)

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(args.checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    base_env.reset(seed=0)
    obs = env.get_observations()
    robot = base_env.scene["robot"]
    site_ids, _ = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
    sensor = base_env.scene.sensors["ladder_foot_contact"]
    max_root_x = float("-inf")
    max_foot_z = torch.full((2,), float("-inf"), device=base_env.device)
    max_target = 0
    target_steps: list[int] = []
    done_count = 0
    success_count = 0
    success_steps: list[int] = []
    transition_step: int | None = None
    target_history: list[tuple[int, int]] = []

    for step in range(args.steps):
        with torch.inference_mode():
            if args.hold_bootstrap_action and base_env.action_manager.action is not None:
                actions = base_env.action_manager.action.clone()
            else:
                actions = policy(obs)
        target_before = int(
            getattr(base_env, "_ladder_target_rung", torch.zeros(1))[0].item()
        )
        obs, _, dones, extras = env.step(actions)
        root_x = float(robot.data.root_link_pos_w[0, 0].item())
        feet_z = robot.data.site_pos_w[0, site_ids, 2]
        max_root_x = max(max_root_x, root_x)
        max_foot_z = torch.maximum(max_foot_z, feet_z)
        target = int(getattr(base_env, "_ladder_target_rung", torch.zeros(1))[0].item())
        if not target_history or target != target_history[-1][1]:
            target_history.append((step + 1, target))
        if target_before == 0 and target >= 1:
            transition_step = step + 1
        if (
            args.save_transition is not None
            and target >= 1
            and transition_step is not None
            and step + 1 - transition_step >= max(args.save_transition_delay, 0)
            and not args.save_transition.exists()
        ):
            args.save_transition.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "qpos": base_env.sim.data.qpos[0].detach().cpu(),
                    "qvel": base_env.sim.data.qvel[0].detach().cpu(),
                    # The actor observes the previous action.  A bridge state
                    # without this value is not Markov-equivalent to the
                    # rollout state and can create a discontinuous first
                    # command immediately after reset.
                    "action": base_env.action_manager.action[0].detach().cpu()
                    if base_env.action_manager.action is not None
                    else None,
                    "target_rung": target,
                    "common_step": int(getattr(base_env, "common_step_counter", 0)),
                },
                args.save_transition,
            )
            print(f"SAVED_TRANSITION {args.save_transition}")
        if target_before > max_target:
            max_target = target_before
            target_steps.append(step + 1)
        if target > max_target:
            max_target = target
            target_steps.append(step + 1)
        log = extras.get("log", {}) if isinstance(extras, dict) else {}
        if bool(dones[0].item()) and (step < 3 or args.debug_every):
            termination_log = {
                key: value
                for key, value in log.items()
                if str(key).startswith("Episode_Termination/")
            }
            print("DONE_LOG", step + 1, termination_log)
        success_events = int(log.get("Episode_Termination/ladder_footstep_success", 0))
        if bool(dones[0].item()) and success_events:
            success_count += success_events
            success_steps.extend([step + 1] * success_events)
            max_target = max(max_target, args.active_rungs)
            target_steps.append(step + 1)
        if args.debug_every and (step + 1) % args.debug_every == 0:
            found_now = sensor.data.found[0].detach().cpu().tolist()
            time_now = sensor.data.current_contact_time
            if time_now is not None:
                time_now = time_now[0].detach().cpu().tolist()
            feet_now = robot.data.site_pos_w[0, site_ids].detach().cpu().tolist()
            active_now = int(
                getattr(base_env, "_ladder_target_rung", torch.zeros(1))[0].item()
            )
            origin_now = base_env.scene.terrain.env_origins[0].detach().cpu().tolist()
            goal_now = ladder_mdp._ladder_target_world(
                base_env,
                torch.tensor([target], device=base_env.device),
                14,
                0.18,
                0.035,
                0.72,
                0.67,
                0.065,
                ladder_mdp._LADDER_FOOT_SITE_Z_OFFSET,
            )[0].detach().cpu().tolist()
            selected_now = feet_now[1]
            along_now = ((selected_now[0] - goal_now[0]) ** 2 + (selected_now[2] - goal_now[2]) ** 2) ** 0.5
            lateral_now = abs(selected_now[1] - origin_now[1])
            print(
                "DEBUG",
                step + 1,
                "target",
                target,
                "active",
                int(getattr(base_env, "_ladder_debug_active", torch.zeros(1))[0].item()),
                "selected_found",
                bool(getattr(base_env, "_ladder_debug_selected_found", torch.zeros(1, dtype=torch.bool))[0].item()),
                "selected_time",
                round(float(getattr(base_env, "_ladder_debug_selected_time", torch.zeros(1))[0].item()), 4),
                "gate_err",
                round(float(getattr(base_env, "_ladder_debug_along_height", torch.zeros(1))[0].item()), 4),
                round(float(getattr(base_env, "_ladder_debug_lateral", torch.zeros(1))[0].item()), 4),
                "state_selected",
                getattr(base_env, "_ladder_debug_selected_pos", torch.zeros((1, 3)))[0].detach().cpu().tolist(),
                "state_goal",
                getattr(base_env, "_ladder_debug_target_pos", torch.zeros((1, 3)))[0].detach().cpu().tolist(),
                "root",
                getattr(base_env, "_ladder_debug_root_pos", torch.zeros((1, 3)))[0].detach().cpu().tolist(),
                "root_gap",
                round(float(getattr(base_env, "_ladder_debug_root_support_gap", torch.zeros(1))[0].item()), 4),
                "root_upright",
                round(float(getattr(base_env, "_ladder_debug_root_cos_tilt", torch.zeros(1))[0].item()), 4),
                "root_speed",
                round(float(getattr(base_env, "_ladder_debug_root_speed", torch.zeros(1))[0].item()), 4),
                "episode_len",
                int(base_env.episode_length_buf[0]),
                "common_step",
                int(getattr(base_env, "common_step_counter", 0)),
                "state_step",
                int(getattr(base_env, "_ladder_state_step", -1)),
                "advanced",
                bool(getattr(base_env, "_ladder_advanced_this_step", torch.zeros(1, dtype=torch.bool))[0].item()),
                "found",
                found_now,
                "contact_time",
                time_now,
                "feet",
                feet_now,
                "goal0",
                goal_now,
                "err",
                round(along_now, 4),
                round(lateral_now, 4),
            )
        done_count += int(dones[0].item())
        if bool(dones[0].item()):
            transition_step = None

    found = sensor.data.found[0].detach().cpu().tolist()
    contact_time = sensor.data.current_contact_time
    if contact_time is not None:
        contact_time = contact_time[0].detach().cpu().tolist()
    final_feet = robot.data.site_pos_w[0, site_ids].detach().cpu().tolist()
    print(f"MAX_TARGET {max_target}")
    print(f"TARGET_STEPS {target_steps}")
    print(f"SUCCESS_COUNT {success_count}")
    print(f"SUCCESS_STEPS {success_steps}")
    print(f"DONE_COUNT {done_count}")
    print(f"MAX_ROOT_X {max_root_x:.4f}")
    print("MAX_FOOT_Z " + " ".join(f"{float(value):.4f}" for value in max_foot_z.cpu()))
    print("FINAL_FEET " + " ".join(
        "(" + ",".join(f"{float(value):.4f}" for value in foot) + ")"
        for foot in final_feet
    ))
    print("FINAL_RUNG_CONTACT " + " ".join(str(float(value)) for value in found))
    if contact_time is not None:
        print("FINAL_CONTACT_TIME " + " ".join(f"{float(value):.4f}" for value in contact_time))
    raw_rung_flags = ladder_mdp._ladder_raw_rung_contact_flags(base_env, 14)
    if raw_rung_flags is not None:
        print(
            "FINAL_RAW_RUNG_CONTACT "
            + " ".join(str(int(value)) for value in raw_rung_flags[0].flatten().cpu())
        )
    max_entry_target = min(max_target, entry_target_count)
    completed_rungs = max(0, max_target - entry_target_count) // rung_target_stride
    print(f"MAX_ENTRY_TARGET {max_entry_target}/{entry_target_count}")
    print(f"MAX_PHYSICAL_RUNG_TARGET {completed_rungs}")
    print("TARGET_HISTORY " + " ".join(f"{step}:{target}" for step, target in target_history))
    env.close()


if __name__ == "__main__":
    main()
