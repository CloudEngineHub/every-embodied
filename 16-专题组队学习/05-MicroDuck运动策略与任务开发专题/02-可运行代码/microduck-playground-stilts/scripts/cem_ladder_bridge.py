"""Search a physical second-rung bridge residual around a walking policy.

The search runs on vectorized MuJoCo environments. The walking checkpoint
provides the nominal gait; CEM searches a low-dimensional, piecewise-constant
residual over both leg chains so it can first transfer the COM and then swing
the planned foot. A candidate is scored from live foot pose, support contact,
and termination state. No root pose or foothold state is edited during the
rollout.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_video_ladder_footstep_env_cfg import RUNG_COUNT
from rsl_rl.runners import OnPolicyRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--segments", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--elite-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--bootstrap-state",
        type=Path,
        default=None,
        help="optional real target-0->1 state used to search the next transfer",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.10,
        help="maximum normalized residual added to the position action",
    )
    parser.add_argument(
        "--initial-std",
        type=float,
        default=0.45,
        help="initial CEM standard deviation in normalized residual units",
    )
    parser.add_argument(
        "--target-rung",
        type=int,
        default=2,
        help="uncompleted rung whose swing-foot transfer should be searched",
    )
    parser.add_argument(
        "--control-scope",
        choices=("legs", "all"),
        default="legs",
        help="search leg residuals only, or all 14 joints including head balance",
    )
    parser.add_argument(
        "--base-control",
        choices=("policy", "hold"),
        default="policy",
        help="use the PPO policy or the captured bootstrap action as the nominal control",
    )
    args = parser.parse_args()
    if args.residual_scale <= 0.0 or args.initial_std <= 0.0:
        raise ValueError("--residual-scale and --initial-std must be positive")
    if args.bootstrap_state is not None:
        # The task reads this before constructing the reset event.  The
        # bootstrap remains a real qpos/qvel/action state; only the start
        # distribution changes for this bridge search.
        os.environ["MICRODUCK_LADDER_BOOTSTRAP_STATE"] = str(args.bootstrap_state)
        # A captured bridge state is dynamic.  Zeroing joint velocity turns it
        # into a different, usually unstable static pose before CEM sees the
        # first action.  Preserve the measured joint momentum by default.
        os.environ.setdefault("MICRODUCK_LADDER_BOOTSTRAP_JOINT_VELOCITY_SCALE", "1.0")

    task_id = "Mjlab-Video-Ladder-Footstep-MicroDuck"
    env_cfg = load_env_cfg(task_id, play=True)
    # ``mjlab_microduck.tasks`` is imported before ``main`` runs, so setting
    # MICRODUCK_LADDER_BOOTSTRAP_STATE above is too late for the registry's
    # config factory.  Install the reset event explicitly to guarantee that
    # every CEM candidate starts from the captured dynamic support state.
    if args.bootstrap_state is not None:
        env_cfg.events["ladder_bootstrap_reset"] = EventTermCfg(
            func=microduck_mdp.ladder_bootstrap_reset,
            mode="reset",
            params={
                "state_path": str(args.bootstrap_state),
                "target_rung": int(
                    os.environ.get(
                        "MICRODUCK_LADDER_BOOTSTRAP_TARGET_RUNG",
                        str(args.target_rung),
                    )
                ),
                "velocity_scale": 0.0,
                "joint_velocity_scale": float(
                    os.environ.get(
                        "MICRODUCK_LADDER_BOOTSTRAP_JOINT_VELOCITY_SCALE",
                        "1.0",
                    )
                ),
            },
        )
    env_cfg.scene.num_envs = args.envs
    # CEM is a teacher-search diagnostic.  It may pass through a low-force
    # rail/rung contact while discovering a foot arc; the final PPO task still
    # uses its strict collision threshold.  Keep this override explicit.
    body_force_threshold = os.environ.get("MICRODUCK_LADDER_CEM_BODY_FORCE_THRESHOLD")
    if body_force_threshold is not None:
        env_cfg.terminations["ladder_body_collision"].params[
            "force_threshold"
        ] = float(body_force_threshold)
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
        robot = base_env.scene["robot"]
        sensor = base_env.scene.sensors["ladder_foot_contact"]
        site_ids, _ = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
        if args.control_scope == "all":
            control_ids = list(range(len(robot.joint_names)))
        else:
            control_ids = [
                i
                for i, name in enumerate(robot.joint_names)
                if name.startswith(("left_", "right_"))
            ]
        print(
            "[cem] control_scope=",
            args.control_scope,
            "joints=",
            [robot.joint_names[i] for i in control_ids],
        )
        action_term = base_env.action_manager.get_term("joint_pos")
        target_rung = torch.full(
            (args.envs,), args.target_rung, dtype=torch.long, device=base_env.device
        )
        swing_id = int(
            microduck_mdp._ladder_target_foot_id(
                torch.tensor([args.target_rung], device=base_env.device),
                RUNG_COUNT,
            )[0].item()
        )
        support_id = 1 - swing_id
        goal = microduck_mdp._ladder_target_world(
            base_env,
            target_rung,
            RUNG_COUNT,
            0.18,
            0.035,
            0.72,
            0.67,
            0.065,
            0.012,
        )

        torch.manual_seed(args.seed)
        mean = torch.zeros((args.segments, len(control_ids)), device=base_env.device)
        std = torch.full_like(mean, args.initial_std)
        best_score = torch.full((args.envs,), -float("inf"), device=base_env.device)
        best_residual = torch.zeros_like(mean)
        best_summary: dict[str, float | int] = {}

        for iteration in range(args.iterations):
            normalized_residuals = mean[None] + std[None] * torch.randn(
                (args.envs, args.segments, len(control_ids)),
                device=base_env.device,
            )
            normalized_residuals = normalized_residuals.clamp(-1.0, 1.0)
            residuals = normalized_residuals * args.residual_scale
            base_env.reset(seed=args.seed + iteration)
            obs = env.get_observations()
            alive = torch.ones(args.envs, dtype=torch.bool, device=base_env.device)
            score = torch.zeros(args.envs, device=base_env.device)
            max_z = torch.full((args.envs,), -float("inf"), device=base_env.device)
            max_x = torch.full((args.envs,), -float("inf"), device=base_env.device)
            reached = torch.zeros(args.envs, dtype=torch.bool, device=base_env.device)

            for step in range(args.horizon):
                with torch.inference_mode():
                    if args.base_control == "hold" and base_env.action_manager.action is not None:
                        policy_actions = base_env.action_manager.action.clone()
                    else:
                        policy_actions = policy(obs)
                actions = policy_actions.detach().clone()
                segment = min(args.segments - 1, step * args.segments // args.horizon)
                actions[:, control_ids] += residuals[:, segment]
                prev_alive = alive.clone()
                obs, _, dones, _ = env.step(actions)
                feet = robot.data.site_pos_w[:, site_ids]
                swing = feet[:, swing_id]
                support = feet[:, support_id]
                origin = base_env.scene.terrain.env_origins
                swing_local = swing - origin
                root_speed = torch.linalg.vector_norm(
                    robot.data.root_link_lin_vel_b[:, :2], dim=-1
                )
                error = torch.linalg.vector_norm(swing - goal, dim=-1)
                max_z = torch.maximum(
                    max_z,
                    torch.where(
                        prev_alive,
                        swing_local[:, 2],
                        torch.full_like(swing_local[:, 2], -float("inf")),
                    ),
                )
                max_x = torch.maximum(
                    max_x,
                    torch.where(
                        prev_alive,
                        swing_local[:, 0],
                        torch.full_like(swing_local[:, 0], -float("inf")),
                    ),
                )
                found = sensor.data.found
                if found.dim() == 3:
                    found = found.any(dim=-1)
                support_contact = found[:, support_id] > 0
                com_y = robot.data.root_com_pos_w[:, 1]
                support_y = support[:, 1]
                transfer_score = torch.exp(
                    -torch.square(com_y - support_y) / 0.06**2
                )
                target = getattr(
                    base_env,
                    "_ladder_target_rung",
                    torch.zeros(args.envs, dtype=torch.long, device=base_env.device),
                )
                reached |= target >= args.target_rung + 1
                # Reward physical progress, but keep a live support foot and
                # penalize candidates that terminate before the next contact.
                live = prev_alive.float()
                score += live * (
                    -1.20 * error
                    + 20.0 * torch.clamp(swing_local[:, 2], min=0.0, max=0.20)
                    + 4.0 * torch.clamp(swing_local[:, 0] - 0.13, min=-0.10, max=0.16)
                    + 0.75 * support_contact.float()
                    + 0.75 * transfer_score * support_contact.float()
                    - 0.10 * root_speed
                )
                alive &= ~dones.bool()
                if reached.any():
                    score += reached.float() * 100.0
                    break

            score += 5.0 * torch.clamp(max_z, min=0.0, max=0.20)
            score += 2.0 * torch.clamp(max_x - 0.13, min=-0.10, max=0.16)
            score -= (~alive).float() * 2.0
            elite_n = min(args.elite_count, args.envs)
            elite_score, elite_idx = torch.topk(score, elite_n)
            elite = normalized_residuals[elite_idx]
            mean = elite.mean(dim=0)
            std = (0.90 * elite.std(dim=0) + 0.02).clamp(max=0.45)
            best_idx = int(torch.argmax(score).item())
            if float(score[best_idx]) > float(best_score.max()):
                best_score = score.detach().clone()
                best_residual = residuals[best_idx].detach().clone()
            best_summary = {
                "iteration": iteration + 1,
                "score": float(score[best_idx]),
                "elite_score": float(elite_score.mean()),
                "max_z": float(max_z[best_idx]),
                "max_x": float(max_x[best_idx]),
                "reached_count": int(reached.sum()),
                "alive_count": int(alive.sum()),
                "std": float(std.mean()),
                "residual_scale": args.residual_scale,
                "bootstrap_state": str(args.bootstrap_state) if args.bootstrap_state else None,
            }
            print("CEM", best_summary, flush=True)
            if int(reached.sum()) > 0:
                break

        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "residual": best_residual.detach().cpu(),
                "segments": args.segments,
                "horizon": args.horizon,
                "control_ids": control_ids,
                "summary": best_summary,
                "checkpoint": str(args.checkpoint),
            },
            args.output,
        )
        print("SAVED", args.output, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
