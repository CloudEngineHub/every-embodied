"""Capture a settled MuJoCo ladder state without changing its foothold target."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--active-rungs", type=int, default=3)
    parser.add_argument("--target-rung", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ["MICRODUCK_LADDER_ACTIVE_RUNGS"] = str(args.active_rungs)
    os.environ["MICRODUCK_LADDER_BOOTSTRAP_STATE"] = str(args.input)
    os.environ["MICRODUCK_LADDER_BOOTSTRAP_TARGET_RUNG"] = str(args.target_rung)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    task_id = "Mjlab-Video-Ladder-Footstep-MicroDuck"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(task_id)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    try:
        base_env.reset(seed=0)
        robot = base_env.scene["robot"]
        action_term = base_env.action_manager.get_term("joint_pos")
        # ``action_manager.action`` is reset-history state, not necessarily
        # the raw action that reproduces the loaded qpos.  Reconstruct the
        # normalized position action explicitly so the bootstrap replay holds
        # the captured pose instead of commanding the default offset pose.
        target_ids = action_term._target_ids
        hold_joint_pos = robot.data.joint_pos[:, target_ids].detach().clone()
        scale = torch.as_tensor(
            action_term._scale, device=base_env.device, dtype=hold_joint_pos.dtype
        ).clamp_min(1e-6)
        offset = torch.as_tensor(
            action_term._offset, device=base_env.device, dtype=hold_joint_pos.dtype
        )
        hold_action = (hold_joint_pos - offset) / scale
        for _ in range(max(args.steps, 0)):
            _, _, dones, _ = env.step(hold_action)
            if bool(dones[0].item()):
                raise RuntimeError(
                    "bootstrap state terminated while holding its reconstructed "
                    f"position action at settle step {_ + 1}"
                )

        output = {
            "qpos": base_env.sim.data.qpos[0].detach().cpu(),
            "qvel": base_env.sim.data.qvel[0].detach().cpu(),
            "action": hold_action[0].detach().cpu(),
            "target_rung": int(args.target_rung),
            "settle_steps": int(args.steps),
            "common_step": int(getattr(base_env, "common_step_counter", 0)),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, args.output)
        print(
            "captured",
            args.output,
            "root=",
            output["qpos"][:3].tolist(),
            "qvel_norm=",
            float(output["qvel"].norm()),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
