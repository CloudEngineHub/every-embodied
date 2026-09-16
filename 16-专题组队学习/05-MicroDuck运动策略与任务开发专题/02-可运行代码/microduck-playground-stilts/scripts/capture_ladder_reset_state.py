"""Capture one deterministic standing reset for ladder pose solving."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ.setdefault("MICRODUCK_LADDER_ACTIVE_RUNGS", "2")
    os.environ.setdefault("MICRODUCK_LADDER_DOUBLE_SUPPORT", "0")
    os.environ.setdefault("MICRODUCK_LADDER_COMPILE_RUNGS", "1")
    os.environ.setdefault("MICRODUCK_LADDER_TOLERANT_BRIDGE", "1")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks import mdp as _microduck_mdp  # noqa: F401

    cfg = load_env_cfg("Mjlab-Video-Ladder-Footstep-MicroDuck", play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        env.reset(seed=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "qpos": env.sim.data.qpos[0].detach().cpu().clone(),
                "qvel": env.sim.data.qvel[0].detach().cpu().clone(),
            },
            args.output,
        )
        print(
            "wrote",
            args.output,
            "qpos=",
            tuple(env.sim.data.qpos.shape),
            "qvel=",
            tuple(env.sim.data.qvel.shape),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
