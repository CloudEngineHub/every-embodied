"""Print the runtime joint/action layout used by the V3 ladder task."""

from __future__ import annotations

import argparse
import os

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ.setdefault("MICRODUCK_LADDER_ACTIVE_RUNGS", "2")
    cfg = load_env_cfg("Mjlab-Video-Ladder-Footstep-MicroDuck", play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        robot = env.scene["robot"]
        print("JOINT_NAMES", robot.joint_names)
        print("JOINT_IDS", robot.find_joints(".*", preserve_order=True))
        print("ACTION_MANAGER", env.action_manager)
        action_term = env.action_manager.get_term("joint_pos")
        print("ACTION_SCALE", action_term.scale)
        print("ACTION_OFFSET", action_term.offset)
        print("ACTION_TARGET_IDS", action_term.target_ids)
        print("ACTION_CLIP", getattr(action_term.cfg, "clip", None))
        print("DEFAULT_JOINT_POS", robot.data.default_joint_pos[0].detach().cpu().tolist())
        print("SIM_MODEL_TYPE", type(env.sim.mj_model))
        print("SIM_DATA_TYPE", type(env.sim.data))
        env.reset(seed=0)
        print("JOINT_POS", robot.data.joint_pos[0].detach().cpu().tolist())
        print("JOINT_VEL", robot.data.joint_vel[0].detach().cpu().tolist())
        print("ROOT_POS", robot.data.root_link_pos_w[0].detach().cpu().tolist())
        print("LEFT_FOOT", robot.data.site_pos_w[0, robot.find_sites(("left_foot",), preserve_order=True)[0]].detach().cpu().tolist())
        print("RIGHT_FOOT", robot.data.site_pos_w[0, robot.find_sites(("right_foot",), preserve_order=True)[0]].detach().cpu().tolist())
    finally:
        env.close()


if __name__ == "__main__":
    main()
