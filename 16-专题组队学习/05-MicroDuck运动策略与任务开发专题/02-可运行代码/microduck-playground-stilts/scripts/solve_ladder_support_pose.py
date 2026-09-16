"""Find a physically plausible two-foot pose for the next ladder rung.

This is a curriculum initializer, not a shortcut around the contact gate.  It
uses the compiled MuJoCo model to solve for a pose in which the current
support foot remains on its rung while the other foot is brought close to the
next planned foothold.  The resulting qpos is later released to the real
simulator with zero velocity and must still make contact and pass the normal
hold/upright/support checks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _find_id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    count = {
        mujoco.mjtObj.mjOBJ_SITE: model.nsite,
        mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
    }[obj]
    for index in range(int(count)):
        full = mujoco.mj_id2name(model, obj, index)
        if full and (full == name or full.endswith("/" + name)):
            return index
    raise KeyError(name)


def _quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_euler("xyz", euler_xyz).as_quat()
    return np.asarray((xyzw[3], xyzw[0], xyzw[1], xyzw[2]), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rung", type=int, default=1)
    parser.add_argument(
        "--target-foot",
        choices=("auto", "left", "right"),
        default="auto",
        help="explicit swing-foot override for a repeated-rung transfer",
    )
    parser.add_argument(
        "--support-rung",
        type=int,
        default=None,
        help="physical rung carrying the other foot; defaults to target-rung-1",
    )
    parser.add_argument(
        "--support-state",
        choices=("rung", "ground"),
        default="rung",
        help=(
            "where to keep the non-target support foot; use ground for the "
            "first transfer from flat ground to the lowest rung"
        ),
    )
    parser.add_argument(
        "--swing-state",
        choices=("rung", "source", "air"),
        default="rung",
        help=(
            "where to place the non-support foot: the target rung, its source "
            "pose (grounded transfer stage), or an elevated air pose"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root-target-x", type=float, default=0.19)
    parser.add_argument("--root-target-y", type=float, default=0.0)
    parser.add_argument("--root-target-z", type=float, default=0.20)
    parser.add_argument(
        "--root-target-euler",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=(0.0, 0.0, 0.0),
        help="desired root XYZ Euler angles in radians",
    )
    parser.add_argument("--target-x-offset", type=float, default=0.0)
    parser.add_argument("--target-z-offset", type=float, default=0.0)
    parser.add_argument(
        "--support-z-offset",
        type=float,
        default=0.0,
        help="additional vertical offset for the already-supporting foot",
    )
    parser.add_argument(
        "--contact-dist-target",
        type=float,
        default=-0.0015,
        help="desired MuJoCo foot/rung distance in metres (slight compression)",
    )
    parser.add_argument(
        "--body-clearance-target",
        type=float,
        default=0.0,
        help="minimum body/ladder distance in metres; positive values enforce a gap",
    )
    args = parser.parse_args()

    # Importing the task registers the physical ladder scene.  Keep this
    # script independent from the bootstrap reset event so the solved pose can
    # be inspected before it is used by PPO.
    os.environ["MICRODUCK_LADDER_ACTIVE_RUNGS"] = "2"
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks import mdp as microduck_mdp
    from mjlab_microduck.tasks.microduck_video_ladder_footstep_env_cfg import (
        RUNG_COUNT,
    )

    cfg = load_env_cfg("Mjlab-Video-Ladder-Footstep-MicroDuck", play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        payload = torch.load(args.input, map_location="cpu", weights_only=True)
        source_qpos = payload["qpos"].numpy().astype(np.float64)
        source_qvel = payload["qvel"].numpy().astype(np.float64)

        model = env.sim.mj_model
        data = mujoco.MjData(model)
        left_site = _find_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        right_site = _find_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        joint_names = tuple(env.scene["robot"].joint_names)
        joint_qpos_adr = np.asarray(
            [
                model.jnt_qposadr[
                    _find_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in joint_names
            ],
            dtype=np.int64,
        )
        body_geom_ids = []
        foot_geom_ids = {}
        ladder_geom_ids = []
        rung_geom_ids = {}
        for geom_id in range(int(model.ngeom)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name in {
                "robot/trunk_body_collision",
                "robot/left_leg_body_collision",
                "robot/right_leg_body_collision",
            }:
                body_geom_ids.append(geom_id)
            if name in {
                "robot/left_foot_collision",
                "robot/right_foot_collision",
            }:
                foot_geom_ids[name.rsplit("/", 1)[-1].removesuffix("_collision")] = geom_id
            if name.startswith("video_ladder_0_"):
                ladder_geom_ids.append(geom_id)
            if name.startswith("video_ladder_0_rung_"):
                rung_id = int(name.rsplit("_", 1)[-1])
                rung_geom_ids[rung_id] = geom_id

        data.qpos[:] = source_qpos
        mujoco.mj_forward(model, data)
        source_feet = data.site_xpos[[left_site, right_site]].copy()

        # Match mdp._ladder_target_world for env 0.  The site targets are
        # planner targets, not fake contacts: the solved state is deliberately
        # checked against the compiled MuJoCo contacts below.  The planner
        # alternates right, left, right, ... (see _LADDER_FIRST_SWING_FOOT),
        # so the target rung must be assigned to the corresponding foot.  A
        # previous version always assigned the target to the left foot; that
        # made rung 2 look complete in the scene while the state machine was
        # waiting for the right foot and produced the observed foot jam.
        def rung_target(rung_index: int, lateral: float) -> np.ndarray:
            # Keep this numerically identical to mdp._ladder_rung_fraction
            # and _ladder_target_world.  The scene starts at rung fraction 0
            # and reserves only the 5% margin at the top; using the old 8%
            # entry margin solves a point between the first two rungs.
            fraction = 0.95 * rung_index / max(RUNG_COUNT - 1, 1)
            return np.asarray(
                (
                    0.18 + fraction * (0.72 - 0.18) - 0.032,
                    lateral,
                    0.035 + fraction * (0.67 - 0.035) + 0.012,
                ),
                dtype=np.float64,
            )

        target_foot = (args.target_rung + 1) % 2  # 0=left, 1=right
        if args.target_foot != "auto":
            target_foot = 0 if args.target_foot == "left" else 1
        support_rung = (
            args.support_rung
            if args.support_rung is not None
            else max(args.target_rung - 1, 0)
        )
        # The transfer curriculum needs a real intermediate state: one foot
        # remains on the rung while the other foot is still on the floor.  The
        # old solver always constrained both feet to ladder geometry, which
        # produced a static two-foot pose but no way to teach COM transfer.
        if target_foot == 0:
            support_target = rung_target(support_rung, -0.065)
            swing_target = rung_target(args.target_rung, 0.065)
            left_target = swing_target
            right_target = support_target
        else:
            support_target = rung_target(support_rung, 0.065)
            swing_target = rung_target(args.target_rung, -0.065)
            left_target = support_target
            right_target = swing_target

        if args.support_state == "ground":
            # The first physical transfer starts with one foot on the floor.
            # Do not constrain that foot to a rung: doing so creates a static
            # two-rung pose that is not the state the policy must learn from.
            support_target = source_feet[1 - target_foot].copy()

        if args.swing_state == "source":
            swing_target = source_feet[target_foot].copy()
        elif args.swing_state == "air":
            swing_target = source_feet[target_foot].copy()
            swing_target[2] = max(float(swing_target[2]) + 0.07, 0.075)
        if target_foot == 0:
            left_target = swing_target
            right_target = support_target
        else:
            left_target = support_target
            right_target = swing_target

        target_offset = np.asarray(
            (args.target_x_offset, 0.0, args.target_z_offset), dtype=np.float64
        )
        support_offset = np.asarray(
            (0.0, 0.0, args.support_z_offset), dtype=np.float64
        )
        if target_foot == 0:
            left_target = left_target + target_offset
            right_target = right_target + support_offset
        else:
            right_target = right_target + target_offset
            left_target = left_target + support_offset

        euler0 = Rotation.from_quat(
            (source_qpos[4], source_qpos[5], source_qpos[6], source_qpos[3])
        ).as_euler("xyz")
        x0 = np.concatenate((source_qpos[:3], euler0, source_qpos[joint_qpos_adr]))

        joint_lower = np.full(len(joint_qpos_adr), -np.inf, dtype=np.float64)
        joint_upper = np.full(len(joint_qpos_adr), np.inf, dtype=np.float64)
        for i, name in enumerate(joint_names):
            jid = _find_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if model.jnt_limited[jid]:
                joint_lower[i], joint_upper[i] = model.jnt_range[jid]

        lower = np.concatenate(
            (
                [x0[0] - 0.10, x0[1] - 0.10, 0.085],
                [-0.40, -0.40, -0.40],
                joint_lower,
            )
        )
        upper = np.concatenate(
            (
                [x0[0] + 0.14, x0[1] + 0.10, 0.30],
                [0.40, 0.40, 0.40],
                joint_upper,
            )
        )

        def unpack(x: np.ndarray) -> np.ndarray:
            qpos = source_qpos.copy()
            qpos[:3] = x[:3]
            qpos[3:7] = _quat_wxyz(x[3:6])
            qpos[joint_qpos_adr] = x[6:]
            return qpos

        root_target = np.asarray(
            (args.root_target_x, args.root_target_y, args.root_target_z),
            dtype=np.float64,
        )

        def body_clearance(qpos: np.ndarray) -> np.ndarray:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            fromto = np.zeros(6, dtype=np.float64)
            penalties = []
            for body_geom in body_geom_ids:
                for ladder_geom in ladder_geom_ids:
                    distance = mujoco.mj_geomDistance(
                        model, data, body_geom, ladder_geom, 1.0, fromto
                    )
                    penalties.append(
                        max(0.0, float(args.body_clearance_target) - float(distance))
                        / 0.01
                    )
            return np.asarray(penalties, dtype=np.float64)

        def foot_contact_distance(qpos: np.ndarray) -> np.ndarray:
            """Constrain only the requested rung feet using real meshes."""
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            fromto = np.zeros(6, dtype=np.float64)
            pairs = ()
            if args.support_state == "rung":
                pairs = (
                    (("left_foot", support_rung),)
                    if target_foot == 1
                    else (("right_foot", support_rung),)
                )
            if args.swing_state == "rung":
                swing_pair = (
                    ("right_foot", args.target_rung)
                    if target_foot == 0
                    else ("left_foot", args.target_rung)
                )
                pairs = pairs + (swing_pair,)
            distances = []
            for foot_name, rung_id in pairs:
                foot_geom = foot_geom_ids[foot_name]
                rung_geom = rung_geom_ids[rung_id]
                distance = mujoco.mj_geomDistance(
                    model, data, foot_geom, rung_geom, 1.0, fromto
                )
                distances.append(
                    (float(distance) - float(args.contact_dist_target)) / 0.004
                )
            return np.asarray(distances, dtype=np.float64)

        def residual(x: np.ndarray) -> np.ndarray:
            qpos = unpack(x)
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            feet = data.site_xpos[[left_site, right_site]]
            return np.concatenate(
                (
                    (feet[0] - left_target) / 0.025,
                    (feet[1] - right_target) / 0.025,
                    (x[:3] - root_target) / np.asarray((0.08, 0.08, 0.06)),
                    (x[3:6] - np.asarray(args.root_target_euler)) / 0.25,
                    (x[6:] - x0[6:]) / 0.75,
                    body_clearance(qpos),
                    foot_contact_distance(qpos),
                )
            )

        # Reset snapshots can contain a transient yaw/tilt or a joint value
        # just outside the nominal XML range.  SciPy rejects such an initial
        # point before evaluating the residual, so project only the optimizer
        # start into the legal box; the solved state is still checked by
        # MuJoCo below.
        x_start = np.minimum(np.maximum(x0, lower + 1e-8), upper - 1e-8)
        clipped = np.flatnonzero(np.abs(x_start - x0) > 1e-7)
        if clipped.size:
            print("[support-pose] clipped initial variables:", clipped.tolist())

        result = least_squares(
            residual,
            x_start,
            bounds=(lower, upper),
            max_nfev=800,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        solved_qpos = unpack(result.x)
        data.qpos[:] = solved_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        solved_feet = data.site_xpos[[left_site, right_site]].copy()
        contacts = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
            contacts.append((geom1, geom2, float(contact.dist)))

        # Store the action history in the same convention as the position
        # action term.  This keeps the first PPO observation Markov-consistent.
        action_term = env.action_manager.get_term("joint_pos")
        offset = action_term.offset[0].detach().cpu().numpy()
        action = solved_qpos[joint_qpos_adr] - offset
        output = {
            "qpos": torch.from_numpy(solved_qpos).float(),
            "joint_pos": torch.from_numpy(solved_qpos[joint_qpos_adr]).float(),
            "qvel": torch.zeros_like(torch.from_numpy(source_qvel)).float(),
            "action": torch.from_numpy(action).float(),
            "target_rung": int(args.target_rung),
            "target_foot": int(target_foot),
            "support_state": args.support_state,
            "solver_success": bool(result.success),
            "solver_cost": float(result.cost),
            "left_target": left_target.tolist(),
            "right_target": right_target.tolist(),
            "left_foot": solved_feet[0].tolist(),
            "right_foot": solved_feet[1].tolist(),
            "contacts": contacts,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, args.output)
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
