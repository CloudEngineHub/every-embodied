#!/usr/bin/env python3
"""Render the public rough-walk-g MicroDuck policy on a small rough course.

This is an offline material-generation script.  It keeps the public policy's
61-D observation / 14-D action contract and the BAM actuator model, while
building a deterministic MuJoCo course with low stairs, scattered blocks, and
a shallow ramp.  It is intentionally separate from the training environment:
the generated video is a reproduction/visual reference, not a claim that the
public ONNX policy was trained on this exact course.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer_policy import (
    PolicyInference,
    load_bam_model,
    load_mujoco_with_bam,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
# The public velocity policy is trained with the stripped walking collision
# model, not the full-body scene used for ladder/contact experiments.
BASE_SCENE = REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene_walk.xml"


def _geom(name: str, pos: tuple[float, float, float], size: tuple[float, float, float], rgba: str) -> str:
    x, y, z = pos
    sx, sy, sz = size
    return (
        f'        <geom name="{name}" type="box" pos="{x:.4f} {y:.4f} {z:.4f}" '
        f'size="{sx:.4f} {sy:.4f} {sz:.4f}" rgba="{rgba}" '
        'contype="1" conaffinity="1" friction="1.1 0.04 0.01" '
        'solref="0.018 1" solimp="0.92 0.98 0.008 0.5 2" />'
    )


def write_course_scene(path: Path, include_course: bool = True) -> None:
    """Create a modest, deterministic rough course in the base scene frame."""
    xml = BASE_SCENE.read_text(encoding="utf-8")
    geoms: list[str] = []
    if not include_course:
        path.write_text(xml, encoding="utf-8")
        return

    # Three low, broad steps.  Their height stays within the public model-card
    # regime (about 2 cm), so the policy is not asked to clear a new ladder.
    step_specs = [
        ("step_a", 0.70, 0.004),
        ("step_b", 0.98, 0.008),
        ("step_c", 1.26, 0.012),
    ]
    for name, x, h in step_specs:
        geoms.append(_geom(name, (x, 0.0, h / 2.0), (0.10, 0.48, h / 2.0), "0.17 0.42 0.30 1"))

    # Alternating small blocks force the feet to encounter uneven contact,
    # without turning the scene into an impassable obstacle field.
    blocks = [
        ("block_00", 1.55, -0.16, 0.008, 0.07, 0.09),
        ("block_01", 1.72, 0.13, 0.012, 0.06, 0.08),
        ("block_02", 1.90, -0.08, 0.010, 0.06, 0.11),
        ("block_03", 2.08, 0.18, 0.014, 0.07, 0.08),
        ("block_04", 2.26, -0.20, 0.011, 0.06, 0.09),
        ("block_05", 2.44, 0.06, 0.016, 0.07, 0.10),
    ]
    for name, x, y, h, sx, sy in blocks:
        geoms.append(_geom(name, (x, y, h / 2.0), (sx, sy, h / 2.0), "0.58 0.34 0.12 1"))

    # A shallow, stepped ramp at the end gives the clip a visible final change
    # of terrain while remaining below the 9-degree reference slope.
    for i in range(6):
        x = 2.40 + i * 0.12
        h = 0.004 + i * 0.004
        geoms.append(_geom(f"ramp_{i:02d}", (x, 0.0, h / 2.0), (0.065, 0.78, h / 2.0), "0.18 0.36 0.48 1"))

    marker = "</worldbody>"
    if marker not in xml:
        raise RuntimeError("base scene is missing </worldbody>")
    path.write_text(xml.replace(marker, "\n" + "\n".join(geoms) + "\n" + marker), encoding="utf-8")


def initialise_policy(model, data, policy: PolicyInference) -> None:
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = int(model.jnt_qposadr[freejoint_id])
    data.qpos[qpos_adr:qpos_adr + 3] = [0.0, 0.0, 0.125]
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    if policy.bam_ctrl is not None:
        policy.bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)


def render(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # MuJoCo resolves <include> paths relative to the scene file, so keep the
    # temporary file beside scene.xml and robot_allcollisions.xml.
    scene_path = BASE_SCENE.parent / ".codex-rough-walk-scene.xml"
    write_course_scene(scene_path, include_course=not args.flat)

    bam_model = load_bam_model(200.0, 7.4, None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        str(scene_path), bam_model, 0.005, 0.1, 6.0
    )
    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=str(args.policy),
        action_scale=1.0,
        bam_ctrl=bam_ctrl,
        use_projected_gravity=True,
        new_cmd_obs=True,
    )
    initialise_policy(model, data, policy)
    policy.set_vel_cmd(args.command, 0.0, 0.0)

    width, height = args.width, args.height
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth = 135.0
    camera.elevation = -22.0
    camera.distance = 1.9
    camera.lookat[:] = [0.75, 0.0, 0.08]

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None

    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    root_x_adr = int(model.jnt_qposadr[freejoint_id])
    physics_per_control = 4
    total_control_steps = int(round(args.seconds * args.fps))
    warmup_steps = int(round(args.warmup * args.fps))
    max_root_z = 0.0
    min_root_z = 10.0
    fallen = False

    try:
        for step in range(total_control_steps):
            if step == warmup_steps:
                policy.set_vel_cmd(args.command, 0.0, 0.0)
            action = policy.infer()
            policy.apply_action(action)
            for _ in range(physics_per_control):
                bam_ctrl.update()
                mujoco.mj_step(model, data)

            root_x = float(data.qpos[root_x_adr])
            root_z = float(data.qpos[root_x_adr + 2])
            max_root_z = max(max_root_z, root_z)
            min_root_z = min(min_root_z, root_z)
            if root_z < 0.075:
                fallen = True

            camera.lookat[0] = max(0.9, root_x + 0.65)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            ffmpeg.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        renderer.close()
        scene_path.unlink(missing_ok=True)

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    print(
        f"wrote {args.output} controls={total_control_steps} "
        f"root_x={float(data.qpos[root_x_adr]):.3f} "
        f"root_z=[{min_root_z:.3f},{max_root_z:.3f}] fallen={fallen}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--command", type=float, default=0.30)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--flat", action="store_true", help="render the same policy on the base flat floor")
    render(parser.parse_args())


if __name__ == "__main__":
    main()
