#!/usr/bin/env python3
"""Record a MuJoCo video with the task actor evaluated on GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from bpu_inloop_video import TASKS, add_perturbation, set_command
from gpu_onnx_runtime import TorchOnnxPolicy


SUPPORTED_TASKS = {name: spec for name, spec in TASKS.items() if name != "ladder"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_one(task_name: str, spec: dict[str, object], args: argparse.Namespace) -> Path:
    os.environ.setdefault("MICRODUCK_STILT_HEIGHT_CM", str(args.stilt_height_cm))
    os.environ.setdefault("MICRODUCK_STILT_BLEND", "0.5")
    source_root = Path(str(spec.get("source_root", ""))).expanduser()
    if source_root.is_dir() and (source_root / "src").is_dir():
        sys.path.insert(0, str(source_root / "src"))

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.wrappers import VideoRecorder

    import mjlab_microduck.tasks  # noqa: F401

    env_cfg = load_env_cfg(str(spec["task_id"]), play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    env_cfg.viewer.distance = args.camera_distance
    env_cfg.viewer.elevation = args.camera_elevation
    env_cfg.viewer.azimuth = args.camera_azimuth
    if args.seed is not None:
        env_cfg.seed = args.seed
    set_command(env_cfg, tuple(spec["command"]))

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.physics_device, render_mode="rgb_array")
    recorder = VideoRecorder(
        env,
        video_folder=str(args.output.parent),
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        name_prefix=args.output.stem,
        disable_logger=False,
    )
    policy = TorchOnnxPolicy(args.onnx, device=args.policy_device)
    print("GPU 策略推理已启用；不展示具体显卡型号。")
    obs, _ = recorder.reset()
    terminated_count = 0
    reset_steps: list[int] = []
    longest_uninterrupted_steps = 0
    current_uninterrupted_steps = 0
    first_termination_step: int | None = None
    try:
        for step in range(args.steps):
            actor_obs = obs["actor"][0].detach().cpu().numpy()
            actor_obs = actor_obs[: int(spec["obs_dim"])]
            if actor_obs.size != int(spec["obs_dim"]):
                raise RuntimeError(
                    f"{task_name} actor obs is {actor_obs.size}D, expected {spec['obs_dim']}D"
                )
            if task_name == "perturbation":
                add_perturbation(env, step)
            action = np.clip(policy.infer(actor_obs), -1.0, 1.0)
            action_tensor = torch.from_numpy(action).reshape(1, 14).to(device=env.device)
            obs, _, terminated, truncated, _ = recorder.step(action_tensor)
            current_uninterrupted_steps += 1
            if bool(terminated[0]) or bool(truncated[0]):
                terminated_count += 1
                reset_steps.append(step + 1)
                longest_uninterrupted_steps = max(
                    longest_uninterrupted_steps, current_uninterrupted_steps
                )
                current_uninterrupted_steps = 0
                if first_termination_step is None:
                    first_termination_step = step + 1
                if args.stop_on_reset:
                    break
                policy.reset()
                obs, _ = recorder.reset()
    finally:
        policy.close()
        recorder.close()

    produced = sorted(args.output.parent.glob(args.output.stem + "-step-0.mp4"))
    if not produced:
        raise FileNotFoundError(f"VideoRecorder did not create {args.output.stem}-step-0.mp4")
    generated = produced[-1]
    if generated != args.output:
        generated.replace(args.output)
    report = args.output.with_suffix(".json")
    report.write_text(
        json.dumps(
            {
                "task": task_name,
                "task_id": spec["task_id"],
                "display_name": spec["name"],
                "backend": "GPU ONNX via Torch CUDA",
                "policy_device": "GPU",
                "onnx_file": str(args.onnx.resolve()),
                "onnx_sha256": sha256(args.onnx),
                "observation_dim": int(spec["obs_dim"]),
                "action_dim": 14,
                "recurrent": bool(spec.get("recurrent", False)),
                "steps": args.steps,
                "terminated_or_truncated_resets": terminated_count,
                "reset_steps": reset_steps,
                "first_termination_step": first_termination_step,
                "longest_uninterrupted_steps": max(
                    longest_uninterrupted_steps, current_uninterrupted_steps
                ),
                "longest_uninterrupted_time_s": max(
                    longest_uninterrupted_steps, current_uninterrupted_steps
                )
                * 0.02,
                "video_nominal_time_s": args.steps * 0.02,
                "mean_inference_ms": float(np.mean(policy.latencies_ms)) if policy.latencies_ms else None,
                "p95_inference_ms": float(np.percentile(policy.latencies_ms, 95)) if policy.latencies_ms else None,
                "note": "Ubuntu runs the MuJoCo physics and renderer; the task actor is evaluated by Torch on GPU.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PASS-GPU-VIDEO: {args.output}")
    print(f"PASS-GPU-REPORT: {report}")
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(SUPPORTED_TASKS), required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--physics-device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--stop-on-reset", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=1.4)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stilt-height-cm", type=float, default=25.0)
    args = parser.parse_args()
    if args.steps <= 0 or not args.onnx.is_file():
        parser.error("--steps must be positive and --onnx must exist")
    record_one(args.task, SUPPORTED_TASKS[args.task], args)


if __name__ == "__main__":
    main()
