#!/usr/bin/env python3
"""Record the FastSAC ball-balance video with the actor evaluated on GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from gpu_onnx_runtime import TorchOnnxPolicy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(args: argparse.Namespace) -> None:
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry
    from motrix_env_core.renderer import create_renderer
    from motrix_env_core.sim.backend import RenderConfig
    from motrix_env_core.sim.registry import list_sim_backends

    if not list_sim_backends():
        from motrix_env_motrixsim.register import register

        register()

    env = registry.make("microduck-ball-balance", num_envs=1, mode="play", seed=args.seed)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(round(args.steps * float(env.cfg.ctrl_dt) * args.fps)))
    renderer = create_renderer(
        env,
        RenderConfig(
            headless=True,
            path=output,
            fps=args.fps,
            num_frames=frame_count,
            width=args.width,
            height=args.height,
            camera_lookat=(0.0, 0.0, 0.2),
            camera_distance=args.camera_distance,
            camera_elevation=args.camera_elevation,
            camera_azimuth=args.camera_azimuth,
        ),
    )
    if renderer is None:
        raise RuntimeError("Motrix environment did not provide a renderer")
    policy = TorchOnnxPolicy(args.onnx, device=args.policy_device)
    print("GPU 策略推理已启用；不展示具体显卡型号。")
    state = env.init_state()
    actual_obs_dim = int(state.obs.policy.shape[-1])
    if actual_obs_dim != args.obs_dim:
        raise RuntimeError(f"Motrix actor observation is {actual_obs_dim}D, expected {args.obs_dim}D")
    resets = 0
    try:
        for _ in range(args.steps):
            if not renderer.render():
                break
            raw_obs = state.obs.policy[0]
            if hasattr(raw_obs, "detach"):
                raw_obs = raw_obs.detach().cpu().numpy()
            action = policy.infer(np.asarray(raw_obs, dtype=np.float32))
            state = env.step(action.reshape(1, args.action_dim).astype(np.float32, copy=False))
            if not renderer.render():
                break
            if bool(state.done[0]):
                resets += 1
                policy.reset()
    finally:
        policy.close()
        renderer.close()
        close = getattr(env, "close", None)
        if callable(close):
            close()

    if not output.is_file() or output.stat().st_size == 0:
        raise FileNotFoundError(f"Motrix renderer did not create {output}")
    report = output.with_suffix(".json")
    report.write_text(
        json.dumps(
            {
                "task": "ball_balance",
                "task_id": "microduck-ball-balance",
                "display_name": "Motrix FastSAC ball balance",
                "backend": "GPU ONNX via Torch CUDA",
                "policy_device": "GPU",
                "renderer": "Ubuntu MotrixSim headless renderer",
                "onnx_file": str(args.onnx.resolve()),
                "onnx_sha256": sha256(args.onnx),
                "observation_dim": args.obs_dim,
                "action_dim": args.action_dim,
                "steps": args.steps,
                "video_frames": frame_count,
                "resets": resets,
                "mean_inference_ms": float(np.mean(policy.latencies_ms)) if policy.latencies_ms else None,
                "p95_inference_ms": float(np.percentile(policy.latencies_ms, 95)) if policy.latencies_ms else None,
                "note": "MotrixSim renders on Ubuntu; the FastSAC actor is evaluated by Torch on GPU.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PASS-GPU-VIDEO: {output}")
    print(f"PASS-GPU-REPORT: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=1.2)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--obs-dim", type=int, default=54)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.steps <= 0 or not args.onnx.is_file():
        parser.error("--steps must be positive and --onnx must exist")
    record(args)


if __name__ == "__main__":
    main()
