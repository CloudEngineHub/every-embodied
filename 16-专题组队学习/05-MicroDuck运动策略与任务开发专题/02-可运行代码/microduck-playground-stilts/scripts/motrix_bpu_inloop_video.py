#!/usr/bin/env python3
"""Record MotrixLab ball-balance video with every action produced by RDK X5 BPU.

MotrixLab owns the NumPy manager environment and MuJoCo/MotrixSim renderer.
The policy request crosses the same TCP protocol as the other MicroDuck
tasks, so the Ubuntu process never evaluates the actor locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


MAGIC = b"MDP1"
HELLO = struct.Struct("!4sHHB3x")
COUNT = struct.Struct("!I")


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("RDK closed the BPU connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class BpuClient:
    def __init__(self, host: str, port: int, obs_dim: int, action_dim: int = 14):
        self.sock = socket.create_connection((host, port), timeout=15.0)
        self.sock.settimeout(15.0)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sock.sendall(HELLO.pack(MAGIC, obs_dim, action_dim, 0))
        magic, got_obs, got_action, recurrent = HELLO.unpack(
            recv_exact(self.sock, HELLO.size)
        )
        if magic != MAGIC or got_obs != obs_dim or got_action != action_dim or recurrent:
            raise RuntimeError(
                f"RDK contract mismatch: got {magic!r}/{got_obs}/{got_action}/{recurrent}, "
                f"expected {obs_dim}/{action_dim}/non-recurrent"
            )
        self.latencies_ms: list[float] = []

    def infer(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(self.obs_dim)
        started = time.perf_counter()
        self.sock.sendall(COUNT.pack(1) + value.astype("<f4", copy=False).tobytes())
        (count,) = COUNT.unpack(recv_exact(self.sock, COUNT.size))
        if count != 1:
            raise RuntimeError(f"Unexpected BPU response batch: {count}")
        action = np.frombuffer(
            recv_exact(self.sock, self.action_dim * 4),
            dtype="<f4",
            count=self.action_dim,
        ).copy()
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        return np.clip(action, -1.0, 1.0)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(args: argparse.Namespace) -> None:
    # Imports are intentionally local: --help and protocol checks must work
    # without importing MotrixSim's native renderer.
    import motrix_envs  # noqa: F401, WPS433
    from motrix_env_core import registry
    from motrix_env_core.sim.registry import list_sim_backends

    # Installed wheels register this through an entry point.  The teaching
    # runner also supports a source checkout, where that metadata is absent.
    if not list_sim_backends():
        from motrix_env_motrixsim.register import register

        register()

    from motrix_env_core.renderer import create_renderer
    from motrix_env_core.sim.backend import RenderConfig

    env = registry.make(
        "microduck-ball-balance",
        num_envs=1,
        mode="play",
        seed=args.seed,
    )
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
        ),
    )
    if renderer is None:
        raise RuntimeError("Motrix environment did not provide a renderer")

    bpu = BpuClient(args.bpu_host, args.bpu_port, obs_dim=args.obs_dim)
    state = env.init_state()
    actual_obs_dim = int(state.obs.policy.shape[-1])
    if actual_obs_dim != args.obs_dim:
        raise RuntimeError(f"Motrix actor observation is {actual_obs_dim}D, expected {args.obs_dim}D")

    resets = 0
    try:
        for _ in range(args.steps):
            if not renderer.render():
                break
            action = bpu.infer(state.obs.policy[0])
            state = env.step(action.reshape(1, args.action_dim).astype(np.float32, copy=False))
            if not renderer.render():
                break
            if bool(state.done[0]):
                resets += 1
    finally:
        bpu.close()
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
                "backend": "RDK X5 BPU via hbm_runtime",
                "renderer": "Ubuntu MotrixSim headless renderer",
                "bpu_host": args.bpu_host,
                "bpu_port": args.bpu_port,
                "hbm_file": str(Path(args.hbm).resolve()),
                "hbm_sha256": sha256(Path(args.hbm)),
                "observation_dim_sent": args.obs_dim,
                "action_dim": args.action_dim,
                "steps": args.steps,
                "video_frames": frame_count,
                "resets": resets,
                "mean_round_trip_ms": float(np.mean(bpu.latencies_ms)) if bpu.latencies_ms else None,
                "p95_round_trip_ms": float(np.percentile(bpu.latencies_ms, 95)) if bpu.latencies_ms else None,
                "note": "MotrixSim renders on Ubuntu; RDK X5 BPU produces every FastSAC actor action.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PASS-BPU-VIDEO: {output}")
    print(f"PASS-BPU-REPORT: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpu-host", default="192.168.8.128")
    parser.add_argument("--bpu-port", type=int, default=8765)
    parser.add_argument("--hbm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--obs-dim", type=int, default=54)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.steps <= 0 or args.fps <= 0 or args.obs_dim <= 0 or args.action_dim <= 0:
        parser.error("steps, fps, obs-dim, and action-dim must be positive")
    if not args.hbm.is_file():
        parser.error(f"HBM not found: {args.hbm}")
    record(args)


if __name__ == "__main__":
    main()
