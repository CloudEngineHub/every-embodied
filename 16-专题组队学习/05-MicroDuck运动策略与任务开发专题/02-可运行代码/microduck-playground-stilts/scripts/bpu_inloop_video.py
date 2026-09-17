#!/usr/bin/env python3
"""Record a MuJoCo video while the action is produced by an RDK X5 BPU.

Ubuntu runs the task physics and EGL renderer.  Every environment step sends
the actor observation to ``rdk_bpu_policy_server.py`` over TCP; the RDK runs
the task-specific HBM and returns the 14-dimensional normalized action.  The
resulting MP4 is therefore a BPU-in-the-loop video, rather than a reference
GIF or a CPU ONNX replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import torch


MAGIC = b"MDP1"
HELLO = struct.Struct("!4sHHB3x")
COUNT = struct.Struct("!I")


TASKS = {
    "basketball": {
        "task_id": "Mjlab-Basketball-MicroDuck",
        "hbm": "basketball.bin",
        "obs_dim": 61,
        "command": (0.0, 0.0, 0.0),
        "name": "basketball-balance",
        "recurrent": True,
        "source_root": "/home/ubuntu/workspaces/microduck-playground-official",
    },
    "walking": {
        "task_id": "Mjlab-Velocity-Flat-MicroDuck",
        "hbm": "walking.bin",
        "obs_dim": 61,
        "command": (0.40, 0.0, 0.0),
        "name": "navigation",
    },
    "perturbation": {
        "task_id": "Mjlab-Velocity-Flat-MicroDuck",
        "hbm": "walking.bin",
        "obs_dim": 61,
        "command": (0.28, 0.0, 0.0),
        "name": "physical-perturbation",
    },
    "stilt": {
        "task_id": "Mjlab-Stilt-Flat-MicroDuck",
        "hbm": "stilts25.bin",
        "obs_dim": 61,
        "command": (0.16, 0.0, 0.0),
        "name": "stilts-25cm",
    },
    "swing": {
        "task_id": "Mjlab-SwingPump-MicroDuck",
        "hbm": "swing.bin",
        "obs_dim": 61,
        "command": (0.0, 0.0, 0.0),
        "name": "swing-pump",
    },
    "ladder": {
        "task_id": "Mjlab-Video-Ladder-Footstep-MicroDuck",
        "hbm": "ladder.bin",
        "obs_dim": 218,
        "command": (0.14, 0.0, 0.0),
        "name": "ladder-footstep",
    },
}


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
    def __init__(
        self,
        host: str,
        port: int,
        obs_dim: int,
        action_dim: int = 14,
        recurrent: bool = False,
    ):
        self.host = host
        self.port = port
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sock = socket.create_connection((host, port), timeout=15.0)
        self.sock.settimeout(15.0)
        recurrent_flag = int(recurrent)
        self.sock.sendall(HELLO.pack(MAGIC, obs_dim, action_dim, recurrent_flag))
        magic, got_obs, got_action, got_recurrent = HELLO.unpack(
            recv_exact(self.sock, HELLO.size)
        )
        if magic != MAGIC or got_obs != obs_dim or got_action != action_dim:
            raise RuntimeError(
                f"RDK contract mismatch: got {magic!r}/{got_obs}/{got_action}/{got_recurrent}, "
                f"expected {obs_dim}/{action_dim}"
            )
        self.recurrent = bool(got_recurrent)
        self.latencies_ms: list[float] = []

    def infer(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(self.obs_dim)
        started = time.perf_counter()
        self.sock.sendall(COUNT.pack(1) + value.astype("<f4", copy=False).tobytes())
        (count,) = COUNT.unpack(recv_exact(self.sock, COUNT.size))
        if count != 1:
            raise RuntimeError(f"Unexpected BPU response batch: {count}")
        action = np.frombuffer(
            recv_exact(self.sock, self.action_dim * 4), dtype="<f4", count=self.action_dim
        ).copy()
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        return np.clip(action, -1.0, 1.0)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def set_command(env_cfg: object, command_values: tuple[float, float, float]) -> None:
    commands = getattr(env_cfg, "commands", None)
    if commands is None or "twist" not in commands:
        return
    command = commands["twist"]
    vx, vy, wz = command_values
    if hasattr(command, "ranges"):
        command.ranges.lin_vel_x = (vx, vx)
        command.ranges.lin_vel_y = (vy, vy)
        command.ranges.ang_vel_z = (wz, wz)
    for name, value in (
        ("rel_standing_envs", 0.0),
        ("rel_heading_envs", 0.0),
        ("rel_turn_in_place_envs", 0.0),
    ):
        if hasattr(command, name):
            setattr(command, name, value)
    if hasattr(command, "heading_command"):
        command.heading_command = False
        if hasattr(command, "ranges"):
            command.ranges.heading = None


def add_perturbation(env: ManagerBasedRlEnv, step: int) -> None:
    """Apply two visible, deterministic pushes for the web perturbation task."""
    if step not in (75, 175):
        return
    base = env.scene["robot"]
    impulse = torch.zeros((1, 6), device=env.device, dtype=torch.float32)
    impulse[0, 0] = 0.55 if step == 75 else -0.45
    base.write_root_com_velocity_to_sim(impulse)
    print(f"perturbation: applied base velocity impulse at step {step}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_one(task_name: str, spec: dict[str, object], args: argparse.Namespace) -> Path:
    # Importing the package registers the custom task IDs.  Stilt morphology
    # must be selected before the task module creates its robot config.
    os.environ.setdefault("MICRODUCK_STILT_HEIGHT_CM", str(args.stilt_height_cm))
    os.environ.setdefault("MICRODUCK_STILT_BLEND", "0.5")
    source_root = Path(str(spec.get("source_root", ""))).expanduser()
    if source_root.is_dir() and (source_root / "src").is_dir():
        # Basketball lives in the official playground checkout, while the
        # other task configs live in the teaching checkout.  Insert the
        # selected source before importing mjlab_microduck so the task and its
        # robot assets come from one consistent package.
        sys.path.insert(0, str(source_root / "src"))
    # Import mjlab only after the selected MicroDuck source has been inserted.
    # mjlab may import the installed MicroDuck package as a plugin, so a
    # top-level import would make it impossible to select the basketball
    # checkout for just one task in a shared Python process.
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.wrappers import VideoRecorder

    import mjlab_microduck.tasks  # noqa: F401

    task_id = str(spec["task_id"])
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    if args.seed is not None:
        env_cfg.seed = args.seed
    if task_name == "ladder" and args.curriculum_step is not None:
        # The current ladder task exposes 224 actor columns; the HBM was built
        # against the stable first 218 columns.  The runner slices those exact
        # columns below, while the environment still computes all terms.
        pass
    set_command(env_cfg, tuple(spec["command"]))

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="rgb_array")
    recorder = VideoRecorder(
        env,
        video_folder=str(args.output.parent),
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        name_prefix=args.output.stem,
        disable_logger=False,
    )
    bpu = BpuClient(
        args.bpu_host,
        args.bpu_port,
        int(spec["obs_dim"]),
        recurrent=bool(spec.get("recurrent", False)),
    )
    obs, _ = recorder.reset()
    if task_name == "ladder" and args.curriculum_step is not None:
        env.common_step_counter = int(args.curriculum_step)

    terminated_count = 0
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
            action = bpu.infer(actor_obs)
            action_tensor = torch.from_numpy(action).reshape(1, 14)
            obs, _, terminated, truncated, _ = recorder.step(action_tensor)
            if bool(terminated[0]) or bool(truncated[0]):
                terminated_count += 1
                obs, _ = recorder.reset()
    finally:
        bpu.close()
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
                "task_id": task_id,
                "display_name": spec["name"],
                "backend": "RDK X5 BPU via hbm_runtime",
                "bpu_host": args.bpu_host,
                "bpu_port": args.bpu_port,
                "hbm_file": str(Path(args.hbm).resolve()),
                "hbm_sha256": sha256(Path(args.hbm)),
                "observation_dim_sent": int(spec["obs_dim"]),
                "action_dim": 14,
                "recurrent": bool(spec.get("recurrent", False)),
                "source_root": str(source_root) if source_root else None,
                "steps": args.steps,
                "terminated_or_truncated_resets": terminated_count,
                "mean_round_trip_ms": float(np.mean(bpu.latencies_ms)) if bpu.latencies_ms else None,
                "p95_round_trip_ms": float(np.percentile(bpu.latencies_ms, 95)) if bpu.latencies_ms else None,
                "note": "Ubuntu MuJoCo renders; RDK X5 BPU produces every policy action.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PASS-BPU-VIDEO: {args.output}")
    print(f"PASS-BPU-REPORT: {report}")
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "all"])
    parser.add_argument("--bpu-host", default="192.168.8.128")
    parser.add_argument("--bpu-port", type=int, default=8765)
    parser.add_argument("--hbm", default="")
    parser.add_argument("--output", type=Path, default=Path("outputs/bpu_video.mp4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bpu_videos"))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stilt-height-cm", type=float, default=25.0)
    parser.add_argument("--curriculum-step", type=int, default=None)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.task is None:
        parser.error("--task is required")

    selected = list(TASKS) if args.task == "all" else [args.task]
    for task_name in selected:
        spec = TASKS[task_name]
        hbm = Path(args.hbm) if args.hbm and args.task != "all" else Path(
            os.getenv("RDK_BPU_HBM_DIR", "/home/ubuntu/workspaces/microduck_bpu_models/board_hbm")
        ) / str(spec["hbm"])
        if not hbm.is_file():
            raise FileNotFoundError(
                f"HBM not found for {task_name}: {hbm}. Copy the board HBM to Ubuntu or pass --hbm."
            )
        args.hbm = str(hbm)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.task == "all":
            args.output = args.output_dir / f"{task_name}_bpu_latest.mp4"
        record_one(task_name, spec, args)


if __name__ == "__main__":
    main()
