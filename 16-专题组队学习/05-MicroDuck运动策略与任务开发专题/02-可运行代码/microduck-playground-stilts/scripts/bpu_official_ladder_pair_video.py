#!/usr/bin/env python3
"""Record the ladder policy pair with GPU, CPU, or RDK X5 inference.

Ubuntu owns MuJoCo physics, rendering, and the privileged simulator-only
handoff detector.  With the BPU backend, separate TCP sessions run the
climber and get-up HBMs on the RDK; CPU/GPU backends use the ONNX actors.
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
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.sock = socket.create_connection((host, port), timeout=15.0)
        self.sock.settimeout(15.0)
        self.sock.sendall(HELLO.pack(MAGIC, 61, 14, 0))
        magic, obs_dim, action_dim, recurrent = HELLO.unpack(
            recv_exact(self.sock, HELLO.size)
        )
        if (magic, obs_dim, action_dim, recurrent) != (MAGIC, 61, 14, 0):
            raise RuntimeError(
                f"RDK contract mismatch on {host}:{port}: "
                f"{magic!r}/{obs_dim}/{action_dim}/{recurrent}"
            )
        self.latencies_ms: list[float] = []

    def infer(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(61)
        if not np.isfinite(value).all():
            raise ValueError("non-finite official actor observation")
        started = time.perf_counter()
        self.sock.sendall(COUNT.pack(1) + value.astype("<f4", copy=False).tobytes())
        count = COUNT.unpack(recv_exact(self.sock, COUNT.size))[0]
        if count != 1:
            raise RuntimeError(f"unexpected BPU response batch: {count}")
        action = np.frombuffer(
            recv_exact(self.sock, 14 * 4), dtype="<f4", count=14
        ).copy()
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        if not np.isfinite(action).all():
            raise RuntimeError("RDK returned a non-finite action")
        # The official ladder policy emits joint-position offsets in radians;
        # do not apply the normalized-action clip used by walking policies.
        return action

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class OnnxClient:
    """Run one official actor with CPU ONNX or the teaching GPU executor."""

    def __init__(self, model: Path, backend: str = "cpu", policy_device: str = "cuda") -> None:
        self.backend = backend
        if backend == "gpu":
            from gpu_onnx_runtime import TorchOnnxPolicy

            self.policy = TorchOnnxPolicy(model, device=policy_device)
            self.latencies_ms = self.policy.latencies_ms
            return
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.latencies_ms: list[float] = []

    def infer(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(1, 61)
        if not np.isfinite(value).all():
            raise ValueError("non-finite official actor observation")
        if self.backend == "gpu":
            action = self.policy.infer(value.reshape(61))
            if not np.isfinite(action).all():
                raise RuntimeError("GPU ONNX returned a non-finite action")
            return action
        started = time.perf_counter()
        action = self.session.run(None, {self.input_name: value})[0]
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        action = np.asarray(action, dtype=np.float32).reshape(14)
        if not np.isfinite(action).all():
            raise RuntimeError("CPU ONNX returned a non-finite action")
        return action

    def close(self) -> None:
        if self.backend == "gpu":
            self.policy.close()
            return
        self.session = None


class OfficialPair:
    """Official policy-pair action contract from experiments/desk-climb."""

    def __init__(self, climber: object, getup: object, alpha_scale: float = 1.0) -> None:
        self.climber = climber
        self.getup = getup
        self.recovering = False
        self.raw = np.zeros(14, dtype=np.float32)
        self.executed = np.zeros(14, dtype=np.float32)
        self.alpha = np.full(14, 0.7, dtype=np.float32)
        # Official RUNTIME.md: neck/head use 0.5, legs use 0.7.
        self.alpha[5:9] = 0.5
        if not np.isfinite(alpha_scale) or not 0.0 < alpha_scale <= 1.0:
            raise ValueError("get-up alpha scale must be in (0, 1]")
        self.alpha *= float(alpha_scale)

    def step(self, observation: np.ndarray, eligible: bool) -> tuple[np.ndarray, bool]:
        value = np.asarray(observation, dtype=np.float32).reshape(61).copy()
        if not np.isfinite(value).all() or np.any(value[48:] != 0.0):
            raise ValueError("official ladder observation contract violated")
        # The official handoff contract feeds the previous raw policy output
        # back into slots 34:48, including after the filtered get-up handoff.
        value[34:48] = self.raw
        self.recovering = self.recovering or bool(eligible)
        self.raw = (self.getup if self.recovering else self.climber).infer(value)
        if self.recovering:
            self.executed = self.alpha * self.raw + (1.0 - self.alpha) * self.executed
        else:
            self.executed = self.raw.copy()
        return self.executed.copy(), self.recovering

    def close(self) -> None:
        self.climber.close()
        self.getup.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def foot_desktop_eligible(
    env: object,
    robot: object,
    mdp: object,
    floor_desk: object,
    env_index: int,
) -> tuple[bool, dict[str, object]]:
    """Replicate the official privileged simulator handoff detector."""
    import mujoco

    model = env.sim.mj_model
    geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        for index in range(model.ngeom)
    ]
    desktop_id = next(
        index for index, name in enumerate(geom_names) if name == "tread_31/desktop"
    )
    foot_ids = [
        next(
            index
            for index, name in enumerate(geom_names)
            if name == foot_name
        )
        for foot_name in (
            "robot/left_foot_collision",
            "robot/right_foot_collision",
        )
    ]
    contact_count = int(env.sim.data.nacon[0].item())
    contact = env.sim.data.contact
    geom_pairs = contact.geom[:contact_count]
    distances = contact.dist[:contact_count]
    world_ids = contact.worldid[:contact_count]
    feet = []
    for foot_id in foot_ids:
        hit = (
            (world_ids == env_index)
            & (distances <= 0)
            & (geom_pairs == desktop_id).any(dim=1)
            & (geom_pairs == foot_id).any(dim=1)
        )
        feet.append(bool(hit.any().item()))

    stair_state = mdp._stair_state(env)
    root = robot.data.root_link_pos_w[env_index] - env.scene.env_origins[env_index]
    root = root.clone()
    root[0] -= stair_state.x0[env_index]
    root[1] -= stair_state.y0[env_index]
    desk = next(record for record in floor_desk.RECORDS if record["name"] == "desktop")
    pos = desk["pos"]
    size = desk["size"]
    rx, ry, rz = (float(value) for value in root.detach().cpu())
    inside = (
        rx > pos[0] - size[0] + 0.04
        and rx < pos[0] + size[0] - 0.05
        and abs(ry) < size[1] - 0.05
        and rz > 0.66
    )
    eligible = bool(any(feet) and inside)
    return eligible, {
        "feet_on_desktop": feet,
        "root_ladder_relative": [float(value) for value in root.detach().cpu()],
        "inside_desktop": inside,
    }


def desktop_standing(
    robot: object,
    env_index: int,
    detector: dict[str, object],
) -> tuple[bool, dict[str, float]]:
    """Require a real, low-speed upright landing after the policy handoff."""
    gravity_z = float(robot.data.projected_gravity_b[env_index, 2].detach().cpu())
    linear_speed = float(
        robot.data.root_link_lin_vel_w[env_index].norm().detach().cpu()
    )
    angular_speed = float(
        robot.data.root_link_ang_vel_w[env_index].norm().detach().cpu()
    )
    feet = detector.get("feet_on_desktop", [False, False])
    standing = bool(
        all(feet)
        and detector.get("inside_desktop", False)
        and gravity_z < -0.82
        and linear_speed < 0.25
        and angular_speed < 2.5
    )
    return standing, {
        "gravity_z": gravity_z,
        "linear_speed": linear_speed,
        "angular_speed": angular_speed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, default=Path("/home/ubuntu/workspaces/microduck-playground-official/experiments/desk-climb"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("bpu", "cpu", "gpu"), default="bpu")
    parser.add_argument("--climber-hbm", type=Path, default=None)
    parser.add_argument("--getup-hbm", type=Path, default=None)
    parser.add_argument("--climber-onnx", type=Path, default=None)
    parser.add_argument("--getup-onnx", type=Path, default=None)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--bpu-host", default="192.168.8.128")
    parser.add_argument("--climber-port", type=int, default=8875)
    parser.add_argument("--getup-port", type=int, default=8876)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Number of official vector environments; 64 matches the published switch evidence.",
    )
    parser.add_argument(
        "--record-env",
        type=int,
        default=17,
        help="Vector environment rendered and controlled by the BPU policy pair.",
    )
    parser.add_argument("--physics-device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-distance", type=float, default=1.8)
    parser.add_argument("--camera-elevation", type=float, default=-10.0)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument(
        "--getup-alpha-scale",
        type=float,
        default=1.0,
        help="Scale the official get-up action filter alpha for BPU robustness.",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Optional npz path for selected-env observations and actions (diagnostic only).",
    )
    args = parser.parse_args()

    if args.backend == "bpu":
        if args.climber_hbm is None or args.getup_hbm is None:
            parser.error("BPU backend requires --climber-hbm and --getup-hbm")
    elif args.climber_onnx is None or args.getup_onnx is None:
        parser.error("CPU/GPU backend requires --climber-onnx and --getup-onnx")

    official_root = args.official_root.expanduser().resolve()
    source_root = official_root / "source"
    training_root = official_root / "training"
    # These are the fixed geometry choices used by the official desk-climb
    # evaluation patch.  geometry_patch imports them during module loading.
    os.environ.setdefault("ENDING_ROLE", "above")
    os.environ.setdefault("LADDER_SHIFT", "0.06")
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(training_root))

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.wrappers import VideoRecorder

    class SelectedVideoRecorder(VideoRecorder):
        """Record one selected vector environment instead of hard-coded env 0."""

        def __init__(self, *recorder_args: object, record_env: int, **recorder_kwargs: object) -> None:
            super().__init__(*recorder_args, **recorder_kwargs)
            self.record_env = record_env

        def _record_frame(self) -> None:
            if self._wrapped_env.render_mode == "rgb_array":
                frame = self._wrapped_env.render()
                if frame is not None:
                    rgb_frame = (
                        frame[self.record_env]
                        if isinstance(frame, np.ndarray) and frame.ndim == 4
                        else frame
                    )
                    self.current_video_frames.append(rgb_frame)

    import geometry_patch
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab_microduck.robot import floor_desk
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_desk_recovery_env_cfg import make_desk_recovery

    configure_torch_backends()
    torch.set_num_threads(1)
    cfg = make_desk_recovery(play=True)
    geometry_patch.configure_cfg(cfg)
    cfg.seed = args.seed
    if args.num_envs < 1 or not 0 <= args.record_env < args.num_envs:
        raise ValueError("--record-env must be within [0, --num-envs)")
    cfg.scene.num_envs = args.num_envs
    cfg.viewer.width = args.width
    cfg.viewer.height = args.height
    cfg.viewer.distance = args.camera_distance
    cfg.viewer.elevation = args.camera_elevation
    cfg.viewer.azimuth = args.camera_azimuth
    # The physics loop controls record_env; keep the offscreen renderer on the
    # same vector environment instead of silently recording env 0.
    cfg.viewer.env_idx = args.record_env
    cfg.auto_reset = False
    cfg.episode_length_s = args.seconds + 1.0
    cfg.terminations.pop("reached_top", None)
    cfg.events["reset_stair_ladder"].params.update(
        min_start_tread=0,
        max_start_tread=0,
        desk_probability=0.0,
        floor_spawn_prob=1.0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    steps = round(args.seconds / 0.02)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.physics_device, render_mode="rgb_array")
    recorder = SelectedVideoRecorder(
        env,
        video_folder=str(args.output.parent),
        step_trigger=lambda step: step == 0,
        video_length=steps,
        name_prefix=args.output.stem,
        disable_logger=False,
        record_env=args.record_env,
    )
    if args.backend == "bpu":
        climber = BpuClient(args.bpu_host, args.climber_port)
        getup = BpuClient(args.bpu_host, args.getup_port)
    else:
        climber = OnnxClient(args.climber_onnx, backend=args.backend, policy_device=args.policy_device)
        getup = OnnxClient(args.getup_onnx, backend=args.backend, policy_device=args.policy_device)
    pair = OfficialPair(climber, getup, alpha_scale=args.getup_alpha_scale)
    switch_events: list[dict[str, object]] = []
    contact_trace: list[dict[str, object]] = []
    termination_steps: list[int] = []
    termination_terms: list[dict[str, object]] = []
    stable_frames = 0
    max_stable_frames = 0
    stable_start_step: int | None = None
    stable_trace: list[dict[str, object]] = []
    trace_observations: list[np.ndarray] = []
    trace_raw_actions: list[np.ndarray] = []
    trace_executed_actions: list[np.ndarray] = []
    obs, _ = recorder.reset()
    robot = env.scene["robot"]
    base_gains = [actuator.kp_scale.clone() for actuator in robot.actuators]
    try:
        for step in range(steps):
            eligible, detector = foot_desktop_eligible(
                env, robot, mdp, floor_desk, args.record_env
            )
            if eligible and not pair.recovering:
                switch_events.append({"step": step, "seconds": step * 0.02, **detector})
            executed, recovering = pair.step(
                obs["actor"][args.record_env].detach().cpu().numpy(), eligible
            )
            if args.trace_output is not None:
                trace_observations.append(
                    obs["actor"][args.record_env].detach().cpu().numpy().copy()
                )
                trace_raw_actions.append(pair.raw.copy())
                trace_executed_actions.append(executed.copy())
            for actuator, gain in zip(robot.actuators, base_gains):
                actuator.kp_scale.copy_(gain * (0.8 if recovering else 1.0))
            # Keep the official vectorized scene shape while controlling and
            # rendering only the selected environment through the BPU pair.
            action = torch.zeros(args.num_envs, 14, device=env.device)
            action[args.record_env] = torch.from_numpy(executed).to(device=env.device)
            if hasattr(env, "_manual_reset_pending"):
                env._manual_reset_pending.zero_()
            obs, _, terminated, truncated, _ = recorder.step(action)
            if bool(terminated[args.record_env]) or bool(truncated[args.record_env]):
                termination_steps.append(step + 1)
                terms = []
                for name in cfg.terminations:
                    try:
                        if bool(env.termination_manager.get_term(name)[args.record_env]):
                            terms.append(name)
                    except (AttributeError, KeyError, IndexError, RuntimeError):
                        pass
                termination_terms.append({"step": step + 1, "terms": terms})
            if hasattr(env, "_manual_reset_pending"):
                env._manual_reset_pending.zero_()
            if recovering:
                post_eligible, post_detector = foot_desktop_eligible(
                    env, robot, mdp, floor_desk, args.record_env
                )
                standing, stability = desktop_standing(
                    robot, args.record_env, post_detector
                )
                if standing:
                    if stable_frames == 0:
                        stable_start_step = step + 1
                    stable_frames += 1
                    max_stable_frames = max(max_stable_frames, stable_frames)
                else:
                    stable_frames = 0
                if step % 25 == 0:
                    stable_trace.append(
                        {
                            "step": step + 1,
                            "standing": standing,
                            "eligible": post_eligible,
                            "consecutive_frames": stable_frames,
                            **stability,
                            **post_detector,
                        }
                    )
            if step % 25 == 0:
                contact_trace.append(
                    {
                        "step": step,
                        "recovering": recovering,
                        "root": [
                            float(value)
                            for value in robot.data.root_link_pos_w[args.record_env]
                            .detach()
                            .cpu()
                        ],
                        **detector,
                    }
                )
    finally:
        pair.close()
        recorder.close()

    generated = sorted(args.output.parent.glob(args.output.stem + "-step-0.mp4"))
    if not generated:
        raise FileNotFoundError(f"VideoRecorder did not create {args.output.stem}-step-0.mp4")
    if generated[-1] != args.output:
        generated[-1].replace(args.output)

    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace_output,
            observations=np.asarray(trace_observations, dtype=np.float32),
            raw_actions=np.asarray(trace_raw_actions, dtype=np.float32),
            executed_actions=np.asarray(trace_executed_actions, dtype=np.float32),
            dt=np.float32(0.02),
        )

    report = args.output.with_suffix(".json")
    report.write_text(
        json.dumps(
            {
                "task": "ladder-official-policy-pair",
                "task_id": "Mjlab-FloorDeskBlind-MicroDuck",
                "backend": (
                    "RDK X5 BPU via hbm_runtime"
                    if args.backend == "bpu"
                    else ("GPU ONNX via Torch CUDA" if args.backend == "gpu" else "CPU ONNX reference")
                ),
                "policy_device": "GPU" if args.backend == "gpu" else ("RDK BPU" if args.backend == "bpu" else "CPU"),
                "bpu_host": args.bpu_host,
                "climber_port": args.climber_port,
                "getup_port": args.getup_port,
                "climber_hbm": str(args.climber_hbm) if args.climber_hbm else None,
                "getup_hbm": str(args.getup_hbm) if args.getup_hbm else None,
                "climber_hbm_sha256": sha256(args.climber_hbm) if args.climber_hbm else None,
                "getup_hbm_sha256": sha256(args.getup_hbm) if args.getup_hbm else None,
                "climber_onnx": str(args.climber_onnx) if args.climber_onnx else None,
                "getup_onnx": str(args.getup_onnx) if args.getup_onnx else None,
                "trace_output": str(args.trace_output) if args.trace_output else None,
                "official_onnx_contract": {"observation_dim": 61, "action_dim": 14, "command_slots_zero": True},
                "runtime_contract": {"climber_filter": "none", "getup_filter_alpha_head": 0.5, "getup_filter_alpha_legs": 0.7, "getup_kp_ratio": 0.8},
                "steps": steps,
                "num_envs": args.num_envs,
                "record_env": args.record_env,
                "nominal_video_seconds": steps * 0.02,
                "switch_events": switch_events,
                "termination_steps": termination_steps,
                "termination_terms": termination_terms,
                "stable_start_step": stable_start_step,
                "max_stable_frames": max_stable_frames,
                "stable_seconds": max_stable_frames * 0.02,
                "handoff_success": bool(switch_events and max_stable_frames >= 100),
                "stable_trace": stable_trace,
                "contact_trace": contact_trace,
                "mean_climber_round_trip_ms": float(np.mean(climber.latencies_ms)) if climber.latencies_ms else None,
                "p95_climber_round_trip_ms": float(np.percentile(climber.latencies_ms, 95)) if climber.latencies_ms else None,
                "mean_getup_round_trip_ms": float(np.mean(getup.latencies_ms)) if getup.latencies_ms else None,
                "p95_getup_round_trip_ms": float(np.percentile(getup.latencies_ms, 95)) if getup.latencies_ms else None,
                "note": "Ubuntu runs MuJoCo and the privileged simulator-only foot-desktop handoff detector; the selected environment is controlled by either the RDK X5 HBM pair, the GPU ONNX actor, or the CPU ONNX reference, while inert background environments preserve the official vectorized scene shape.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PASS-OFFICIAL-LADDER-PAIR-VIDEO: {args.output}")
    print(f"PASS-OFFICIAL-LADDER-PAIR-REPORT: {report}")
    print(json.dumps({"switch_events": switch_events, "termination_steps": termination_steps}, ensure_ascii=False))


if __name__ == "__main__":
    main()
