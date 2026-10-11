"""Render one complete physical episode and record its trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata as metadata
import json
from pathlib import Path


def inspect_runtime(output):
    import torch
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from jump_lesson import TASK
    env_cfg, agent = load_env_cfg(TASK, play=True), load_rl_cfg(TASK)
    checkpoint_schema = {"actor": asdict(agent.actor), "critic": asdict(agent.critic)}
    result = {"cuda_available": torch.cuda.is_available(), "task": TASK,
              "versions": {p: metadata.version(p) for p in
                           ("torch", "mjlab", "mujoco", "mujoco-warp", "onnx", "rsl-rl-lib")},
              "control_dt_s": env_cfg.sim.mujoco.timestep * env_cfg.decimation,
              "network": checkpoint_schema,
              "reward_names": [name for name, cfg in env_cfg.rewards.items() if cfg is not None],
              "termination_names": [name for name, cfg in env_cfg.terminations.items() if cfg is not None]}
    assert result["cuda_available"], "This lesson requires CUDA for policy inference and training."
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def render(args):
    import imageio.v2 as iio
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab_microduck.tasks import platform_jump_mdp as jump_terms
    from chimney_presentation import configure_video_cfg, fix_render_shadows, recolored_robot_cfg
    from jump_lesson import TASK, SOURCE_REVISION, sha256

    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = 12.0
    cfg.scene.entities["robot"] = recolored_robot_cfg(cfg.scene.entities["robot"], "cream")
    configure_video_cfg(cfg, width=720, height=720)
    cfg.viewer.azimuth, cfg.viewer.elevation, cfg.viewer.distance = 110, -12, 2.0
    raw = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array")
    agent = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames, trace, termination = 0, [], []
    try:
        runner = load_runner_cls(TASK)(env, asdict(agent), device="cuda:0")
        runner.load(str(args.checkpoint), map_location="cuda:0")
        actor = runner.get_inference_policy(device="cuda:0")
        session = None
        if args.backend == "onnx":
            import onnxruntime as ort
            session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        # Model construction/load may consume RNG; reset after it for paired evaluation.
        obs, _ = raw.reset(seed=args.seed)
        robot = raw.scene["robot"]
        fix_render_shadows(raw, light_dir=(-0.42, 0.11, -1.0))

        def record(step):
            position = robot.data.root_link_pos_w[0].detach().cpu().numpy()
            quaternion = robot.data.root_link_quat_w[0].detach().cpu().numpy()
            upright = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
            state = jump_terms._st(raw)
            target_contact = jump_terms._found(raw, jump_terms.FEET_B)[0]
            trace.append({"time_s": round(step * raw.step_dt, 4),
                          "x_m": float(position[0]), "y_m": float(position[1]),
                          "z_m": float(position[2]), "upright_cos": float(upright),
                          "jump_phase": int(state["phase"][0]),
                          "feet_on_target": int(target_contact.sum()),
                          "failed": bool(state["failed"][0])})

        with iio.get_writer(output, fps=50, codec="libx264", macro_block_size=None,
                            ffmpeg_params=["-movflags", "+faststart"]) as writer, torch.inference_mode():
            frame = raw.render()
            writer.append_data(frame)
            iio.imwrite(output.with_suffix(".start.png"), frame)
            frames += 1
            record(0)
            for step in range(round(12.0 / raw.step_dt)):
                if session:
                    action = session.run(None, {session.get_inputs()[0].name:
                                                obs["actor"].cpu().numpy()})[0]
                    assert action.shape == (1, 14) and np.isfinite(action).all()
                    action = torch.as_tensor(action, device="cuda:0")
                else:
                    action = actor(obs)
                assert torch.isfinite(action).all()
                obs, _, done, _ = env.step(action)
                assert torch.isfinite(raw.sim.data.qpos).all()
                if bool(done[0]):
                    termination = [name for name in raw.termination_manager.active_terms
                                   if bool(raw.termination_manager.get_term(name)[0])]
                    # mjlab auto-resets on done; never append a reset as a landing frame.
                    break
                record(step + 1)
                frame = raw.render()
                writer.append_data(frame)
                frames += 1
            iio.imwrite(output.with_suffix(".end.png"), frame)
        report = {"task": TASK, "source_revision": SOURCE_REVISION, "seed": args.seed,
                  "reset_after_model_load": True,
                  "backend": "PyTorch CUDA" if session is None else "ONNX Runtime CPU",
                  "physics": "MuJoCo Warp", "frames": frames, "fps": 50,
                  "video_seconds": frames / 50, "control_steps": len(trace) - 1,
                  "termination": termination, "checkpoint_sha256": sha256(args.checkpoint),
                  "gap_m": 0.30, "drop_m": 0.25, "trace": trace,
                  "final_position_m": [trace[-1][key] for key in ("x_m", "y_m", "z_m")],
                  "max_height_m": max(row["z_m"] for row in trace),
                  "landed_on_target": any(row["jump_phase"] == 2 for row in trace),
                  "final_target_support": trace[-1]["feet_on_target"] > 0,
                  "final_upright_cos": trace[-1]["upright_cos"],
                  "reached_time_limit": "time_out" in termination or "time_limit" in termination,
                  "finite_states": True}
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("EPISODE_SAVED", output, frames, termination)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("operation", choices=("inspect", "render"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--onnx", type=Path)
    args = parser.parse_args()
    if args.operation == "inspect":
        inspect_runtime(args.output)
    else:
        if not args.checkpoint or (args.backend == "onnx" and not args.onnx):
            parser.error("render needs a checkpoint, and ONNX mode also needs --onnx")
        render(args)


if __name__ == "__main__":
    main()
