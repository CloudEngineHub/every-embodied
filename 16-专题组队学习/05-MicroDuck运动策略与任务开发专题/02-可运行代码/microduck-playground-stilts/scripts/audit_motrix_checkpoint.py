"""Compare the exact source checkpoint with ONNX on real simulator observations."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--motrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry
    from motrix_rl.fastsac.networks import Actor
    from motrix_rl.fastsac.buffer import EmpiricalNormalization

    model = onnx.load(args.onnx)
    metadata = {item.key: item.value for item in model.metadata_props}
    checkpoint = Path(metadata["source_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = args.motrix_root / checkpoint
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = OmegaConf.load(checkpoint.parent.parent / "task_config.yaml").algo.agent
    actor = Actor(54, 14, config.actor_hidden_dim, config.log_std_max, config.log_std_min,
                  use_tanh=config.use_tanh, use_layer_norm=config.use_layer_norm, device="cpu")
    actor.load_state_dict(state_dict["actor"], strict=True)
    actor.eval()
    normalizer = EmpiricalNormalization(54, device="cpu")
    normalizer.load_state_dict(state_dict["obs_normalizer"], strict=True)
    normalizer.eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    results = {}
    for backend in ("checkpoint", "onnx"):
        env = registry.make("microduck-ball-balance", num_envs=1, mode="play", seed=1)
        state = env.init_state()
        errors, reset_events, observations = [], [], []
        for step in range(args.steps):
            obs = np.asarray(state.obs.policy, dtype=np.float32).copy()
            observations.append(obs[0])
            with torch.inference_mode():
                reference = actor(normalizer(torch.from_numpy(obs), update=False))[0].numpy()
            exported = session.run(None, {session.get_inputs()[0].name: obs})[0]
            errors.append(float(np.max(np.abs(reference - exported))))
            state = env.step(reference if backend == "checkpoint" else exported)
            if state.done[0]:
                reset_events.append({"step": step + 1, "terms": [
                    name for name in env.termination_manager.terms if bool(state.metrics[name][0])
                ]})
        results[backend] = {"steps": args.steps, "max_abs_action_error": max(errors),
            "resets": len(reset_events), "reset_events": reset_events,
            "termination_counts": dict(Counter(name for event in reset_events for name in event["terms"]))}
        np.savez_compressed(args.output.with_name(args.output.stem + "_" + backend + ".npz"), obs=observations)
        close = getattr(env, "close", None)
        if callable(close):
            close()
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "source_checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        "training_global_step": state_dict["global_step"], "training_update_idx": state_dict["update_idx"],
        "normalization_restored": True, "rollouts": results}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
