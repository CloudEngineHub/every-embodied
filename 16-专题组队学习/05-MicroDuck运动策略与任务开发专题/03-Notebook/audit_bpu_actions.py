"""Compare board actions with float ONNX on the exact recorded observations."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


FILES = {"basketball": "basketball", "walking": "walking", "perturbation": "walking",
         "stilt": "stilts25", "swing": "swing", "ball_balance": "motrix_ball_balance_restored"}


def compare(trace, model, clip_action=True):
    data = np.load(trace)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    states = {v.name: np.zeros(v.shape, dtype=np.float32) for v in inputs if v.name != "obs"}
    errors, reference_actions = [], []
    max_hidden = 0.0
    max_cell = 0.0
    for observation, action, done in zip(data["obs"], data["actions"], data["done"]):
        output = session.run(None, {"obs": observation.reshape(1, -1).astype(np.float32), **states})
        reference = output[0].reshape(14)
        if clip_action:
            reference = np.clip(reference, -1.0, 1.0)
        reference_actions.append(reference)
        errors.append(np.abs(reference - action))
        if states:
            names = [v.name for v in session.get_outputs()]
            states = {"h_in": output[names.index("h_out")], "c_in": output[names.index("c_out")]}
            max_hidden = max(max_hidden, float(np.abs(states["h_in"]).max()))
            max_cell = max(max_cell, float(np.abs(states["c_in"]).max()))
            if done:
                for value in states.values():
                    value.fill(0)
    errors = np.asarray(errors)
    return {"observation_samples": len(errors), "max_abs_action_error": float(errors.max()),
        "mean_abs_action_error": float(errors.mean()), "p95_abs_action_error": float(np.percentile(errors, 95)),
        "first_action_max_error": float(errors[0].max()),
        "float_obs_max_abs": float(np.abs(data["obs"]).max()),
        "float_h_max_abs": max_hidden, "float_c_max_abs": max_cell,
        "fraction_action_errors_over_0_1": float(np.mean(errors > 0.1)),
        "source": "same real observations, episode-aligned recurrent state, float ONNX vs recorded board actions"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("/home/ubuntu/workspaces/microduck_bpu_models/converted"))
    parser.add_argument("--video-root", type=Path, default=Path(__file__).resolve().parent / "outputs/bpu_videos")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "tasks": {}}
    for task, name in FILES.items():
        trace = args.video_root / f"{task}_bpu_latest.trace.npz"
        if not trace.is_file():
            raise FileNotFoundError(trace)
        report = json.loads(trace.with_name(f"{task}_bpu_latest.json").read_text())
        item = compare(trace, args.model_root / (name + ".opset11.onnx"), clip_action=task != "ball_balance")
        item.update(run_id=report["run_id"], board_model_sha256=report["hbm_sha256"],
                    resets=report.get("resets", report.get("terminated_or_truncated_resets")))
        result["tasks"][task] = item
        print(task, item)
    trace = args.video_root / "ladder_official_pair_bpu_latest.trace.npz"
    if trace.is_file():
        data = np.load(trace)
        report = json.loads(trace.with_name("ladder_official_pair_bpu_latest.json").read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        sessions = {role: ort.InferenceSession(str(args.model_root / f"official_ladder_{role}.opset11.onnx"),
            sess_options=options, providers=["CPUExecutionProvider"]) for role in ("climber", "getup")}
        switch = report["switch_events"][0]["step"] if report["switch_events"] else len(data["observations"])
        errors = []
        previous = np.zeros(14, dtype=np.float32)
        for step, (obs, action) in enumerate(zip(data["observations"], data["raw_actions"])):
            obs = obs.copy()
            obs[34:48] = previous
            session = sessions["getup" if step >= switch else "climber"]
            reference = session.run(None, {session.get_inputs()[0].name: obs.reshape(1, 61)})[0].reshape(14)
            errors.append(np.abs(reference - action))
            previous = action
        errors = np.asarray(errors)
        result["tasks"]["ladder"] = {"observation_samples": len(errors),
            "max_abs_action_error": float(errors.max()), "mean_abs_action_error": float(errors.mean()),
            "p95_abs_action_error": float(np.percentile(errors, 95)), "handoff_success": report["handoff_success"],
            "run_id": report["run_id"], "source": "same real observations and recorded previous raw actions"}
        print("ladder", result["tasks"]["ladder"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
