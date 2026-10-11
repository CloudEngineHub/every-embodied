"""Persist the source, quantization, and repeat-run stability evidence."""

from datetime import datetime, timezone
import json
from pathlib import Path

def build(folder):
    folder = Path(folder)
    diagnostics = folder / "outputs/diagnostics"
    source = json.loads((diagnostics / "motrix_checkpoint.json").read_text())
    bpu = json.loads((diagnostics / "bpu_action_comparison.json").read_text())
    latest = {}
    for backend, file in (("gpu", folder / "outputs/gpu_videos/ladder_gpu_latest.json"),
                          ("bpu", folder / "outputs/bpu_videos/ladder_official_pair_bpu_latest.json")):
        report = json.loads(file.read_text())
        latest[backend] = {k: report.get(k) for k in ("run_id", "generated_at_utc", "steps", "handoff_success", "stable_seconds", "physics_device", "switch_events")}
    model_root = Path("/home/ubuntu/workspaces/microduck_bpu_models")
    repeat_runs = []
    for name in ("official_cpu_physics_gpu_repeat", "precision_cpu_physics_bpu"):
        path = model_root / "ladder_stability" / (name + ".json")
        if path.is_file():
            report = json.loads(path.read_text())
            repeat_runs.append({"test": name, **{k: report.get(k) for k in ("run_id", "generated_at_utc", "steps", "handoff_success", "stable_seconds")}})
    def current(task, backend):
        report = json.loads((folder / "outputs" / (backend + "_videos") / f"{task}_{backend}_latest.json").read_text())
        return {k: report[k] for k in ("run_id", "generated_at_utc", "steps", "resets", "failure_resets", "timeout_resets", "reset_events", "onnx_sha256", "hbm_sha256", "board_model_sha256_verified") if k in report}
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "board": {"host": "192.168.8.128", "control_port": 8766, "policy_port": 8765,
                  "ladder_ports": [8875, 8876], "fresh_task_videos": 7},
        "ball_balance": {"training_global_step": source["training_global_step"],
            "checkpoint_sha256": source["checkpoint_sha256"], "onnx_sha256": source["onnx_sha256"],
            "rollouts": source["rollouts"],
            "latest": {backend: current("ball_balance", backend) for backend in ("gpu", "bpu")},
            "finding": "Restored the trained 5002-iteration checkpoint and normalization. Removed erroneous clipping of joint-target actions to [-1,1]. Calibrated INT16 with real GPU/BPU observations, external normalization and split dense weights. Short training outputs remain candidates, not automatic replacements."},
        "bpu_action_comparison": bpu["tasks"],
        "calibration_float_parity": {name: json.loads((model_root / "converted" / (name + ".float_parity.json")).read_text()) for name in (
            "ball_balance_restored_int16_normalized_split_expanded", "ladder_climber_recalibrated_int16_normalized_split", "ladder_getup_recalibrated_int16_normalized_split")},
        "basketball": {"calibration": "max, matched real observation/hidden/cell samples, all INT16",
            "calibration_sample_count": len(list((model_root / "calibration/basketball_recalibrated_int16/obs").glob("*.bin"))),
            "latest": {backend: current("basketball", backend) for backend in ("gpu", "bpu")},
            "finding": "Rebuilt calibration ranges for observations and LSTM states. Recurrent outputs are dequantized and requantized with the corresponding next-input scale and dtype rather than copied as INT8."},
        "ladder": {"repeat_runs": repeat_runs, "latest": latest,
            "finding": "Retained official climber/get-up observation and contact-switch logic. Use CPU contact physics for repeatability, GPU or board BPU for actions. Externally normalized INT16 models with split dense weights reduced deployment errors; success requires climbing, handoff and stable standing on the tabletop."},
        "reset_fixes": ["Reconnect recurrent BPU session on each episode reset.",
                        "Do not reset simulator twice when the environment already auto-resets.",
                        "Extend episode time limits past the video duration; report failure and timeout resets separately."],
    }
    (folder / "诊断记录.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STABILITY-AUDIT", len(repeat_runs) + len(latest), "ladder runs")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent)
