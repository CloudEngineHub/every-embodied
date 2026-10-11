"""Resolve one task actor and verify evidence from a new simulation run."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


MODEL_FILES = {
    "basketball": "basketball.opset11.onnx",
    "perturbation": "walking.opset11.onnx",
    "walking": "walking.opset11.onnx",
    "stilt": "stilts25.opset11.onnx",
    "swing": "swing.opset11.onnx",
    "ball_balance": "motrix_ball_balance_restored.opset11.onnx",
    "ladder": "official_ladder_climber.opset11.onnx",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_model(task, topic, hint=None):
    explicit = [os.getenv(key) for key in ("MICRODUCK_GPU_ONNX", "MICRODUCK_ONNX")]
    explicit = [Path(p).expanduser().resolve() for p in explicit if p]
    if len(set(explicit)) > 1:
        raise ValueError("MICRODUCK_ONNX and MICRODUCK_GPU_ONNX must select the same actor")
    if task == "ladder" and os.getenv("MICRODUCK_GPU_LADDER_CLIMBER"):
        explicit = [Path(os.environ["MICRODUCK_GPU_LADDER_CLIMBER"]).expanduser().resolve()]
    root = Path(os.getenv("MICRODUCK_MODEL_ROOT", os.getenv(
        "MICRODUCK_BPU_MODEL_ROOT", "/home/ubuntu/workspaces/microduck_bpu_models"))).expanduser()
    if explicit:
        candidates = explicit
    else:
        candidates = [root / "converted" / MODEL_FILES[task]]
        if hint:
            candidates.append(Path(topic) / hint)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"No {task} actor found: {candidates}. Set MICRODUCK_ONNX to this task's ONNX; "
        "model preparation is documented in the task tutorial."
    )


def policy_python(playground):
    configured = os.getenv("MICRODUCK_POLICY_PYTHON")
    candidates = [Path(configured)] if configured else [
        Path(playground) / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python"),
        Path("/home/ubuntu/workspaces/microduck_rl/.venv/bin/python"),
        Path(sys.executable),
    ]
    probe = (
        "import torch,mjlab,onnx,onnxruntime; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'"
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        result = subprocess.run([str(candidate), "-c", probe], capture_output=True, text=True, timeout=90)
        if result.returncode == 0:
            return candidate.absolute()
    raise RuntimeError(
        "No CUDA/mjlab/ONNX environment found. Set MICRODUCK_POLICY_PYTHON to "
        "the task environment's Python, or prepare the playground environment first."
    )


def inspect_policy(path, python):
    source = """
import json,sys,onnx
m=onnx.load(sys.argv[1]); onnx.checker.check_model(m)
def shape(v):
    return [d.dim_value if d.dim_value else (d.dim_param or '?') for d in v.type.tensor_type.shape.dim]
print(json.dumps({'inputs':[(v.name,shape(v)) for v in m.graph.input],
    'outputs':[(v.name,shape(v)) for v in m.graph.output],
    'nodes':len(m.graph.node),'metadata':{v.key:v.value for v in m.metadata_props}}))
"""
    result = subprocess.run([str(python), "-c", source, str(path)],
                            capture_output=True, text=True, check=True, timeout=90)
    report = json.loads(result.stdout)
    report["onnx_sha256"] = sha256(path)
    if report["outputs"][0][1][-1] != 14:
        raise ValueError("Task actor must return 14 joint actions")
    return report


def local_model_check(path, python):
    source = """
import json,sys,numpy as np,onnxruntime as ort
opt=ort.SessionOptions(); opt.intra_op_num_threads=1; opt.inter_op_num_threads=1
s=ort.InferenceSession(sys.argv[1],sess_options=opt,providers=['CPUExecutionProvider'])
feeds={v.name:np.zeros([d if isinstance(d,int) and d>0 else 1 for d in v.shape],dtype=np.float32) for v in s.get_inputs()}
outputs=s.run(None,feeds)
assert all(np.isfinite(o).all() for o in outputs), 'Non-finite model output'
assert outputs[0].shape[-1]==14, 'Expected 14 actions'
print(json.dumps({'providers':s.get_providers(),'inputs':[(v.name,v.shape) for v in s.get_inputs()],
    'outputs':[(v.name,v.shape) for v in s.get_outputs()],'finite':True}))
"""
    result = subprocess.run([str(python), "-c", source, str(path)],
                            capture_output=True, text=True, check=True, timeout=90)
    return json.loads(result.stdout)


def verify_gpu_report(video, run_id, models):
    video = Path(video)
    if not video.is_file() or video.stat().st_size == 0:
        raise FileNotFoundError(video)
    report = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))
    if report.get("run_id") != run_id or report.get("policy_device") != "GPU":
        raise ValueError("Report is not from the current GPU run")
    expected = [sha256(path) for path in models]
    if len(models) == 1:
        recorded = [report.get("onnx_sha256")]
        validations = [report.get("policy_validation", {})]
    else:
        recorded = [report.get("climber_onnx_sha256"), report.get("getup_onnx_sha256")]
        validations = list((report.get("policy_validation") or {}).values())
    if expected != recorded:
        raise ValueError("Simulator did not use the selected task model")
    calls = sum(v.get("inference_calls", 0) for v in validations)
    checks = sum(v.get("real_observation_parity_checks", 0) for v in validations)
    if calls <= 0 or calls != report.get("steps") or checks <= 0:
        raise ValueError("Missing actual policy calls or live-observation parity checks")
    if report.get("requested_steps", calls) != calls:
        raise ValueError("Simulation ended before the requested inference steps")
    for validation in validations:
        error = validation.get("max_abs_output_error", float("nan"))
        if not math.isfinite(error):
            raise ValueError("Missing finite GPU/reference comparison result")
    report["fresh_inference_verified"] = True
    return report


def verify_bpu_report(video, run_id, models, board_hashes):
    video = Path(video)
    if not video.is_file() or video.stat().st_size == 0:
        raise FileNotFoundError(video)
    report = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))
    if report.get("run_id") != run_id or report.get("backend") != "RDK X5 BPU via hbm_runtime":
        raise ValueError("Report is not from the current BPU run")
    expected = [sha256(model) for model in models]
    recorded = ([report.get("hbm_sha256")] if len(models) == 1 else
                [report.get("climber_hbm_sha256"), report.get("getup_hbm_sha256")])
    if recorded != expected or list(board_hashes) != expected:
        raise ValueError("Board model and selected video model do not match")
    calls = report.get("steps", 0)
    if calls <= 0 or report.get("requested_steps", calls) != calls:
        raise ValueError("BPU loop did not finish all requested policy calls")
    report.update(fresh_inference_verified=True, board_model_sha256_verified=list(board_hashes))
    video.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
