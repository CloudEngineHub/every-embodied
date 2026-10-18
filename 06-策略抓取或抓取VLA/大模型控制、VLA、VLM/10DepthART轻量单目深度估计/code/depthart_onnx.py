"""Run the released DepthART MobileNetV4-M-slim-SPF graphs on CPU.

Model and preprocessing: https://github.com/xuefeng-cvr/DepthART
This tutorial runner adds verified downloads, Unicode paths and explicit timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


SOURCE_REVISION = "b307c123f19051e2d969960bba93ee3684874b90"
MODEL_REVISION = "9e154201cd8fb3453c9def581997aaf1b4da3c22"
HASHES = {
    224: "7b38623362745ef9d60405f68c679fcf05a5cfbec160c9b031fbf6983507aef2",
    448: "825114de5341612386de409f82f5dfb0496bb9afa559565ac633466bf172fb71",
}
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_file(directory: Path, resolution: int, offline: bool) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"relative_mnv4m_slim_spf_{resolution}_default.onnx"
    url = f"https://huggingface.co/Fengxue93/DepthART/resolve/{MODEL_REVISION}/onnx/{path.name}"
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"Offline mode: missing {path}")
        # A partial network transfer must never become a valid cached model.
        temporary = path.with_suffix(".onnx.part")
        print(f"Downloading {path.name}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if sha256(temporary) != HASHES[resolution]:
                raise ValueError("Downloaded model SHA256 differs from upstream CHECKSUMS.sha256")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    actual = sha256(path)
    if actual != HASHES[resolution]:
        raise ValueError(f"Cached model SHA256 mismatch: {path}; remove it and retry")
    return path, url


def preprocess(bgr: np.ndarray, resolution: int) -> np.ndarray:
    # Match relative/infer_image.py: square cubic resize, RGB and ImageNet norm.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None])


def restore(prediction: np.ndarray, height: int, width: int) -> np.ndarray:
    # PyTorch interpolate(..., align_corners=True), using identical corner coordinates.
    small = prediction[0, 0]
    map_x = np.linspace(0, small.shape[1] - 1, width, dtype=np.float32)
    map_y = np.linspace(0, small.shape[0] - 1, height, dtype=np.float32)
    x0, y0 = np.floor(map_x).astype(np.int32), np.floor(map_y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, small.shape[1] - 1), np.minimum(y0 + 1, small.shape[0] - 1)
    wx = (map_x - x0).astype(np.float32)
    wy = (map_y - y0).astype(np.float32)[:, None]
    horizontal = small[:, x0] * (1 - wx) + small[:, x1] * wx
    return (horizontal[y0] * (1 - wy) + horizontal[y1] * wy).astype(np.float32)


def colorize(prediction: np.ndarray) -> np.ndarray:
    low, high = np.percentile(prediction, (2, 98))
    value = np.clip((prediction - low) / max(high - low, 1e-6), 0, 1)
    return cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def write_png(path: Path, bgr: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    encoded.tofile(path)


def stats(values: list[float]) -> dict:
    milliseconds = np.asarray(values) * 1000
    return {
        "mean_ms": float(milliseconds.mean()),
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "fps_from_mean": float(1000 / milliseconds.mean()),
        "samples_ms": milliseconds.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(224, 448), default=224)
    parser.add_argument("--model-dir", type=Path, default=Path.home() / ".cache/every-embodied/depthart")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1 or args.threads < 1:
        parser.error("warmup >= 0, samples >= 1 and threads >= 1 are required")

    bgr = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode image: {args.image}")
    height, width = bgr.shape[:2]
    path, url = model_file(args.model_dir.expanduser(), args.resolution, args.offline)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    cv2.setNumThreads(1)
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()
    expected = [1, 3, args.resolution, args.resolution]
    if len(input_meta) != 1 or input_meta[0].shape != expected or input_meta[0].type != "tensor(float)":
        raise ValueError(f"Unexpected model input contract: {input_meta}")
    input_name = input_meta[0].name
    tensor = preprocess(bgr, args.resolution)
    for _ in range(args.warmup):
        session.run(None, {input_name: tensor})

    model_seconds, pipeline_seconds = [], []
    for _ in range(args.samples):
        started = time.perf_counter()
        prediction = session.run(None, {input_name: tensor})[0]
        model_seconds.append(time.perf_counter() - started)
    for _ in range(args.samples):
        started = time.perf_counter()
        prediction = session.run(None, {input_name: preprocess(bgr, args.resolution)})[0]
        restored = restore(prediction, height, width)
        colored = colorize(restored)
        pipeline_seconds.append(time.perf_counter() - started)

    if prediction.shape != (1, 1, args.resolution, args.resolution):
        raise ValueError(f"Unexpected output shape: {prediction.shape}")
    if not np.isfinite(restored).all() or float(np.ptp(restored)) <= 0:
        raise ValueError("Invalid prediction: non-finite or constant output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "relative_prediction.npy", restored)
    write_png(args.output_dir / "relative_prediction.png", colored)
    # Keep input and prediction together for visual inspection; colors have no meter scale.
    comparison = np.concatenate((bgr, colored), axis=1)
    write_png(args.output_dir / "comparison.png", comparison)

    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "DepthART MobileNetV4-M-slim-SPF relative",
        "upstream_source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "model_url": url,
        "model_sha256": sha256(path),
        "model_bytes": path.stat().st_size,
        "input_image": args.image.name,
        "input_image_sha256": sha256(args.image),
        "input_image_hw": [height, width],
        "tensor_shape": expected,
        "output_shape": list(restored.shape),
        "output_finite": bool(np.isfinite(restored).all()),
        "output_range": [float(restored.min()), float(restored.max())],
        "units": "affine-invariant relative prediction; not meters",
        "environment": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "providers": session.get_providers(),
            "ort_intra_op_threads": args.threads,
            "ort_inter_op_threads": 1,
            "opencv_threads": 1,
        },
        "warmup": args.warmup,
        "samples": args.samples,
        "model_only": stats(model_seconds),
        "pipeline": stats(pipeline_seconds),
        "timing_scope": {
            "model_only": "synchronous CPU session.run, prebuilt input tensor",
            "pipeline": "in-memory BGR -> RGB/resize/normalize -> session.run -> original-size bilinear restore -> percentile colorize",
            "excluded": "download, session startup, image decode, disk writes, camera capture and transport",
        },
        "accuracy_evaluation": "not performed; no ground truth in this single-image smoke test",
    }
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key]["mean_ms"] for key in ("model_only", "pipeline")}, indent=2))
    print(f"Saved {args.output_dir / 'comparison.png'} and benchmark.json")


if __name__ == "__main__":
    main()
