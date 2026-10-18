"""Validate motion arrays in NPZ files/directories/ZIPs without extracting ZIPs."""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def inspect(name, source):
    result = {"file": name, "errors": [], "arrays": {}}
    try:
        with np.load(source, allow_pickle=False) as data:
            for key in data.files:
                array = data[key]
                result["arrays"][key] = list(array.shape)
                if array.dtype.kind in "fciu" and not np.isfinite(array).all():
                    result["errors"].append(f"{key}: non-finite values")
            raw = {"root_trans", "root_rot_quat", "dof"}
            body = {"joint_pos", "body_pos_w", "body_quat_w"}
            keys = set(data.files)
            if raw <= keys:
                joint = data["dof"]
                shapes = {"root_trans": (3,), "root_rot_quat": (4,)}
                if "root_rot_6d" in keys:
                    shapes["root_rot_6d"] = (6,)
                quat_key = "root_rot_quat"
                result["schema"] = "root_and_dof"
            elif body <= keys:
                joint = data["joint_pos"]
                shapes = {"body_pos_w": None, "body_quat_w": None}
                quat_key = "body_quat_w"
                result["schema"] = "body"
            else:
                raise ValueError("unrecognized motion schema (contact-only files are not motion)")
            if joint.ndim != 2 or min(joint.shape) < 1:
                raise ValueError("joint array must have nonzero shape [T, D]")
            if joint.dtype.kind not in "fiu":
                raise ValueError("joint array must contain real numbers")
            frames, dofs = joint.shape
            result.update(frames=frames, dofs=dofs)
            for key, tail in shapes.items():
                value = data[key]
                if value.dtype.kind not in "fiu":
                    result["errors"].append(f"{key}: expected real numbers")
                if value.shape[0] != frames:
                    result["errors"].append(f"{key}: frame count differs from joints")
                if tail is not None and value.shape != (frames, *tail):
                    result["errors"].append(f"{key}: invalid shape")
            quat = data[quat_key]
            if quat.shape[-1] != 4 or not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=0.02):
                result["errors"].append(f"{quat_key}: invalid unit quaternion")
            if result["schema"] == "body":
                pos = data["body_pos_w"]
                if pos.ndim != 3 or pos.shape[-1] != 3 or quat.shape != (*pos.shape[:2], 4):
                    result["errors"].append("body arrays must have matching [T, B, 3/4] shapes")
            fps_array = data["fps"]
            if fps_array.size != 1:
                raise ValueError("fps must contain one number")
            fps = float(fps_array.reshape(-1)[0])
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("fps must be positive and finite")
            result.update(fps=fps, duration_seconds=frames / fps)
    except Exception as exc:
        result["errors"].append(str(exc))
    result["passed"] = not result["errors"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.input.suffix.lower() == ".zip":
        with zipfile.ZipFile(args.input) as archive:
            # Limit decompression and never extract server-supplied paths.
            entries = [e for e in archive.infolist() if e.filename.endswith(".npz")]
            if sum(e.file_size for e in entries) > 512 * 1024 * 1024:
                parser.error("ZIP motion payload exceeds 512 MiB")
            results = [inspect(e.filename, io.BytesIO(archive.read(e))) for e in entries]
    else:
        paths = sorted(args.input.rglob("*.npz")) if args.input.is_dir() else [args.input]
        results = [inspect(str(p.relative_to(args.input) if args.input.is_dir() else p.name), p) for p in paths]
    report = {"passed": bool(results) and all(r["passed"] for r in results),
              "files": results, "note": "Array validity does not establish dynamic executability."}
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
