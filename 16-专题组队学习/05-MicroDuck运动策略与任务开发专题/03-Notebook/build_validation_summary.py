"""Collect inference evidence and decode every current demo video."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import nbformat
from nbformat.sign import NotebookNotary
import numpy as np
from PIL import Image, ImageDraw

from notebook_runtime import verify_gpu_report, verify_bpu_report
from patch_inference_cells import TASKS


def build(folder):
    folder = Path(folder).resolve()
    result = {"validated_at_utc": datetime.now(timezone.utc).isoformat(),
              "bpu": "not_run", "notebooks": []}
    videos = []
    notary = NotebookNotary()
    for path in sorted(folder.glob("[0-9][0-9]_*.ipynb")):
        book = nbformat.read(path, 4)
        code = [c for c in book.cells if c.cell_type == "code" and c.source.strip()]
        assert all(c.execution_count is not None and c.outputs for c in code)
        assert not any(o.output_type == "error" for c in code for o in c.outputs)
        notary.sign(book)
        item = {"notebook": path.name, "executed_code_cells": len(code), "cells_with_output": len(code)}
        prefix = path.name[:2]
        if prefix in TASKS:
            task = TASKS[prefix]
            text = "\n".join(o.get("text", "") for c in code for o in c.outputs)
            video = folder / "outputs/gpu_videos" / f"{task}_gpu_latest.mp4"
            report = json.loads(video.with_suffix(".json").read_text())
            models = ([report["climber_onnx"], report["getup_onnx"]] if task == "ladder"
                      else [report["onnx_file"]])
            verify_gpu_report(video, report["run_id"], models)
            assert report["run_id"] in text, (task, "Notebook GPU output is stale")
            item.update(task=task, backend=report["backend"], run_id=report["run_id"],
                        actual_policy_calls=report["steps"], numerical_validation=report["policy_validation"],
                        model_sha256=([report["climber_onnx_sha256"], report["getup_onnx_sha256"]]
                                      if task == "ladder" else report["onnx_sha256"]))
            item["task_result"] = {k: report[k] for k in (
                "terminated_or_truncated_resets", "resets", "longest_uninterrupted_time_s",
                "handoff_success", "stable_seconds") if k in report}
            videos.append((task, video))
            bpu_video = folder / "outputs/bpu_videos" / (
                "ladder_official_pair_bpu_latest.mp4" if task == "ladder" else f"{task}_bpu_latest.mp4")
            if "PASS-FRESH-BPU" in text:
                bpu_report = json.loads(bpu_video.with_suffix(".json").read_text())
                assert bpu_report["run_id"] in text, (task, "Notebook BPU output is stale")
                bpu_models = ([bpu_report["climber_hbm"], bpu_report["getup_hbm"]] if task == "ladder"
                              else [bpu_report["hbm_file"]])
                verify_bpu_report(bpu_video, bpu_report["run_id"], bpu_models,
                                  bpu_report["board_model_sha256_verified"])
                item["bpu"] = {k: bpu_report[k] for k in (
                    "run_id", "steps", "board_model_sha256_verified", "resets",
                    "terminated_or_truncated_resets", "recurrent_state_resets", "reset_events",
                    "handoff_success", "stable_seconds") if k in bpu_report}
                if task in {"basketball", "ball_balance"}:
                    assert bpu_report["failure_resets"] == 0 and bpu_report["timeout_resets"] == 0
                if task == "ladder":
                    assert bpu_report["handoff_success"] and bpu_report["stable_seconds"] >= 2
                result["bpu"] = "fresh_closed_loop_verified"
                videos.append((task + "_BPU", bpu_video))
        elif prefix == "00":
            report = json.loads((folder / "outputs/control_helloworld/control_report.json").read_text())
            item.update(backend="ONNX Runtime CPU + native MuJoCo", actual_policy_calls=report["steps"],
                        model_sha256=report["model_sha256"], phases=report["phases"])
            videos.append(("HelloWorld", folder / "outputs/control_helloworld/microduck_forward_turn_stop.mp4"))
        elif prefix == "08":
            nav_output = Path(os.getenv("LIGHTNAV_OUTPUT", "/data/Data14TB/lightnav-recordings/teaching/notebook"))
            reports = json.loads((nav_output / "lesson_summary.json").read_text())
            item.update(backend="LightNav GPU + walking actor", scenes={})
            for name, report in reports.items():
                item["scenes"][name] = {k: report[k] for k in (
                    "prediction_count", "sim_seconds", "termination", "initial_target_distance_m",
                    "final_target_distance_m")}
                assert report["prediction_count"] > 0
                videos.append((name, folder / "outputs/lightnav_gpu" / f"{name}_latest.mp4"))
        elif prefix == "09":
            report = json.loads((folder / "outputs/long_jump/lesson_summary.json").read_text())
            assets = folder.parent / "01-任务资料/08-MicroDuck跨缝跳跃/assets"
            published = dict(report)
            published["deployment_selection"] = {k: v for k, v in report["deployment_selection"].items()
                                                  if k not in {"checkpoint", "onnx"}}
            (assets / "validation.json").write_text(json.dumps(published, ensure_ascii=False, indent=2), encoding="utf-8")
            parity = json.loads((folder / "outputs/long_jump/onnx_parity_latest.json").read_text())
            (assets / "onnx_validation.json").write_text(json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8")
            item.update(backend="PyTorch CUDA / ONNX Runtime CPU", training_iterations=report["training"]["additional_iterations"],
                        deployment_selection=report.get("deployment_selection"),
                        task_result={k: {field: episode[field] for field in (
                            "video_seconds", "termination", "landed_on_target", "final_target_support")}
                                     for k, episode in report["episodes"].items()})
            assert item["training_iterations"] == 10
            for name in ("pretrained", "continued", "onnx"):
                videos.append(("jump_" + name, folder / "outputs/long_jump" / f"long_jump_{name}_latest.mp4"))
        result["notebooks"].append(item)
    output = folder / "outputs/validation/visual_checks"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, video in videos:
        decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(video), "-f", "null", "-"],
                                 capture_output=True, text=True, timeout=180)
        assert decoded.returncode == 0 and not decoded.stderr.strip(), (label, decoded.stderr)
        with imageio.get_reader(video) as reader:
            metadata = reader.get_meta_data()
            count = reader.count_frames()
            assert count > 0 and metadata["duration"] > 0
            row = Image.new("RGB", (960, 205), "white")
            ImageDraw.Draw(row).text((4, 3), f"{label} | {count} frames | {metadata['duration']:.2f} s", fill="black")
            for col, fraction in enumerate((0.1, 0.5, 0.9)):
                frame = reader.get_data(min(count - 1, int(count * fraction)))
                assert np.isfinite(frame).all() and frame.std() > 2, (label, "blank video")
                image = Image.fromarray(frame)
                image.thumbnail((320, 180))
                row.paste(image, (col * 320, 25))
            rows.append(row)
        print("VIDEO-OK", label, count, metadata["duration"])
    for start in range(0, len(rows), 5):
        sheet = Image.new("RGB", (960, 205 * len(rows[start:start + 5])), "white")
        for index, row in enumerate(rows[start:start + 5]):
            sheet.paste(row, (0, index * 205))
        sheet.save(output / f"sheet_{start // 5}.jpg")
    result["total_executed_code_cells"] = sum(n["executed_code_cells"] for n in result["notebooks"])
    result["decoded_video_count"] = len(videos)
    (folder / "验证记录.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("VERIFIED", len(result["notebooks"]), "notebooks", result["total_executed_code_cells"], "cells")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent)
