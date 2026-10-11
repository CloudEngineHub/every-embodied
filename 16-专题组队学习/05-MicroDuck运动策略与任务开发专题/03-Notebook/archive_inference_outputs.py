"""Keep executed outputs and generate portable previews from this run's videos."""

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil

import imageio.v2 as imageio
import nbformat
from PIL import Image

from patch_inference_cells import TASKS


def media_specs():
    for prefix, task in TASKS.items():
        for backend in ("gpu", "bpu"):
            name = ("ladder_official_pair_bpu_latest.mp4"
                    if task == "ladder" and backend == "bpu"
                    else f"{task}_{backend}_latest.mp4")
            yield prefix, backend.upper(), f"{backend}_videos", name, f"{task}_{backend}.gif"
    yield "00", "CPU", "control_helloworld", "microduck_forward_turn_stop.mp4", "forward_turn_stop.gif"
    for name in ("official_apartment", "living_room", "corridor"):
        yield "08", "GPU", "lightnav_gpu", f"{name}_latest.mp4", f"{name.replace('_', '-')}.gif"
    for name in ("pretrained", "continued", "onnx"):
        yield "09", "CPU" if name == "onnx" else "GPU", "long_jump", f"long_jump_{name}_latest.mp4", f"long_jump_{name}.gif"


def publish_recordings(folder, destination):
    entries = []
    for prefix, backend, group, name, preview in media_specs():
        source = folder / "outputs" / group / name
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(source)
        relative = Path("recordings") / group / name
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if group == "control_helloworld":
            report_source = source.parent / "control_report.json"
        elif group == "lightnav_gpu":
            nav_root = Path(os.getenv("LIGHTNAV_OUTPUT", "/data/Data14TB/lightnav-recordings/teaching/notebook"))
            report_source = nav_root / "lesson_summary.json"
        else:
            report_source = source.with_suffix(".json")
        report = json.loads(report_source.read_text(encoding="utf-8"))
        report_target = target.parent / report_source.name
        shutil.copy2(report_source, report_target)
        entries.append({
            "notebook_prefix": prefix, "backend": backend,
            "video": relative.as_posix(), "preview": preview,
            "report": report_target.relative_to(destination).as_posix(),
            "source_video": source.relative_to(folder).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "run_id": report.get("run_id"),
        })
    by_preview = {entry["preview"]: entry for entry in entries}
    used = []
    for path in sorted(destination.glob("[0-9][0-9]_*.ipynb")):
        notebook = nbformat.read(path, 4)
        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                cell.source = cell.source.replace(
                    "完整视频写入 `outputs/`，不提交到 Git。",
                    "新生成的视频写入 `outputs/`；本轮发布视频与运行报告保存在 `recordings/`。")
                cell.source = cell.source.replace(
                    "本 Notebook 只做契约检查，不从本地 14 级任务导出替代模型",
                    "本单元检查这两个模型的接口，后续分别接入 GPU/BPU 与 MuJoCo 完成爬梯、登桌和起身")
            if cell.cell_type != "code":
                continue
            outputs = []
            for output in cell.outputs:
                markdown = output.get("data", {}).get("text/markdown", "")
                matches = [entry for preview, entry in by_preview.items() if f"/{preview})" in markdown]
                if not matches:
                    outputs.append(output)
                    continue
                assert len(matches) == 1
                entry = matches[0]
                assert entry["notebook_prefix"] == path.name[:2]
                video, report = entry["video"], entry["report"]
                output.data["text/markdown"] = markdown + f"\n\n[播放 / 下载 MP4]({video}) · [运行报告]({report})"
                output.data["text/plain"] = f"{entry['backend']} inference video: {video}; report: {report}"
                outputs.append(output)
                outputs.append(nbformat.v4.new_output("display_data", data={
                    "text/html": f'<video controls preload="metadata" width="720" style="max-width:100%;height:auto;"><source src="{escape(video, quote=True)}" type="video/mp4"></video>',
                    "text/plain": f"MP4: {video}",
                }))
                used.append(video)
            cell.outputs = outputs
        notebook.metadata["published_media"] = "relative-mp4-with-gif-preview"
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
    assert set(used) == {entry["video"] for entry in entries}, "Every recording needs a Notebook playback entry"
    manifest = {"archived_at_utc": datetime.now(timezone.utc).isoformat(),
                "video_count": len(entries), "total_video_bytes": sum(e["bytes"] for e in entries),
                "videos": entries}
    (destination / "recordings/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 本次 Notebook 推理视频", "", "视频与 Notebook 中保存的本次运行输出配套；GIF 用于页面预览，MP4 保留完整录制过程。", "",
             "| Notebook | 推理端 | MP4 | 运行报告 |", "| --- | --- | --- | --- |"]
    for entry in entries:
        video = Path(entry["video"]).relative_to("recordings").as_posix()
        report = Path(entry["report"]).relative_to("recordings").as_posix()
        lines.append(f"| {entry['notebook_prefix']} | {entry['backend']} | [{Path(video).stem}]({video}) | [JSON]({report}) |")
    lines.extend(["", "文件体积与 SHA-256 见 [manifest.json](./manifest.json)。", ""])
    (destination / "recordings/README.md").write_text("\n".join(lines), encoding="utf-8")
    print("PUBLISHED-MP4", len(entries), "videos", round(manifest["total_video_bytes"] / 2**20, 2), "MiB")


def gif(video, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_reader(video) as reader:
        fps = reader.get_meta_data()["fps"]
        frames = []
        for index, frame in enumerate(reader):
            if index % max(1, round(fps / 6)):
                continue
            im = Image.fromarray(frame)
            im.thumbnail((360, 270))
            im = im.quantize(colors=80, dither=Image.Dither.NONE)
            frames.append(im)
    assert frames
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=round(1000 * max(1, round(fps / 6)) / fps), loop=0,
                   optimize=True)


def archive(topic, destination=None):
    topic = Path(topic).resolve()
    folder = topic / "03-Notebook"
    destination = Path(destination).resolve() if destination else folder / "outputs/published_notebooks"
    if destination == folder:
        raise ValueError("Publish to a separate destination to preserve the live executed notebooks")
    destination.mkdir(parents=True, exist_ok=True)
    if destination != folder:
        for path in folder.glob("[0-9][0-9]_*.ipynb"):
            shutil.copy2(path, destination / path.name)
    for path in sorted(folder.glob("0[1-7]_*.ipynb")):
        notebook = nbformat.read(path, 4)
        task = TASKS[path.name[:2]]
        video = folder / "outputs/gpu_videos" / f"{task}_gpu_latest.mp4"
        report = json.loads(video.with_suffix(".json").read_text())
        assert report.get("run_id") and report.get("policy_validation")
        preview = topic / "01-任务资料/精选视频/assets/notebook-replays" / f"{task}_gpu.gif"
        gif(video, preview)
        image = "../" + preview.relative_to(topic).as_posix()
        bpu_video = folder / "outputs/bpu_videos" / (
            "ladder_official_pair_bpu_latest.mp4" if task == "ladder" else f"{task}_bpu_latest.mp4")
        bpu_preview = preview.with_name(f"{task}_bpu.gif")
        bpu_report = json.loads(bpu_video.with_suffix(".json").read_text())
        assert bpu_report.get("fresh_inference_verified") and bpu_report.get("run_id")
        gif(bpu_video, bpu_preview)
        bpu_image = "../" + bpu_preview.relative_to(topic).as_posix()
        codes = [c for c in notebook.cells if c.cell_type == "code"]
        assert all(c.execution_count and c.outputs for c in codes)
        for cell in codes:
            for out in cell.outputs:
                assert out.output_type != "error"
                if "<video" in out.get("data", {}).get("text/html", ""):
                    is_gpu = "GPU_TASK =" in cell.source
                    assert is_gpu or "BPU_TASK =" in cell.source
                    image_path = image if is_gpu else bpu_image
                    backend = "GPU" if is_gpu else "BPU"
                    out.data = {"text/markdown": f"![本次{task} {backend}推理]({image_path})",
                                "text/plain": f"Fresh {task} inference preview; MP4 generated when rerun"}
        notebook.metadata["teaching"] = {"executed_code_cells": len(codes),
            "video_preview": "relative-gif", "inference_run_id": report["run_id"],
            "bpu_validation": "fresh_closed_loop_verified", "bpu_run_id": bpu_report["run_id"]}
        nbformat.validate(notebook)
        nbformat.write(notebook, destination / path.name)
        print("ARCHIVED", path.name)
    path = next(folder.glob("08_*.ipynb"))
    notebook = nbformat.read(path, 4)
    for name in ("official_apartment", "living_room", "corridor"):
        video = folder / "outputs/lightnav_gpu" / f"{name}_latest.mp4"
        preview = topic / "01-任务资料/精选视频/assets/lightnav" / f"{name.replace('_', '-')}.gif"
        gif(video, preview)
        image = "../" + preview.relative_to(topic).as_posix()
        for cell in notebook.cells:
            for out in cell.get("outputs", []):
                html = out.get("data", {}).get("text/html", "")
                if "<video" in html and f"{name}_latest.mp4" in html:
                    out.data = {"text/markdown": f"![本次导航：{name}]({image})",
                                "text/plain": f"Fresh LightNav inference: {name}"}
    nbformat.validate(notebook)
    nbformat.write(notebook, destination / path.name)
    print("ARCHIVED", path.name)
    for prefix in ("00", "09"):
        path = next(folder.glob(prefix + "_*.ipynb"))
        notebook = nbformat.read(path, 4)
        if prefix == "00":
            previews = {"microduck_forward_turn_stop.mp4": topic / "01-任务资料/07-RDK端侧与网页部署/assets/hello_world/forward_turn_stop.gif"}
        else:
            previews = {f"long_jump_{name}_latest.mp4": topic / "01-任务资料/08-MicroDuck跨缝跳跃/assets" / f"long_jump_{name}.gif" for name in ("pretrained", "continued", "onnx")}
        for cell in notebook.cells:
            for out in cell.get("outputs", []):
                html = out.get("data", {}).get("text/html", "")
                if "<video" not in html:
                    continue
                matches = [preview for name, preview in previews.items() if name in html]
                assert len(matches) == 1 and matches[0].is_file(), (path, "Missing video preview")
                relative = "../" + matches[0].relative_to(topic).as_posix()
                out.data = {"text/markdown": f"![本次推理]({relative})", "text/plain": "Fresh inference preview; MP4 generated when rerun"}
        nbformat.validate(notebook)
        nbformat.write(notebook, destination / path.name)
        print("ARCHIVED", path.name)

    publish_recordings(folder, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    archive(Path(__file__).resolve().parents[1], args.destination)
