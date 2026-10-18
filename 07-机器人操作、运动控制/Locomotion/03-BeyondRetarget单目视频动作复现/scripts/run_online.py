"""Run the official two-stage Space API; keep the same session for both stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    video = args.video.resolve(strict=True)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(video),
    ], check=True, capture_output=True, text=True)
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if not 0 < duration <= 10:
        parser.error("The official demo requires a video of at most 10 seconds.")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "official_hosted_inference",
        "space": "bear-ty/BeyondRetarget",
        "input_sha256": sha256(video),
        "input_duration_seconds": duration,
        "status": "started",
    }
    manifest_path = args.output / "run.json"
    started = time.monotonic()
    try:
        from gradio_client import Client, handle_file

        # State is stored server-side; a second Client would lose it.
        client = Client("bear-ty/BeyondRetarget", verbose=False,
                        download_files=str(args.output / "downloads"))
        print("Running GPU stage on the official Space...", flush=True)
        client.predict({"video": handle_file(str(video))},
                       api_name="/zero_gpu_inference")
        print("Rendering and packaging on the official Space...", flush=True)
        rendered, package = client.predict(api_name="/finish_cpu_stage")
        video_file = Path(rendered["video"] if isinstance(rendered, dict) else rendered)
        package_file = Path(package)
        shutil.copyfile(video_file, args.output / "eight_robots.mp4")
        shutil.copyfile(package_file, args.output / "motions.zip")
        manifest.update(status="completed", outputs={
            name: {"bytes": (args.output / name).stat().st_size,
                   "sha256": sha256(args.output / name)}
            for name in ("eight_robots.mp4", "motions.zip")
        })
    except Exception as exc:
        # Do not persist URLs, tokens, server paths, or exception bodies.
        manifest.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 2)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
