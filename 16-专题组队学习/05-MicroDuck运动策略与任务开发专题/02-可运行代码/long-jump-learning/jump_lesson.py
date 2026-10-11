"""Pinned source, trusted model downloads and bounded experiment commands."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request


SOURCE_REVISION = "b6aed572587b52e41a452a9b2be517c95d2b3d96"
MODEL_REVISION = "fdcdc787e72b8784bb842159ee70607859cd17ed"
MODEL_REPO = "HannesVonEssen/microduck-long-jump"
TASK = "Mjlab-PlatformJump-MicroDuck"
HASHES = {
    "checkpoint.pt": "114ab38724b1bea789b0d3f12a1a18a5981b40572863117e212a2a0601f9ed4a",
    "policy.onnx": "cbd366e4ec6b95dd989d795cb73e1cf28731f85114c60368542dc9605a46fe06",
}
MODEL_FILES = (
    "checkpoint.pt", "policy.onnx", "config.json", "manifest.json",
    "SHA256SUMS", "LICENSE", "training/source.json", "training/resume.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_env(source=None, proxy=None, profile=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICRODUCK_")}
    env.update(MUJOCO_GL="egl", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               WANDB_MODE="disabled", HF_HUB_DISABLE_XET="1")
    if proxy:
        env.update(http_proxy=proxy, https_proxy=proxy, all_proxy=proxy)
    if source:
        paths = [str(Path(source) / "src"), str(Path(source) / "scripts")]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
    if profile:
        env.update(MICRODUCK_PJ_GAP="0.30", MICRODUCK_PJ_DROP="0.25")
    return env


def run_logged(args, log, *, cwd=None, env=None, timeout=1800):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(x) for x in args], cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout)
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"Command failed ({result.returncode}); log: {log}\n{tail}")
    return {"returncode": result.returncode, "seconds": round(time.monotonic() - started, 2),
            "log": str(log)}


def download_file(name, target, proxy=None):
    target = Path(target)
    if target.exists() and (name not in HASHES or sha256(target) == HASHES[name]):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {"https": proxy, "http": proxy} if proxy else {}))
    for attempt in range(3):
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                with opener.open(url, timeout=90) as response:
                    shutil.copyfileobj(response, stream)
            if name in HASHES and sha256(temporary) != HASHES[name]:
                raise ValueError(f"SHA-256 mismatch: {name}")
            temporary.replace(target)
            return
        except Exception:
            if "temporary" in locals():
                temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def prepare_source(home, proxy=None):
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    source = home / "source-release"
    marker = source / ".source_revision"
    if marker.exists():
        assert marker.read_text(encoding="utf-8").strip() == SOURCE_REVISION
        return source
    if source.exists():
        raise RuntimeError(f"Incomplete source directory; preserve it and use a new home: {source}")
    archive = home / "source.tar.gz"
    if not archive.exists():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(
            {"https": proxy, "http": proxy} if proxy else {}))
        url = f"https://codeload.github.com/Vottivott/microduck-playground/tar.gz/{SOURCE_REVISION}"
        partial = archive.with_suffix(".gz.part")
        with opener.open(url, timeout=300) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        partial.replace(archive)
    # Retain code, assets and task documentation, not the upstream video gallery.
    with tempfile.TemporaryDirectory(dir=home) as staging:
        with tarfile.open(archive, "r:gz") as package:
            prefix = f"microduck-playground-{SOURCE_REVISION}"
            members = []
            for member in package.getmembers():
                parts = Path(member.name).parts
                assert parts[0] == prefix
                rest = parts[1:]
                if not rest or len(rest) == 1 or rest[0] in ("src", "scripts", "tests") or (
                    rest[:2] == ("experiments", "parkour") and "media" not in rest
                ):
                    members.append(member)
            package.extractall(staging, members=members, filter="data")
        extracted = Path(staging) / prefix
        (extracted / ".source_revision").write_text(SOURCE_REVISION + "\n", encoding="utf-8")
        (extracted / ".archive_sha256").write_text(sha256(archive) + "\n", encoding="utf-8")
        extracted.replace(source)
    return source


def prepare_assets(home, proxy=None):
    home = Path(home).expanduser().resolve()
    source, model = prepare_source(home, proxy), home / "model"
    for name in MODEL_FILES:
        download_file(name, model / name, proxy)
    manifest = json.loads((model / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["revision"] == SOURCE_REVISION
    assert manifest["task"] == TASK
    # Verify every downloaded package file against the pinned publisher inventory.
    inventory = {}
    for line in (model / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        inventory[name.lstrip("*")] = digest
    for name in MODEL_FILES:
        if name != "SHA256SUMS":
            assert sha256(model / name) == inventory[name], name
    return source, model


def prepare_environment(source, home, proxy=None):
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    result = run_logged([uv, "sync", "--project", source, "--locked"],
                        Path(home) / "setup-logs/uv-sync.log", env=command_env(proxy=proxy))
    python = Path(source) / ".venv/bin/python"
    assert python.is_file(), python
    return python, result


def inspect_runtime(source, python, output):
    result_path = Path(output) / "runtime.json"
    run_logged([python, Path(__file__).with_name("jump_runtime.py"), "inspect",
                "--output", result_path], Path(output) / "runtime.log",
               cwd=source, env=command_env(source), timeout=300)
    return json.loads(result_path.read_text(encoding="utf-8"))


def render_episode(source, python, checkpoint, output, *, seed=28, backend="torch", onnx=None):
    output = Path(output)
    args = [python, Path(__file__).with_name("jump_runtime.py"), "render",
            "--checkpoint", checkpoint, "--output", output, "--seed", seed,
            "--backend", backend]
    if onnx:
        args += ["--onnx", onnx]
    env = command_env(source, profile=True)
    env["PYTHONHASHSEED"] = str(seed)
    timing = run_logged(args, output.with_suffix(".log"), cwd=source,
                        env=env, timeout=1200)
    report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    report["wall_seconds"] = timing["seconds"]
    return report


def continue_training(source, python, checkpoint, output, *, iterations=10, num_envs=64):
    """Keep only the latest lesson run; never replace the publisher checkpoint."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="candidate-", dir=output)) / "run"
    try:
        run_logged([python, Path(source) / "scripts/resume_release.py", "--recipe",
                    Path(source) / "experiments/parkour/resume.json", "--profile", "long-jump",
                    "--checkpoint", checkpoint, "--output", run,
                    "--num-envs", num_envs, "--iterations", iterations],
                   output / "training-latest.log", cwd=source,
                   env=command_env(source), timeout=1800)
        report = json.loads((run / "resume-validation.json").read_text(encoding="utf-8"))
        assert report["additional_iterations"] == iterations
        assert report["finite_all_training_steps"] and report["normalized_export"]
        for name in ("continued.pt", "policy.onnx", "resume-validation.json"):
            shutil.copy2(run / name, output / name)
        shutil.copytree(run / "params", output / "params", dirs_exist_ok=True)
        return report
    finally:
        shutil.rmtree(run.parent)


def verify_onnx(source, python, checkpoint, onnx, output):
    run_logged([python, Path(source) / "scripts/parkour_release.py", "long-jump", "verify",
                "--checkpoint", checkpoint, "--onnx", onnx, "--output", output],
               Path(output).with_suffix(".log"), cwd=source,
               env=command_env(source), timeout=600)
    return json.loads(Path(output).read_text(encoding="utf-8"))


def select_deployment_model(baseline, candidate, model_dir, training_dir, seconds=12):
    def passed(report):
        return (bool(report["landed_on_target"]) and bool(report["final_target_support"])
                and report["video_seconds"] >= seconds - 0.02
                and "fell" not in report["termination"])

    accepted = passed(candidate)
    if not accepted and not passed(baseline):
        raise RuntimeError("Baseline did not pass the landing/standing task assessment")
    root = Path(training_dir if accepted else model_dir)
    return {"candidate_accepted": accepted, "selected": "continued" if accepted else "pretrained",
            "checkpoint": str(root / ("continued.pt" if accepted else "checkpoint.pt")),
            "onnx": str(root / "policy.onnx"),
            "reason": "candidate_task_passed" if accepted else "candidate_task_failed_keep_baseline"}


def from_scratch_command(source, python, output, iterations=6000):
    return [str(python), "-m", "mjlab_microduck.train_cli", TASK,
            "--env.scene.num-envs", "4096", "--agent.max-iterations", str(iterations),
            "--agent.logger", "tensorboard", "--agent.run-name", "long-jump-from-scratch"]


def make_gif(video, gif):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install ffmpeg to generate the GitHub preview.")
    run_logged([ffmpeg, "-y", "-i", video, "-filter_complex",
                "fps=8,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                "-loop", "0", gif], Path(gif).with_suffix(".log"), timeout=300)
    return Path(gif)
