"""Pinned upstream demos. Standard library only; no credentials are stored here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "upstream.json").read_text(encoding="utf-8"))
DEFAULT_SOURCE = Path(os.environ.get("AGNES_DEMO_SOURCE", HERE.parents[2] / "third_party" / "agnes-harness"))
ACTIONS = {
    "build": ["pnpm", "--filter", "@agnes/cli", "build:local"],
    "mhs-build": ["pnpm", "--filter", "@agnes/mhs", "build:plugin"],
    "devices": ["pnpm", "--filter", "@agnes/mhs", "dev-hub"],
    "mars-manual": ["pnpm", "--filter", "@agnes/mars-world", "start"],
    "mars-world": ["pnpm", "--filter", "@agnes/mars-world", "dev"],
    "mhs-install": ["pnpm", "--filter", "@agnes/mhs", "plugin:install"],
    "web": ["node", "packages/cli/dist/local/agnes.mjs", "serve"],
    "plugins-test": ["pnpm", "exec", "vitest", "run", "tools/public-docs/examples.test.ts", "--maxWorkers=1"],
    "plugins-smoke": ["node", "--import", "tsx", "tools/public-docs/smoke.mjs"],
}
VERIFIERS = {"verify-office": "office", "verify-mars": "mars", "verify-plugins": "plugins"}


def run(command, cwd, env=None, capture=False):
    command = [str(x) for x in command]
    executable = shutil.which(command[0])
    if not executable:
        raise RuntimeError(f"Missing executable: {command[0]}")
    command[0] = executable
    result = subprocess.run(command, cwd=cwd, env=env, check=True,
                            text=True, encoding="utf-8", capture_output=capture)
    return result.stdout.strip() if capture else None


def check_source(source):
    head = run(["git", "rev-parse", "HEAD"], source, capture=True)
    if head != SPEC["commit"]:
        raise RuntimeError(f"Expected upstream {SPEC['commit']}, found {head}")
    # Run only the reviewed source; never overwrite or reset a user's checkout.
    if run(["git", "status", "--porcelain", "--untracked-files=no"], source, capture=True):
        raise RuntimeError("Upstream tracked files changed; use a clean dedicated checkout.")


def check_runtime(source):
    node = run(["node", "--version"], source, capture=True).lstrip("v")
    pnpm = run(["pnpm", "--version"], source, capture=True)
    if tuple(map(int, node.split(".")[:3])) < (24, 10, 0):
        raise RuntimeError(f"Node >=24.10.0 required; found {node}")
    if pnpm != SPEC["pnpm"]:
        raise RuntimeError(f"pnpm {SPEC['pnpm']} required; found {pnpm}")


def assets(source):
    root = source / "examples/mars-world/assets"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cache = (root / "cache").resolve()
    for asset in manifest["assets"]:
        for item in asset["files"]:
            target = (cache / item["path"]).resolve()
            if not target.is_relative_to(cache):
                raise RuntimeError("Asset path escapes cache")
            yield item, target


def valid(item, path):
    return (path.is_file() and path.stat().st_size == item["size"]
            and hashlib.md5(path.read_bytes()).hexdigest() == item["md5"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "prepare", "check", "install", "assets", *ACTIONS, *VERIFIERS])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--download-wrapper", type=Path,
                        help="LabDownload.cmd / lab-download, used only for install and asset downloads")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without starting any process")
    parser.add_argument("--output", type=Path, default=HERE / "runs", help="Verification output directory")
    args = parser.parse_args()
    source = args.source.resolve()
    command = ACTIONS.get(args.action)
    if args.action in VERIFIERS:
        mode = VERIFIERS[args.action]
        script = HERE / ("verify-plugins.mjs" if mode == "plugins" else "verify-devices.mjs")
        command = ["node", str(script), str(source), str(args.output.resolve())]
        if mode != "plugins":
            command.append(mode)
    if args.action == "list":
        print(json.dumps({"source": str(source), "upstream": SPEC, "actions": ACTIONS,
                          "verification_modes": VERIFIERS}, indent=2))
        return
    if args.dry_run:
        print(json.dumps({"action": args.action, "cwd": str(source),
                          "command": command, "commit": SPEC["commit"]}, indent=2))
        return
    if args.action == "prepare":
        if source.exists():
            check_source(source)
        else:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.mkdir()
            run(["git", "init"], source)
            run(["git", "remote", "add", "origin", SPEC["repository"]], source)
            run(["git", "fetch", "--depth", "1", "origin", SPEC["commit"]], source)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], source)
            check_source(source)
        print(f"Ready: {source}")
        return
    check_source(source)
    if args.action == "assets":
        for item, target in assets(source):
            if valid(item, target):
                continue
            if not args.download_wrapper or not args.download_wrapper.is_file():
                raise RuntimeError("Asset download requires --download-wrapper")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            run([args.download_wrapper.resolve(), "curl", "--fail", "--location", "--retry", "3",
                 "--output", partial, "--url", item["url"]], source)
            if not valid(item, partial):
                raise RuntimeError(f"Asset size/hash mismatch: {item['path']}")
            partial.replace(target)
        print("All Mars assets match upstream sizes and MD5 hashes.")
        return
    check_runtime(source)
    if args.action == "check":
        print("Pinned source and runtime versions OK.")
        return
    if args.action == "install":
        if not args.download_wrapper or not args.download_wrapper.is_file():
            raise RuntimeError("Dependency install requires --download-wrapper")
        run([args.download_wrapper.resolve(), "--download-exec", "pnpm", "install", "--frozen-lockfile"], source)
        return
    if args.action in {"mars-manual", "mars-world"}:
        if os.name == "nt":
            raise RuntimeError("Pinned upstream Mars scripts use POSIX URL paths. Use WSL/Linux/macOS.")
    if args.action in VERIFIERS and os.name == "nt":
        raise RuntimeError("Headless verification is supported on Linux/macOS; use WSL on Windows.")
    if args.action in {"mars-manual", "mars-world", "verify-mars"}:
        if not str(source).isascii() or " " in str(source):
            raise RuntimeError("Pinned Mars scripts require an ASCII source path without spaces; use --source.")
        if any(not valid(item, target) for item, target in assets(source)):
            raise RuntimeError("Run assets first; runtime must not fetch uncached resources.")
    if args.action == "plugins-smoke" and os.name == "nt":
        raise RuntimeError("This smoke uses Unix sockets and /tmp; use Linux/macOS.")
    env = os.environ.copy()
    # Same dedicated profile in every terminal, outside tutorial sources.
    env["AGH_HOME"] = str(source.parent / "agnes-demo-home")
    env["AGNES_PROFILE"] = "local-dev"
    env["AGNES_HUB_ORIGIN"] = "http://127.0.0.1:4180"
    run(command, source, env=env)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
