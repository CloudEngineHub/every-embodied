#!/usr/bin/env python3
"""Launch the installed MotrixLab CLI, legacy scripts, or tutorial videos."""

from __future__ import annotations

import runpy
import sys
from importlib.util import find_spec
from pathlib import Path


COMMANDS = {"train": "train", "export": "export-onnx", "play": "play", "view": "view"}
LEGACY_SCRIPTS = {
    "train": "train.py", "export": "export_onnx.py", "play": "play.py", "view": "view.py",
}


def has_unified_cli() -> bool:
    try:
        return find_spec("motrixlab.cli") is not None
    except ModuleNotFoundError:
        return False


def resolve_entrypoint(mode: str, args: list[str]) -> tuple[str, str, list[str]]:
    if mode not in {*COMMANDS, "video"}:
        raise ValueError(f"Unknown MotrixLab command: {mode}")
    args = list(args)
    # Accept older notebooks that still pass scripts/train.py explicitly.
    script = Path(args.pop(0)).expanduser().resolve() if args and args[0].endswith(".py") else None
    if mode != "video" and has_unified_cli():
        return "module", "motrixlab.cli", [COMMANDS[mode], *args]
    if script is None and mode != "video":
        script = Path.cwd() / "scripts" / LEGACY_SCRIPTS[mode]
    if script is None or not script.is_file():
        raise FileNotFoundError(
            f"No MotrixLab entry point for {mode}: {script}. "
            "Run from the MotrixLab repository root using its installed .venv Python; "
            "new releases provide python -m motrixlab.cli, older releases use scripts/."
        )
    return "script", str(script), args


def prepare_legacy_runtime() -> None:
    # Source checkouts do not have the package entry point that normally
    # registers MotrixSim, so do it explicitly before the target script loads.
    from motrix_env_core.sim.registry import list_sim_backends

    if not list_sim_backends():
        from motrix_env_motrixsim.register import register

        register()

    try:
        import torch

        mark_step_begin = getattr(
            getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None
        )
        if mark_step_begin is not None:
            torch.compiler.cudagraph_mark_step_begin = lambda: None
    except ImportError:
        pass


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: motrix_runner.py {train|export|play|view|video} [script.py] [args ...]")
    kind, target, args = resolve_entrypoint(sys.argv[1], sys.argv[2:])
    sys.argv = [target, *args]
    if kind == "module":
        # The installed CLI owns plugin discovery and config search paths.
        runpy.run_module(target, run_name="__main__")
    else:
        prepare_legacy_runtime()
        runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
