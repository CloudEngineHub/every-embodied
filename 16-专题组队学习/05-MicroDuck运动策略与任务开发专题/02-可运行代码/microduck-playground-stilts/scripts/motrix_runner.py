#!/usr/bin/env python3
"""Run a MotrixLab script after registering the source-tree simulator backend."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in {"train", "export", "video"}:
        raise SystemExit(
            "usage: motrix_runner.py {train|export|video} /path/to/script.py [args ...]"
        )

    script = Path(sys.argv[2]).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(script)

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

    sys.argv = [str(script), *sys.argv[3:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
