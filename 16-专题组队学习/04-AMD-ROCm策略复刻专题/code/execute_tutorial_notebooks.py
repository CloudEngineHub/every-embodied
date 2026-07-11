#!/usr/bin/env python3
"""Execute tutorial notebooks without requiring Jupyter/nbclient.

The AMD learning image contains the project runtime but may not include the
Jupyter execution stack. This small executor runs code cells in one shared
Python namespace and persists stream/error outputs in standard nbformat JSON.
Interactive collection and long training remain controlled by their RUN_*
flags inside the notebooks.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path


MAX_STREAM_CHARS = 100_000
REDACT_ENV_NAMES = [
    "NOTEBOOK_PYTHON",
    "TRAIN_DATA_ROOT",
    "COLLECTION_ROOT",
    "EVAL_DATA_ROOT",
    "OUTPUT_ROOT",
    "MODEL_ROOT",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "NOTEBOOK_TOPIC_ROOT",
]


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def redact_paths(value: str) -> str:
    replacements = []
    for name in REDACT_ENV_NAMES:
        path = os.environ.get(name)
        if path:
            replacements.append((str(Path(path).expanduser()), f"${name}"))
    for path, label in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        value = value.replace(path, label)
    return value


def stream_output(name: str, value: str) -> dict | None:
    if not value:
        return None
    value = redact_paths(value)
    if len(value) > MAX_STREAM_CHARS:
        value = value[:MAX_STREAM_CHARS] + "\n... output truncated by tutorial executor ...\n"
    return {"name": name, "output_type": "stream", "text": value}


def execute_notebook(path: Path, stop_on_error: bool = True) -> tuple[int, int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__notebook__",
        "__file__": str(path),
    }
    execution_count = 0
    errors = 0

    old_cwd = Path.cwd()
    os.chdir(path.parent.parent)
    try:
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            execution_count += 1
            cell["execution_count"] = execution_count
            cell["outputs"] = []
            stdout = io.StringIO()
            stderr = io.StringIO()
            source = cell_source(cell)
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(source, f"{path.name}:cell-{cell_index}", "exec"), namespace)
            except Exception as exc:
                errors += 1
                out = stream_output("stdout", stdout.getvalue())
                err = stream_output("stderr", stderr.getvalue())
                if out:
                    cell["outputs"].append(out)
                if err:
                    cell["outputs"].append(err)
                cell["outputs"].append(
                    {
                        "ename": type(exc).__name__,
                        "evalue": redact_paths(str(exc)),
                        "output_type": "error",
                        "traceback": redact_paths(traceback.format_exc()).splitlines(),
                    }
                )
                if stop_on_error:
                    break
            else:
                out = stream_output("stdout", stdout.getvalue())
                err = stream_output("stderr", stderr.getvalue())
                if out:
                    cell["outputs"].append(out)
                if err:
                    cell["outputs"].append(err)
    finally:
        os.chdir(old_cwd)

    metadata = notebook.setdefault("metadata", {})
    metadata["amd_rocm_tutorial_execution"] = {
        "executor": "code/execute_tutorial_notebooks.py",
        "python": sys.version.split()[0],
        "code_cells_executed": execution_count,
        "errors": errors,
        "long_tasks_enabled": False,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return execution_count, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", type=Path, nargs="+")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to later cells after an exception.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_errors = 0
    for path in args.notebooks:
        cells, errors = execute_notebook(path.resolve(), stop_on_error=not args.keep_going)
        total_errors += errors
        print(f"{path}: executed={cells}, errors={errors}")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
