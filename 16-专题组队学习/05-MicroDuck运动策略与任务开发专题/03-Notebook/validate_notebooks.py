"""Execute lessons in fresh kernels and retain an auditable validation summary."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import nbformat
from nbclient import NotebookClient


def validate(folder, prefixes=None, kernel="microduck-lightnav", bpu=False):
    folder = Path(folder).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = folder / "outputs" / "validation" / stamp
    output.mkdir(parents=True)
    os.environ["MICRODUCK_RUN_BPU"] = "1" if bpu else "0"
    os.environ["MICRODUCK_COMPILE_BPU"] = "0"
    os.environ["MICRODUCK_ALLOW_CACHED_BPU"] = "0"
    for key in ("MICRODUCK_RUN_BASKETBALL_SMOKE", "MICRODUCK_RUN_FASTSAC_SMOKE",
                "MICRODUCK_RUN_TRAINING", "MICRODUCK_RUN_SMOKE", "MICRODUCK_RUN_LADDER_TRAINING"):
        os.environ[key] = "0"
    results = []
    for path in sorted(folder.glob("[0-9][0-9]_*.ipynb")):
        if prefixes and path.name[:2] not in prefixes:
            continue
        shutil.copy2(path, output / path.name)
        notebook = nbformat.read(path, 4)
        started = time.monotonic()
        result = {"notebook": path.name, "started_utc": datetime.now(timezone.utc).isoformat()}
        print("RUN", path.name, flush=True)
        try:
            NotebookClient(notebook, timeout=1800, kernel_name=kernel,
                           resources={"metadata": {"path": str(folder)}},
                           allow_errors=False).execute()
            code = [c for c in notebook.cells if c.cell_type == "code" and c.source.strip()]
            assert all(c.execution_count is not None for c in code)
            assert all(c.outputs for c in code), "A code cell has no visible output"
            text = "\n".join(o.get("text", "") for c in code for o in c.outputs)
            if path.name[:2] in {"01", "02", "03", "04", "05", "06", "07"}:
                assert "PASS-FRESH-INFERENCE" in text
                assert "PASS-local-onnx" in text
                assert "BPU-CACHED-VIDEO" not in text
                assert ("PASS-FRESH-BPU" if bpu else "NOT-RUN-BPU") in text
                result["bpu"] = "fresh_closed_loop_passed" if bpu else "not_run"
            result.update(status="passed", executed_code_cells=len(code),
                          cells_with_output=sum(bool(c.outputs) for c in code))
            nbformat.write(notebook, path)
            print("PASS", path.name, len(code), "cells", flush=True)
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            nbformat.write(notebook, output / ("failed_" + path.name))
            print("FAIL", path.name, str(exc)[-1500:], flush=True)
        result["wall_seconds"] = round(time.monotonic() - started, 2)
        results.append(result)
        (output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", output / "summary.json", flush=True)
    if any(r["status"] != "passed" for r in results):
        raise SystemExit(1)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", nargs="*")
    parser.add_argument("--kernel", default="microduck-lightnav")
    parser.add_argument("--bpu", action="store_true")
    args = parser.parse_args()
    validate(Path(__file__).resolve().parent, args.prefixes, args.kernel, args.bpu)
