#!/usr/bin/env python3
"""Small LAN control service for switching the pre-registered RDK BPU models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socketserver
import subprocess
import threading
import time
from pathlib import Path


SERVER_SCRIPT = Path("/home/sunrise/microduck_policy/rdk_bpu_policy_server.py")
MODEL_PATHS = {
    "walking": Path("/home/sunrise/microduck_policy/hbm/walking.bin"),
    "perturbation": Path("/home/sunrise/microduck_policy/hbm/walking.bin"),
    "stilt": Path("/home/sunrise/microduck_policy/hbm/stilts25.bin"),
    "swing": Path("/home/sunrise/microduck_policy/hbm/swing.bin"),
    "ladder": Path("/home/sunrise/microduck_policy/hbm/ladder.bin"),
    "basketball": Path("/home/sunrise/microduck_policy/hbm/basketball.bin"),
    "ball_balance": Path("/home/sunrise/microduck_policy/hbm/ball_balance.bin"),
}
PAIR_ROOT = Path("/home/sunrise/microduck_policy/official_ladder_pair")
PAIR_MODELS = {
    "climber": (PAIR_ROOT / "climber.bin", PAIR_ROOT / "official_ladder_climber_feature_scales.json", 8875),
    "getup": (PAIR_ROOT / "getup.bin", PAIR_ROOT / "official_ladder_getup_feature_combined_scales.json", 8876),
}


class PolicySupervisor:
    def __init__(self, port: int, initial_task: str | None):
        self.port = port
        self.lock = threading.Lock()
        self.child: subprocess.Popen[bytes] | None = None
        self.task: str | None = None
        self.pair_children: dict[str, subprocess.Popen] = {}
        if initial_task:
            result = self.switch(initial_task)
            if not result["ok"]:
                raise RuntimeError(result["error"])

    def _stop(self) -> None:
        if self.child is None or self.child.poll() is not None:
            self.child = None
            return
        self.child.terminate()
        try:
            self.child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait(timeout=2)
        self.child = None

    def switch(self, task: str, force: bool = False) -> dict[str, object]:
        if task not in MODEL_PATHS:
            return {"ok": False, "error": f"unknown pre-registered task: {task}"}
        model = MODEL_PATHS[task]
        if not model.is_file():
            return {"ok": False, "error": f"HBM not found: {model}"}
        with self.lock:
            if not force and self.task == task and self.child is not None and self.child.poll() is None:
                return self._model_status(task, model, self.child.pid)
            self._stop()
            log_path = Path(f"/tmp/microduck_{task}_bpu.log")
            log = log_path.open("ab")
            transform = model.with_suffix(".input_transform.json")
            command = ["python3", str(SERVER_SCRIPT), "--model", str(model), "--port", str(self.port)]
            if transform.is_file():
                command[1] = str(SERVER_SCRIPT.with_name("rdk_bpu_policy_server_feature_scaled.py"))
                command.extend(["--input-transform-json", str(transform)])
            self.child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.task = task
            time.sleep(0.5)
            if self.child.poll() is not None:
                return {"ok": False, "error": f"policy server exited; see {log_path}"}
            return self._model_status(task, model, self.child.pid)

    @staticmethod
    def _model_status(task, model, pid):
        transform = model.with_suffix(".input_transform.json")
        return {"ok": True, "task": task, "model": str(model), "pid": pid,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "input_transform_sha256": hashlib.sha256(transform.read_bytes()).hexdigest() if transform.is_file() else None}

    def switch_pair(self):
        with self.lock:
            result = {"ok": True, "models": {}}
            for role, (model, transform, port) in PAIR_MODELS.items():
                if not model.is_file() or not transform.is_file():
                    return {"ok": False, "error": f"Missing pair model/transform: {role}"}
                child = self.pair_children.get(role)
                if child is None or child.poll() is not None:
                    log = Path(f"/tmp/microduck_pair_{role}.log").open("ab")
                    child = subprocess.Popen([
                        "python3", str(SERVER_SCRIPT.with_name("rdk_bpu_policy_server_feature_scaled.py")),
                        "--model", str(model), "--input-transform-json", str(transform), "--port", str(port),
                    ], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    self.pair_children[role] = child
                    time.sleep(0.5)
                if child.poll() is not None:
                    return {"ok": False, "error": f"Pair {role} server exited"}
                info = self._model_status(role, model, child.pid)
                info.update(port=port, input_transform=str(transform),
                            transform_sha256=hashlib.sha256(transform.read_bytes()).hexdigest())
                result["models"][role] = info
            return result

    def status(self) -> dict[str, object]:
        with self.lock:
            return {
                "ok": self.child is not None and self.child.poll() is None,
                "task": self.task,
                "pid": self.child.pid if self.child is not None else None,
                "port": self.port,
                "sha256": hashlib.sha256(MODEL_PATHS[self.task].read_bytes()).hexdigest() if self.task else None,
            }

    def upload_and_switch(self, task: str, payload: bytes, expected_sha256: str, input_transform=None) -> dict[str, object]:
        if task not in MODEL_PATHS:
            return {"ok": False, "error": f"unknown pre-registered task: {task}"}
        model = MODEL_PATHS[task]
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            return {"ok": False, "error": f"sha256 mismatch: {actual_sha256}"}
        if input_transform is not None:
            dim = 54 if task == "ball_balance" else 61
            if not isinstance(input_transform, dict):
                return {"ok": False, "error": "input transform must contain offset and scale"}
            offset, scale = input_transform.get("offset", []), input_transform.get("scale", [])
            if len(offset) != dim or len(scale) != dim or not all(math.isfinite(v) for v in offset + scale) or not all(v > 0 for v in scale):
                return {"ok": False, "error": "invalid input transform"}
            input_transform = {**input_transform, "model_sha256": actual_sha256}
        temporary = model.with_name(model.name + ".uploading")
        temporary.write_bytes(payload)
        temporary.replace(model)
        transform = model.with_suffix(".input_transform.json")
        if input_transform is None:
            transform.unlink(missing_ok=True)
        else:
            transform.write_text(json.dumps(input_transform, sort_keys=True), encoding="utf-8")
        return self.switch(task, force=True)


class RequestHandler(socketserver.StreamRequestHandler):
    supervisor: PolicySupervisor

    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline(8192).decode("utf-8"))
            if request.get("op") == "switch":
                response = self.supervisor.switch(str(request.get("task", "")))
            elif request.get("op") == "switch_pair":
                response = self.supervisor.switch_pair()
            elif request.get("op") == "upload_and_switch":
                size = int(request.get("size", -1))
                if size < 0 or size > 16 * 1024 * 1024:
                    response = {"ok": False, "error": "payload size must be between 0 and 16 MiB"}
                else:
                    payload = self.rfile.read(size)
                    if len(payload) != size:
                        response = {"ok": False, "error": f"short payload: {len(payload)} / {size} bytes"}
                    else:
                        response = self.supervisor.upload_and_switch(
                            str(request.get("task", "")), payload, str(request.get("sha256", "")), request.get("input_transform")
                        )
            elif request.get("op") == "status":
                response = self.supervisor.status()
            else:
                response = {"ok": False, "error": "supported operations: switch, status"}
        except Exception as exc:  # Keep the control port alive for later cells.
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))


class ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--policy-port", type=int, default=8765)
    parser.add_argument("--initial-task", choices=sorted(MODEL_PATHS))
    args = parser.parse_args()
    supervisor = PolicySupervisor(args.policy_port, args.initial_task)
    RequestHandler.supervisor = supervisor
    with ThreadingServer(("0.0.0.0", args.port), RequestHandler) as server:
        print(f"RDK BPU policy supervisor listening on 0.0.0.0:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
