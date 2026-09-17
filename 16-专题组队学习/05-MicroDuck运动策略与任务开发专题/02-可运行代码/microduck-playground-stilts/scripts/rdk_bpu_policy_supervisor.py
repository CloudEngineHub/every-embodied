#!/usr/bin/env python3
"""Small LAN control service for switching the pre-registered RDK BPU models."""

from __future__ import annotations

import argparse
import hashlib
import json
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


class PolicySupervisor:
    def __init__(self, port: int, initial_task: str | None):
        self.port = port
        self.lock = threading.Lock()
        self.child: subprocess.Popen[bytes] | None = None
        self.task: str | None = None
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
                return {"ok": True, "task": task, "model": str(model), "pid": self.child.pid}
            self._stop()
            log_path = Path(f"/tmp/microduck_{task}_bpu.log")
            log = log_path.open("ab")
            self.child = subprocess.Popen(
                ["python3", str(SERVER_SCRIPT), "--model", str(model), "--port", str(self.port)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.task = task
            time.sleep(0.5)
            if self.child.poll() is not None:
                return {"ok": False, "error": f"policy server exited; see {log_path}"}
            return {"ok": True, "task": task, "model": str(model), "pid": self.child.pid}

    def status(self) -> dict[str, object]:
        with self.lock:
            return {
                "ok": self.child is not None and self.child.poll() is None,
                "task": self.task,
                "pid": self.child.pid if self.child is not None else None,
                "port": self.port,
            }

    def upload_and_switch(self, task: str, payload: bytes, expected_sha256: str) -> dict[str, object]:
        if task not in MODEL_PATHS:
            return {"ok": False, "error": f"unknown pre-registered task: {task}"}
        model = MODEL_PATHS[task]
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            return {"ok": False, "error": f"sha256 mismatch: {actual_sha256}"}
        temporary = model.with_name(model.name + ".uploading")
        temporary.write_bytes(payload)
        temporary.replace(model)
        return self.switch(task, force=True)


class RequestHandler(socketserver.StreamRequestHandler):
    supervisor: PolicySupervisor

    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline(4096).decode("utf-8"))
            if request.get("op") == "switch":
                response = self.supervisor.switch(str(request.get("task", "")))
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
                            str(request.get("task", "")), payload, str(request.get("sha256", ""))
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
