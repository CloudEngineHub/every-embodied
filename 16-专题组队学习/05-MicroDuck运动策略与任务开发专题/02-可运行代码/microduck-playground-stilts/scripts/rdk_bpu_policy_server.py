#!/usr/bin/env python3
"""Serve one MicroDuck X5 HBM policy over a small TCP protocol.

The RDK owns policy inference only.  The Ubuntu client sends the already
assembled actor observation as float32 and receives the normalized 14-joint
action as float32.  Quantization, HBM tensor padding, and recurrent state are
kept on the board so the client does not need to know the X5 tensor layout.

Protocol (network byte order for headers, little-endian float32 payloads):

    hello:  !4sHHB3x  (magic, observation_dim, action_dim, recurrent)
    ack:    same layout
    repeat: uint32 count (=1), observation_dim float32, uint32 count (=1),
            action_dim float32

One TCP connection owns one recurrent state.  Closing and reconnecting resets
the LSTM state, which is the desired episode/policy-reset behavior.
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
from pathlib import Path

import numpy as np
from hbm_runtime import HB_HBMRuntime


MAGIC = b"MDP1"
HELLO = struct.Struct("!4sHHB3x")
COUNT = struct.Struct("!I")


def recv_exact(conn: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def first_scale(params: object) -> float:
    scale = np.asarray(getattr(params, "scale"), dtype=np.float32).reshape(-1)
    if scale.size != 1 or not np.isfinite(scale[0]) or scale[0] <= 0:
        raise ValueError(f"Only one positive SCALE quantizer is supported: {scale}")
    return float(scale[0])


def find_name(names: list[str], *candidates: str) -> str:
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for candidate in candidates:
        for name in names:
            if candidate.lower() in name.lower():
                return name
    raise KeyError(f"Cannot find {candidates} in {names}")


class HbmPolicy:
    def __init__(self, model_path: str | Path):
        self.model_path = str(Path(model_path).expanduser())
        self.runtime = HB_HBMRuntime(self.model_path)
        self.model_name = next(iter(self.runtime.input_names))
        self.input_names = list(self.runtime.input_names[self.model_name])
        self.output_names = list(self.runtime.output_names[self.model_name])
        self.obs_name = find_name(self.input_names, "obs", "observation")
        self.action_name = find_name(self.output_names, "actions", "action")
        self.input_shape = list(self.runtime.input_shapes[self.model_name][self.obs_name])
        self.output_shape = list(self.runtime.output_shapes[self.model_name][self.action_name])
        self.obs_dim = int(self.input_shape[-1])
        self.action_dim = int(self.output_shape[-1])
        self.recurrent = "h_in" in self.input_names and "c_in" in self.input_names
        if self.recurrent:
            self.h_in_name = find_name(self.input_names, "h_in")
            self.c_in_name = find_name(self.input_names, "c_in")
            self.h_out_name = find_name(self.output_names, "h_out")
            self.c_out_name = find_name(self.output_names, "c_out")
            self.h_state = np.zeros(
                self.runtime.input_shapes[self.model_name][self.h_in_name], dtype=np.int8
            )
            self.c_state = np.zeros(
                self.runtime.input_shapes[self.model_name][self.c_in_name], dtype=np.int8
            )

        input_quants = self.runtime.input_quants[self.model_name]
        output_quants = self.runtime.output_quants[self.model_name]
        self.input_scales = {
            name: first_scale(input_quants[name]) for name in self.input_names
        }
        self.output_scales = {
            name: first_scale(output_quants[name]) for name in self.output_names
        }

    def reset(self) -> None:
        if self.recurrent:
            self.h_state.fill(0)
            self.c_state.fill(0)

    def quantize(self, value: np.ndarray, name: str) -> np.ndarray:
        scale = self.input_scales[name]
        return np.clip(np.rint(value / scale), -128, 127).astype(np.int8)

    def dequantize(self, value: np.ndarray, name: str) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) * self.output_scales[name]

    def infer(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32).reshape(self.obs_dim)
        feed = {
            self.obs_name: self.quantize(observation, self.obs_name).reshape(self.input_shape)
        }
        if self.recurrent:
            feed[self.h_in_name] = self.h_state
            feed[self.c_in_name] = self.c_state

        outputs = self.runtime.run(feed)[self.model_name]
        action = self.dequantize(outputs[self.action_name], self.action_name)
        if self.recurrent:
            self.h_state = np.asarray(outputs[self.h_out_name], dtype=np.int8).copy()
            self.c_state = np.asarray(outputs[self.c_out_name], dtype=np.int8).copy()
        return np.clip(action.reshape(self.action_dim), -1.0, 1.0).astype(np.float32)

    def describe(self) -> dict[str, object]:
        return {
            "model": self.model_path,
            "model_name": self.model_name,
            "inputs": self.input_names,
            "outputs": self.output_names,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "recurrent": self.recurrent,
            "input_shapes": self.runtime.input_shapes[self.model_name],
            "output_shapes": self.runtime.output_shapes[self.model_name],
            "input_scales": self.input_scales,
            "output_scales": self.output_scales,
        }


def serve_connection(conn: socket.socket, address: tuple[str, int], policy: HbmPolicy) -> None:
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    hello_bytes = recv_exact(conn, HELLO.size)
    if hello_bytes is None:
        return
    magic, obs_dim, action_dim, recurrent = HELLO.unpack(hello_bytes)
    expected_recurrent = int(policy.recurrent)
    if (
        magic != MAGIC
        or obs_dim != policy.obs_dim
        or action_dim != policy.action_dim
        or recurrent != expected_recurrent
    ):
        raise ValueError(
            f"contract mismatch from {address}: got {magic!r}/{obs_dim}/{action_dim}/{recurrent}, "
            f"expected {policy.obs_dim}/{policy.action_dim}/{expected_recurrent}"
        )
    conn.sendall(HELLO.pack(MAGIC, policy.obs_dim, policy.action_dim, expected_recurrent))
    policy.reset()
    count = 0
    total_ms = 0.0
    while True:
        count_bytes = recv_exact(conn, COUNT.size)
        if count_bytes is None:
            break
        (batch_size,) = COUNT.unpack(count_bytes)
        if batch_size != 1:
            raise ValueError(f"Only batch_size=1 is supported, got {batch_size}")
        payload = recv_exact(conn, policy.obs_dim * 4)
        if payload is None:
            break
        observation = np.frombuffer(payload, dtype="<f4", count=policy.obs_dim)
        started = time.perf_counter()
        action = policy.infer(observation)
        total_ms += (time.perf_counter() - started) * 1000.0
        conn.sendall(COUNT.pack(1) + action.astype("<f4", copy=False).tobytes())
        count += 1
        if count == 1 or count % 100 == 0:
            print(
                f"client={address[0]}:{address[1]} frames={count} "
                f"mean_bpu_ms={total_ms / count:.3f}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="X5 HBM file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    policy = HbmPolicy(args.model)
    print(policy.describe(), flush=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(4)
        print(f"RDK X5 BPU policy server listening on {args.host}:{args.port}", flush=True)
        while True:
            conn, address = server.accept()
            print(f"accepted {address}", flush=True)
            try:
                with conn:
                    serve_connection(conn, address, policy)
            except Exception as exc:
                print(f"connection {address} failed: {type(exc).__name__}: {exc}", flush=True)
            finally:
                policy.reset()
                print(f"closed {address}; recurrent state reset", flush=True)


if __name__ == "__main__":
    main()
