"""Board transport and fresh-evidence regression checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      "02-可运行代码/microduck-playground-stilts/scripts"))
from bpu_inloop_video import BpuClient, HELLO, MAGIC, COUNT
from notebook_runtime import sha256, verify_bpu_report


class Socket:
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def recv(self, size):
        data, self.payload = self.payload[:size], self.payload[size:]
        return data

    def sendall(self, data):
        pass

    def settimeout(self, value):
        pass

    def shutdown(self, mode):
        pass

    def close(self):
        self.closed = True


class ContractTests(unittest.TestCase):
    def test_recurrent_reset_reconnects_and_preserves_call_count(self):
        ack = HELLO.pack(MAGIC, 61, 14, 1)
        first = Socket(ack + COUNT.pack(1) + np.zeros(14, dtype="<f4").tobytes())
        second = Socket(ack)
        with patch("bpu_inloop_video.socket.create_connection", side_effect=[first, second]):
            client = BpuClient("board", 8765, 61, recurrent=True)
            client.infer(np.zeros(61))
            client.reset()
            self.assertTrue(first.closed)
            self.assertEqual(client.episode_resets, 1)
            self.assertEqual(len(client.latencies_ms), 1)

    def test_recurrent_contract_rejected(self):
        with patch("bpu_inloop_video.socket.create_connection", return_value=Socket(HELLO.pack(MAGIC, 61, 14, 0))):
            with self.assertRaises(RuntimeError):
                BpuClient("board", 8765, 61, recurrent=True)

    def test_nonfinite_action_rejected(self):
        payload = HELLO.pack(MAGIC, 61, 14, 0) + COUNT.pack(1) + np.full(14, np.nan, dtype="<f4").tobytes()
        with patch("bpu_inloop_video.socket.create_connection", return_value=Socket(payload)):
            client = BpuClient("board", 8765, 61)
            with self.assertRaises(RuntimeError):
                client.infer(np.zeros(61))

    def test_evidence_rejects_wrong_board_model_and_old_video(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video, model = root / "task.mp4", root / "model.bin"
            video.write_bytes(b"test")
            model.write_bytes(b"model")
            digest = sha256(model)
            video.with_suffix(".json").write_text(json.dumps({"run_id": "new", "steps": 2,
                "requested_steps": 2, "backend": "RDK X5 BPU via hbm_runtime", "hbm_sha256": digest}))
            with self.assertRaises(ValueError):
                verify_bpu_report(video, "old", [model], [digest])
            with self.assertRaises(ValueError):
                verify_bpu_report(video, "new", [model], ["wrong"])
            self.assertTrue(verify_bpu_report(video, "new", [model], [digest])["fresh_inference_verified"])


if __name__ == "__main__":
    unittest.main()
