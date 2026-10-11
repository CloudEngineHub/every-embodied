"""Board-independent checks for integer width and recurrent rescaling."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "02-\u53ef\u8fd0\u884c\u4ee3\u7801/microduck-playground-stilts/scripts"


class FakeRuntime:
    def __init__(self, path):
        self.input_names = {"model": ["obs", "h_in", "c_in"]}
        self.output_names = {"model": ["actions", "h_out", "c_out"]}
        self.input_shapes = {"model": {"obs": [1, 61], "h_in": [1, 1, 256], "c_in": [1, 1, 256]}}
        self.output_shapes = {"model": {"actions": [1, 14], "h_out": [1, 1, 256], "c_out": [1, 1, 256]}}
        self.input_quants = {"model": {n: SimpleNamespace(scale=[0.01]) for n in self.input_names["model"]}}
        self.output_quants = {"model": {n: SimpleNamespace(scale=[0.02]) for n in self.output_names["model"]}}
        self.input_dtypes = {"model": {n: "S16" for n in self.input_names["model"]}}
        self.output_dtypes = {"model": {n: "S16" for n in self.output_names["model"]}}

    def run(self, feed):
        self.feed = feed
        return {"model": {"actions": np.full((1, 14), 175, np.int16),
                          "h_out": np.full((1, 1, 256), 300, np.int16),
                          "c_out": np.full((1, 1, 256), -400, np.int16)}}


class QuantizationTests(unittest.TestCase):
    def test_recurrent_output_requantized_to_input_scale(self):
        fake = ModuleType("hbm_runtime")
        fake.HB_HBMRuntime = FakeRuntime
        for filename in ["rdk_bpu_policy_server.py", "rdk_bpu_policy_server_feature_scaled.py"]:
            with self.subTest(filename=filename), patch.dict(sys.modules, {"hbm_runtime": fake}):
                spec = importlib.util.spec_from_file_location("board_test", SCRIPTS / filename)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                policy = module.HbmPolicy("unused.bin")
                action = policy.infer(np.full(61, 5.0, np.float32))
                np.testing.assert_allclose(action, 3.5)
                self.assertEqual(policy.runtime.feed["obs"].dtype, np.int16)
                self.assertEqual(policy.runtime.feed["obs"].max(), 500)
                np.testing.assert_array_equal(policy.h_state, 600)
                np.testing.assert_array_equal(policy.c_state, -800)
                policy.reset()
                self.assertFalse(policy.h_state.any())
                self.assertFalse(policy.c_state.any())

    def test_motrix_client_preserves_scaled_actions(self):
        spec = importlib.util.spec_from_file_location("motrix_client_test", SCRIPTS / "motrix_bpu_inloop_video.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        client = module.BpuClient.__new__(module.BpuClient)
        client.sock = SimpleNamespace(sendall=lambda value: None)
        client.obs_dim, client.action_dim, client.latencies_ms = 54, 14, []
        responses = [module.COUNT.pack(1), np.full(14, 3.5, dtype="<f4").tobytes()]
        with patch.object(module, "recv_exact", side_effect=responses):
            np.testing.assert_array_equal(client.infer(np.zeros(54, np.float32)), 3.5)


if __name__ == "__main__":
    unittest.main()
