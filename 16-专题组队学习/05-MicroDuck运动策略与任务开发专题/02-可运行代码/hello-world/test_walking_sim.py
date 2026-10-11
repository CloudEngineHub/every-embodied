import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from walking_sim import bounded_command, prepare_policy, save_report


class ControlTests(unittest.TestCase):
    def test_command_order_and_limits(self):
        np.testing.assert_allclose(bounded_command(1, -1, 3), [0.3, -0.2, 1.5])
        self.assertEqual(bounded_command().dtype, np.float32)

    def test_invalid_command(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                bounded_command(yaw_rate=bad)

    def test_cached_model_requires_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BEST_alpha_walking.onnx"
            path.write_bytes(b"test")
            with patch("walking_sim.sha256", return_value="matching"), \
                 patch("walking_sim.MODEL_SHA256", "matching"), \
                 patch("walking_sim.urllib.request.build_opener") as opener:
                self.assertEqual(prepare_policy(directory), path)
                opener.assert_not_called()

    def test_yaw_change_unwraps_at_pi(self):
        rows = [{"time_s": i / 50, "phase": "turn", "command": [0, 0, 0.5],
                 "x_m": 0, "y_m": 0, "height_m": 0.12, "upright_cos": 1,
                 "vx_m_s": 0, "yaw_rad": yaw} for i, yaw in enumerate((3.0, -3.0))]
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.onnx"
            policy.write_bytes(b"test")
            report = save_report(Path(directory) / "report.json", rows, policy)
            self.assertGreater(report["phases"]["turn"]["yaw_change_rad"], 0)


if __name__ == "__main__":
    unittest.main()
