import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from notebook_runtime import resolve_model, sha256, verify_gpu_report


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "actor.onnx"
        self.model.write_bytes(b"actor")
        self.video = self.root / "video.mp4"
        self.video.write_bytes(b"video")
        self.report = {"run_id": "new", "policy_device": "GPU", "steps": 10,
                       "onnx_sha256": sha256(self.model), "policy_validation": {
                           "inference_calls": 10, "real_observation_parity_checks": 10,
                           "max_abs_output_error": 1e-6}}

    def check_report(self):
        self.video.with_suffix(".json").write_text(json.dumps(self.report))
        return verify_gpu_report(self.video, "new", [self.model])

    def test_fresh_report(self):
        self.assertTrue(self.check_report()["fresh_inference_verified"])

    def test_stale_video_rejected(self):
        self.report["run_id"] = "old"
        with self.assertRaises(ValueError):
            self.check_report()

    def test_wrong_model_rejected(self):
        self.report["onnx_sha256"] = "other"
        with self.assertRaises(ValueError):
            self.check_report()

    def test_wrong_call_count_rejected(self):
        self.report["steps"] = 11
        with self.assertRaises(ValueError):
            self.check_report()

    def test_early_termination_rejected(self):
        self.report["requested_steps"] = 20
        with self.assertRaises(ValueError):
            self.check_report()

    def test_missing_live_comparison_rejected(self):
        self.report["policy_validation"]["real_observation_parity_checks"] = 0
        with self.assertRaises(ValueError):
            self.check_report()

    def test_perturbation_selects_walking(self):
        converted = self.root / "converted"
        converted.mkdir()
        walking = converted / "walking.opset11.onnx"
        walking.write_bytes(b"walking")
        with patch.dict(os.environ, {"MICRODUCK_MODEL_ROOT": str(self.root)}, clear=True):
            self.assertEqual(resolve_model("perturbation", self.root), walking.resolve())

    def test_missing_explicit_model_does_not_fallback(self):
        with patch.dict(os.environ, {"MICRODUCK_ONNX": str(self.root / "missing")}, clear=True):
            with self.assertRaises(FileNotFoundError):
                resolve_model("walking", self.root, self.model)

    def test_conflicting_models_rejected(self):
        with patch.dict(os.environ, {"MICRODUCK_ONNX": str(self.model),
                                    "MICRODUCK_GPU_ONNX": str(self.root / "other")}, clear=True):
            with self.assertRaises(ValueError):
                resolve_model("walking", self.root)


if __name__ == "__main__":
    unittest.main()
