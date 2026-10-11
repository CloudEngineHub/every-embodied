"""CPU-only tests; never download or execute a training checkpoint."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jump_lesson as lesson


class LessonTests(unittest.TestCase):
    def test_failed_candidate_keeps_working_model(self):
        good = {"landed_on_target": True, "final_target_support": True,
                "video_seconds": 12, "termination": ["time_out"]}
        bad = {**good, "video_seconds": 0.7, "termination": ["fell"]}
        choice = lesson.select_deployment_model(good, bad, Path("baseline"), Path("candidate"))
        self.assertFalse(choice["candidate_accepted"])
        self.assertEqual(choice["selected"], "pretrained")

    def test_successful_candidate_can_be_selected(self):
        good = {"landed_on_target": True, "final_target_support": True,
                "video_seconds": 12, "termination": ["time_out"]}
        self.assertTrue(lesson.select_deployment_model(good, good, "base", "new")["candidate_accepted"])

    def test_failed_baseline_blocks_deployment(self):
        bad = {"landed_on_target": False, "final_target_support": False,
               "video_seconds": 0.5, "termination": ["fell"]}
        with self.assertRaises(RuntimeError):
            lesson.select_deployment_model(bad, bad, "base", "new")

    def test_passing_candidate_can_replace_failed_baseline(self):
        good = {"landed_on_target": True, "final_target_support": True,
                "video_seconds": 12, "termination": ["time_out"]}
        bad = {**good, "video_seconds": 0.7, "termination": ["fell"]}
        self.assertTrue(lesson.select_deployment_model(bad, good, "base", "new")["candidate_accepted"])

    def test_removes_inherited_task_overrides(self):
        with patch.dict(os.environ, {"MICRODUCK_PJ_GAP": "9", "MICRODUCK_TOPIC_ROOT": "/tmp/other"}):
            env = lesson.command_env()
            self.assertFalse(any(key.startswith("MICRODUCK_") for key in env))
            env = lesson.command_env(profile=True)
            self.assertEqual(env["MICRODUCK_PJ_GAP"], "0.30")
            self.assertEqual(env["MICRODUCK_PJ_DROP"], "0.25")

    def test_source_import_order(self):
        env = lesson.command_env(Path("/tmp/source"))
        self.assertTrue(env["PYTHONPATH"].startswith(os.pathsep.join(
            (str(Path("/tmp/source/src")), str(Path("/tmp/source/scripts"))))))

    def test_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "sample"
            file.write_bytes(b"microduck")
            self.assertEqual(lesson.sha256(file), hashlib.sha256(b"microduck").hexdigest())

    def test_matching_download_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "policy.onnx"
            file.write_bytes(b"model")
            with patch.dict(lesson.HASHES, {"policy.onnx": lesson.sha256(file)}), \
                 patch("urllib.request.build_opener") as network:
                lesson.download_file("policy.onnx", file)
                network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
