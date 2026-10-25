import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = next((Path(__file__).resolve().parents[1] / "02-可运行代码").glob("*/scripts/motrix_runner.py"))
SPEC = importlib.util.spec_from_file_location("motrix_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class MotrixEntrypointTests(unittest.TestCase):
    def test_current_commands(self):
        for mode, command in runner.COMMANDS.items():
            with self.subTest(mode=mode), patch.object(runner, "has_unified_cli", return_value=True):
                self.assertEqual(
                    runner.resolve_entrypoint(mode, ["--help"]),
                    ("module", "motrixlab.cli", [command, "--help"]),
                )

    def test_old_notebook_arguments_on_current_release(self):
        with patch.object(runner, "has_unified_cli", return_value=True):
            self.assertEqual(
                runner.resolve_entrypoint("train", ["/removed/scripts/train.py", "task=microduck-ball-balance/motrix.fastsac"]),
                ("module", "motrixlab.cli", ["train", "task=microduck-ball-balance/motrix.fastsac"]),
            )
            self.assertEqual(
                runner.resolve_entrypoint("export", ["/removed/scripts/export_onnx.py", "opset=11"]),
                ("module", "motrixlab.cli", ["export-onnx", "opset=11"]),
            )

    def test_legacy_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            for mode, name in runner.LEGACY_SCRIPTS.items():
                script = root / "scripts" / name
                script.touch()
                with self.subTest(mode=mode), patch.object(runner, "has_unified_cli", return_value=False), patch.object(runner.Path, "cwd", return_value=root):
                    self.assertEqual(runner.resolve_entrypoint(mode, ["--cfg", "job"]), ("script", str(script), ["--cfg", "job"]))

    def test_explicit_legacy_script(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "train.py"
            script.touch()
            with patch.object(runner, "has_unified_cli", return_value=False):
                self.assertEqual(runner.resolve_entrypoint("train", [str(script), "play=false"]), ("script", str(script.resolve()), ["play=false"]))

    def test_video_stays_a_script(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "video.py"
            script.touch()
            with patch.object(runner, "has_unified_cli", return_value=True):
                self.assertEqual(runner.resolve_entrypoint("video", [str(script), "--steps", "10"]), ("script", str(script.resolve()), ["--steps", "10"]))

    def test_missing_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "has_unified_cli", return_value=False), patch.object(runner.Path, "cwd", return_value=Path(directory)):
            with self.assertRaisesRegex(FileNotFoundError, "repository root"):
                runner.resolve_entrypoint("train", [])
            with self.assertRaises(FileNotFoundError):
                runner.resolve_entrypoint("video", [])

    def test_detection_without_installed_package(self):
        with patch.object(runner, "find_spec", side_effect=ModuleNotFoundError):
            self.assertFalse(runner.has_unified_cli())

    def test_module_dispatch_does_not_bootstrap_legacy_plugins(self):
        with patch.object(runner.sys, "argv", ["runner.py", "train", "--help"]), patch.object(runner, "has_unified_cli", return_value=True), patch.object(runner, "prepare_legacy_runtime") as prepare, patch.object(runner.runpy, "run_module") as dispatch:
            runner.main()
            self.assertEqual(runner.sys.argv, ["motrixlab.cli", "train", "--help"])
            dispatch.assert_called_once_with("motrixlab.cli", run_name="__main__")
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
