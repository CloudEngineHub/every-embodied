import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat

from archive_inference_outputs import archive, media_specs, publish_recordings


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.topic = Path(self.temp.name)
        self.folder = self.topic / "03-Notebook"
        self.destination = self.folder / "outputs/published_notebooks"
        self.destination.mkdir(parents=True)
        self.nav = self.topic / "nav"
        self.nav.mkdir()
        (self.nav / "lesson_summary.json").write_text("{}", encoding="utf-8")
        self.books = {}
        for prefix, backend, group, name, preview in media_specs():
            source = self.folder / "outputs" / group / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(name.encode("ascii"))
            report = source.parent / "control_report.json" if group == "control_helloworld" else source.with_suffix(".json")
            if group != "lightnav_gpu":
                report.write_text(json.dumps({"run_id": name}), encoding="utf-8")
            if prefix not in self.books:
                self.books[prefix] = nbformat.v4.new_notebook()
            cell = nbformat.v4.new_code_cell(source="print('actual result')", execution_count=1, outputs=[
                nbformat.v4.new_output("stream", name="stdout", text="actual result\n"),
                nbformat.v4.new_output("display_data", data={"text/markdown": f"![preview](../assets/{preview})"}),
            ])
            self.books[prefix].cells.append(cell)
        for prefix, book in self.books.items():
            nbformat.write(book, self.destination / f"{prefix}_fixture.ipynb")

    def publish(self):
        with patch.dict("os.environ", {"LIGHTNAV_OUTPUT": str(self.nav)}):
            publish_recordings(self.folder, self.destination)

    def test_all_videos_are_independent_and_hash_verified(self):
        self.publish()
        manifest = json.loads((self.destination / "recordings/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["video_count"], 21)
        for entry in manifest["videos"]:
            data = (self.destination / entry["video"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
            self.assertEqual(len(data), entry["bytes"])
            self.assertTrue((self.destination / entry["report"]).is_file())
        self.assertEqual(manifest["total_video_bytes"], sum(e["bytes"] for e in manifest["videos"]))

    def test_execution_outputs_and_local_players_survive(self):
        self.publish()
        for path in self.destination.glob("*.ipynb"):
            book = nbformat.read(path, 4)
            nbformat.validate(book)
            for cell in book.cells:
                self.assertEqual(cell.execution_count, 1)
                self.assertEqual(cell.outputs[0].text, "actual result\n")
                self.assertIn(".mp4)", cell.outputs[1].data["text/markdown"])
                html = cell.outputs[2].data["text/html"]
                self.assertIn("<video controls", html)
                self.assertNotIn("base64", html)

    def test_missing_video_is_not_silently_skipped(self):
        prefix, backend, group, name, preview = next(media_specs())
        (self.folder / "outputs" / group / name).unlink()
        with self.assertRaises(FileNotFoundError):
            self.publish()

    def test_repeated_preview_reuses_the_same_recording(self):
        path = self.destination / "09_fixture.ipynb"
        book = nbformat.read(path, 4)
        book.cells.append(nbformat.v4.new_code_cell(source="print('preview')", execution_count=2, outputs=[
            nbformat.v4.new_output("display_data", data={"text/markdown": "![preview](../assets/long_jump_pretrained.gif)"}),
        ]))
        nbformat.write(book, path)
        self.publish()
        manifest = json.loads((self.destination / "recordings/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["video_count"], 21)

    def test_live_notebooks_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            archive(self.topic, self.folder)


if __name__ == "__main__":
    unittest.main()
