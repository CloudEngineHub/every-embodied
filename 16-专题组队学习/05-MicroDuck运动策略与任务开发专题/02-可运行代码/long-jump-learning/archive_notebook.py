"""Keep executed cells and replace untracked MP4 embeds with tracked GIF previews."""

import argparse
from pathlib import Path

import nbformat


def archive(path):
    notebook = nbformat.read(path, as_version=4)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is not None and cell.outputs, cell.id
        assert not any(output.output_type == "error" for output in cell.outputs), cell.id
        count += 1
        for output in cell.outputs:
            if output.output_type != "display_data":
                continue
            html = output.get("data", {}).get("text/html", "")
            if "<video" not in html:
                continue
            for name in ("pretrained", "continued", "onnx"):
                if f"long_jump_{name}_latest.mp4" in html:
                    image = f"../01-任务资料/08-MicroDuck跨缝跳跃/assets/long_jump_{name}.gif"
                    assert (Path(path).parent / image).is_file(), image
                    output.data = {"text/markdown": f"![跨缝跳跃：{name}]({image})",
                                   "text/plain": f"Long-jump preview: {name}"}
                    break
            else:
                raise ValueError("Unrecognized video output")
    notebook.metadata["teaching"] = {"executed_code_cells": count, "video_previews": "relative-gif",
                                     "full_videos": "outputs/long_jump/*.mp4"}
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Validated {count} executed code cells, all with outputs and no errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("notebook", type=Path)
    archive(parser.parse_args().notebook)
