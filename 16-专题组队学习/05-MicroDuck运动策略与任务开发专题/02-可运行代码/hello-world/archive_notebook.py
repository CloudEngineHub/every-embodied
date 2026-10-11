"""Archive executed HelloWorld outputs with a portable GIF, not a private MP4 path."""

import json
from pathlib import Path

import nbformat


def archive(topic):
    topic = Path(topic)
    path = topic / "03-Notebook/00_MuJoCo策略控制_HelloWorld.ipynb"
    notebook = nbformat.read(path, 4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(cells) == 12
    assert all(cell.execution_count is not None and cell.outputs for cell in cells)
    assert not any(out.output_type == "error" for cell in cells for out in cell.outputs)
    image = "../01-任务资料/07-RDK端侧与网页部署/assets/hello_world/forward_turn_stop.gif"
    assert (path.parent / image).is_file()
    for cell in cells:
        for output in cell.outputs:
            if output.output_type == "display_data" and "<video" in output.get("data", {}).get("text/html", ""):
                output.data = {"text/markdown": f"![前进、左转与停止]({image})",
                               "text/plain": "MuJoCo live policy control: forward, turn, stop"}
    notebook.metadata.teaching.update(executed_code_cells=12, video_preview="relative-gif")
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    report_path = topic / "03-Notebook/outputs/control_helloworld/control_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("trace")
    report.pop("final_state")
    report["checks"] = {
        "forward_displacement_gt_0p3_m": report["phases"]["forward"]["dx_m"] > 0.3,
        "left_turn_gt_0p3_rad": report["phases"]["turn_left"]["yaw_change_rad"] > 0.3,
        "stopped_speed_lt_0p08_m_s": report["phases"]["stop"]["last_second_mean_abs_vx_m_s"] < 0.08,
        "stayed_upright": report["min_upright_cos"] >= 0.7 and report["min_height_m"] >= 0.065,
    }
    assert all(report["checks"].values())
    output = topic / "01-任务资料/07-RDK端侧与网页部署/assets/hello_world/validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Archived 12 executed cells and the physical-control validation report.")


if __name__ == "__main__":
    archive(Path(__file__).resolve().parents[2])
