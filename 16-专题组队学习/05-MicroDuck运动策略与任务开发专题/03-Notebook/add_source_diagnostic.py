"""Add the live source-checkpoint comparison to the FastSAC lesson."""

from pathlib import Path
import nbformat


folder = Path(__file__).resolve().parent
path = next(folder.glob("05_*.ipynb"))
book = nbformat.read(path, 4)
if not book.metadata.get("source_checkpoint_audit"):
    position = next(i for i, cell in enumerate(book.cells)
                    if cell.cell_type == "code" and "GPU_TASK =" in cell.source)
    source = '''SOURCE_AUDIT = OUTPUT_ROOT / "diagnostics" / "motrix_checkpoint.json"
SOURCE_AUDIT.parent.mkdir(parents=True, exist_ok=True)
audit_cmd = [str(MOTRIX_PYTHON), str(MOTRIX_RUNNER), "video",
    str(PLAYGROUND_ROOT / "scripts/audit_motrix_checkpoint.py"),
    "--motrix-root", str(MOTRIX_ROOT), "--onnx", str(MODEL_PATH),
    "--output", str(SOURCE_AUDIT)]
audit_result = subprocess.run(audit_cmd, cwd=MOTRIX_ROOT, capture_output=True, text=True, timeout=180)
if audit_result.returncode:
    raise RuntimeError(audit_result.stderr[-3000:])
source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
print("模型累计训练步数:", source_audit["training_global_step"])
print("优化更新次数:", source_audit["training_update_idx"])
for backend, result in source_audit["rollouts"].items():
    print(backend, "真实控制步数:", result["steps"], "重置次数:", result["resets"],
          "终止原因:", result["termination_counts"], "最大动作误差:", result["max_abs_action_error"])
print("PASS-source-checkpoint: 原始策略与 ONNX 已在真实仿真观测上完成对照。")'''
    book.cells[position:position] = [
        nbformat.v4.new_markdown_cell("## 模型训练进度与终止原因\n\n从 ONNX 元数据找到原始检查点，恢复策略和归一化。在同一场景、同一种子下分别运行原始策略与 ONNX，对照动作误差及重置原因。"),
        nbformat.v4.new_code_cell(source),
    ]
    book.metadata["source_checkpoint_audit"] = 1
    nbformat.validate(book)
    nbformat.write(book, path)
    print(path.name)
