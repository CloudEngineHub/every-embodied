"""Unify task model selection without recreating entire lesson notebooks."""

from pathlib import Path
import textwrap

import nbformat


TASKS = {"01": "basketball", "02": "perturbation", "03": "stilt", "04": "swing",
         "05": "ball_balance", "06": "ladder", "07": "walking"}


def patch(folder):
    for path in sorted(Path(folder).glob("0[1-7]_*.ipynb")):
        notebook = nbformat.read(path, 4)
        if notebook.metadata.get("inference_audit_version") == 1:
            continue
        task = TASKS[path.name[:2]]
        for cell in notebook.cells:
            if cell.cell_type == "code":
                source = cell.source
                if source.startswith("TASK_ID ="):
                    first = "\n".join(source.splitlines()[:2])
                    source = first + f"""
import sys
sys.path.insert(0, str(TOPIC_ROOT / "03-Notebook"))
from notebook_runtime import resolve_model, policy_python, inspect_policy, local_model_check, verify_gpu_report
TASK_KEY = {task!r}
MODEL_PATH = resolve_model(TASK_KEY, TOPIC_ROOT, MODEL_HINT)
POLICY_PYTHON = policy_python(PLAYGROUND_ROOT)
print("任务:", TASK_KEY)
print("实际推理模型:", MODEL_PATH)
print("运动任务环境已就绪；Jupyter 与运动任务可以使用不同 Python 环境。")
"""
                elif source.startswith("def inspect_onnx"):
                    source = """def inspect_onnx(path):
    result = inspect_policy(path, POLICY_PYTHON)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

contract = inspect_onnx(MODEL_PATH)
print("PASS-model-contract: 已检查当前任务实际使用的模型。")"""
                elif source.startswith("def onnx_local_smoke"):
                    source = """local_check = local_model_check(MODEL_PATH, POLICY_PYTHON)
print(json.dumps(local_check, ensure_ascii=False, indent=2))
print("PASS-local-onnx: 输入输出检查完成，下一单元运行任务闭环。")"""
                elif source.startswith("LADDER_ARTIFACT_ROOT ="):
                    source = """CLIMBER_ONNX = MODEL_PATH
GETUP_ONNX = Path(os.getenv("MICRODUCK_GPU_LADDER_GETUP",
    str(MODEL_PATH.parent / "official_ladder_getup.opset11.onnx"))).expanduser()
if not GETUP_ONNX.is_file():
    raise FileNotFoundError(GETUP_ONNX)
for ladder_model in (CLIMBER_ONNX, GETUP_ONNX):
    print("检查爬梯模型:", ladder_model.name)
    inspect_onnx(ladder_model)
print("PASS-ladder-models: 闭环使用这两个 climber/get-up 模型。")"""
                elif "GPU_TASK =" in source:
                    lines = source.splitlines()
                    lines = [
                        "GPU_ONNX = MODEL_PATH" if line.startswith("GPU_ONNX =") else
                        "GPU_LADDER_CLIMBER = CLIMBER_ONNX" if task == "ladder" and line.startswith("GPU_LADDER_CLIMBER =") else
                        "GPU_LADDER_GETUP = GETUP_ONNX" if task == "ladder" and line.startswith("GPU_LADDER_GETUP =") else line
                        for line in lines
                    ]
                    source = "\n".join(lines)
                    source = source.replace("sys.executable, str(GPU_", "str(POLICY_PYTHON), str(GPU_")
                    source = source.replace('print("GPU 回放暂不可运行：请检查任务专属 ONNX、脚本和 Motrix 环境。")',
                                            'raise FileNotFoundError("缺少任务模型、脚本或仿真环境；请按任务教程准备。")')
                    source = source.replace('print("找不到 MotrixLab Python 环境:", MOTRIX_PYTHON)',
                                            'raise FileNotFoundError(MOTRIX_PYTHON)')
                    source = source.replace('    gpu_result = subprocess.run(gpu_cmd, cwd=PLAYGROUND_ROOT, env=os.environ.copy(), capture_output=True, text=True)',
                        '    import uuid\n    GPU_RUN_ID = str(uuid.uuid4())\n    gpu_env = os.environ.copy()\n    gpu_env["MICRODUCK_VALIDATION_RUN_ID"] = GPU_RUN_ID\n'
                        '    gpu_result = subprocess.run(gpu_cmd, cwd=PLAYGROUND_ROOT, env=gpu_env, capture_output=True, text=True, timeout=600)')
                    source = source.replace('        raise RuntimeError("GPU 推理视频生成失败，请查看脚本日志。")',
                        '        log_path = OUTPUT_ROOT / "gpu_videos" / f"{GPU_TASK}_error.log"\n'
                        '        log_path.write_text(gpu_result.stdout + "\\n" + gpu_result.stderr, encoding="utf-8")\n'
                        '        raise RuntimeError(f"GPU 推理失败；完整日志: {log_path}")')
                    source = source.replace('    if GPU_REPORT.is_file():\n        print(json.loads(GPU_REPORT.read_text(encoding="utf-8")))',
                        '    selected_models = [GPU_LADDER_CLIMBER, GPU_LADDER_GETUP] if GPU_TASK == "ladder" else [GPU_ONNX]\n'
                        '    gpu_report = verify_gpu_report(GPU_VIDEO_OUTPUT, GPU_RUN_ID, selected_models)\n'
                        '    print(gpu_report)\n    print("PASS-FRESH-INFERENCE: 本次模型调用、关节动作与新视频已核对。")')
                elif source.startswith("NAV_ONNX ="):
                    source = '''NAV_ONNX = os.getenv("MICRODUCK_NAV_ONNX", "")
if NAV_ONNX:
    MODEL_PATH = Path(NAV_ONNX).expanduser().resolve()
    inspect_onnx(MODEL_PATH)
print("本课行走控制模型:", MODEL_PATH)
print("视觉语言导航请运行第 08 课；本课将行走动作接入仿真与板端接口。")'''
                elif source.startswith("import shutil") and "BPU_STATIC_ONNX =" in source:
                    source = source.replace('BPU_MODEL_ROOT / "compiled" / "ball_balance"',
                                            'OUTPUT_ROOT / "bpu_candidates" / "ball_balance"')
                    source = source.replace('  working_dir: /work/microduck_bpu_models/compiled/ball_balance/model_output',
                                            '  working_dir: {workspace_container_path(BPU_OUTPUT.parent)}')
                    source = source.replace('        """calibration_parameters:', '        f"""calibration_parameters:')
                    source = source.replace('  set_node_data_type: {}', '  set_node_data_type: {{}}')
                    source = """COMPILE_BPU = os.getenv("MICRODUCK_COMPILE_BPU", "0") == "1"
if not COMPILE_BPU:
    print("NOT-RUN-COMPILE: HBM 编译未启用；本轮验证现有模型的推理，不改写板端模型。")
else:
""" + textwrap.indent(source, "    ")
                elif "BPU_TASK =" in source or (source.startswith("import time") and "BPU_SMOKE_MODEL" in source):
                    source = source.replace('ALLOW_CACHED_BPU = os.getenv("MICRODUCK_ALLOW_CACHED_BPU", "0") == "1"',
                                            'ALLOW_CACHED_BPU = False')
                    source = source.replace('print("官方爬梯 BPU 双模型视频暂不可运行：请检查 climber/get-up HBM 和脚本。")',
                                            'raise FileNotFoundError("缺少 climber/get-up HBM 或脚本。")')
                    source = source.replace('print("Ubuntu 端 HBM 不存在:", BPU_HBM)', 'raise FileNotFoundError(BPU_HBM)')
                    source = source.replace('print("找不到板端闭环录制器:", BPU_VIDEO_SCRIPT)', 'raise FileNotFoundError(BPU_VIDEO_SCRIPT)')
                    if "sys.executable" in source:
                        source = source.replace("sys.executable", "str(POLICY_PYTHON)")
                    if "else:\n    show_cached_bpu_video(BPU_VIDEO_OUTPUT," in source:
                        source = source.replace('else:\n    show_cached_bpu_video(BPU_VIDEO_OUTPUT, "本轮板端不可达，保留此前真实 RDK X5 BPU 任务视频。")',
                            'else:\n    if not show_cached_bpu_video(BPU_VIDEO_OUTPUT, "板端不可达，展示历史视频。"):\n'
                            '        raise ConnectionError("RDK 板端推理未完成；请检查开发板网络与策略服务。")')
                    source = """RUN_BPU = os.getenv("MICRODUCK_RUN_BPU", "0") == "1"
if not RUN_BPU:
    print("NOT-RUN-BPU: 板端推理未启用；设置 MICRODUCK_RUN_BPU=1 后连接 RDK 实际运行。")
else:
""" + textwrap.indent(source, "    ")
                if source != cell.source:
                    cell.source = source
                if "RUN_SMOKE" in source or "RUN_FASTSAC_SMOKE" in source or "RUN_TRAIN" in source:
                    cell.source = cell.source.replace('sys.executable', 'str(POLICY_PYTHON)')
                    cell.source = cell.source.replace('os.getenv("MICRODUCK_RUN_FASTSAC_SMOKE", "1")',
                                                     'os.getenv("MICRODUCK_RUN_FASTSAC_SMOKE", "0")')
            elif cell.cell_type == "markdown":
                if "本任务策略闭环与 GPU 视频" in cell.source:
                    cell.source += "\n\n本次模型检查与闭环使用同一个 `MODEL_PATH`。报告记录运行编号、真实调用次数和真实观测上的 GPU/ONNX 数值对照；缺少模型或推理失败会明确报错。"
                if "BPU" in cell.source and cell.source.startswith("##"):
                    cell.source += "\n\n板端实践通过 `MICRODUCK_RUN_BPU=1` 启用；编译通过 `MICRODUCK_COMPILE_BPU=1` 启用。工作站推理可以独立运行，关闭板端时会输出 `NOT-RUN-BPU`。"
        # Old outputs must not be mistaken for results of the changed code.
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        notebook.metadata["inference_audit_version"] = 1
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(path.name)


if __name__ == "__main__":
    patch(Path(__file__).resolve().parent)
