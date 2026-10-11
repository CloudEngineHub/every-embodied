"""Update board evidence checks without rebuilding the teaching notebooks."""

from pathlib import Path
import nbformat


def patch(folder):
    for path in sorted(Path(folder).glob("0[1-7]_*.ipynb")):
        book = nbformat.read(path, 4)
        if book.metadata.get("bpu_audit_version") == 1:
            continue
        for cell in book.cells:
            if cell.cell_type != "code":
                continue
            source = cell.source
            if source.startswith("TASK_ID ="):
                source = source.replace("verify_gpu_report", "verify_gpu_report, verify_bpu_report")
            elif "BPU_TASK =" in source:
                source = source.replace("    import hashlib\n", "    import hashlib\n    import uuid\n    BPU_RUN_ID = str(uuid.uuid4())\n    bpu_env = os.environ.copy()\n    bpu_env['MICRODUCK_VALIDATION_RUN_ID'] = BPU_RUN_ID\n")
                source = source.replace('compiled/official_ladder_climber/model_output/official_ladder_climber.bin',
                    'compiled/official_ladder_climber_feature_normalized_all_int16/model_output/official_ladder_climber_feature_normalized_all_int16.bin')
                source = source.replace('compiled/official_ladder_getup/model_output/official_ladder_getup.bin',
                    'compiled/official_ladder_getup_feature_combined_all_int16/model_output/official_ladder_getup_feature_combined_all_int16.bin')
                source = source.replace("    def restart_rdk_policy_server():\n", "    def restart_rdk_policy_server():\n        global BPU_BOARD_SHA\n")
                source = source.replace('            if response.get("ok"):\n',
                    '            if response.get("ok"):\n                BPU_BOARD_SHA = response.get("sha256")\n                if BPU_BOARD_SHA != hashlib.sha256(BPU_HBM.read_bytes()).hexdigest():\n                    raise ValueError("板端模型 SHA 与所选 HBM 不一致。")\n')
                source = source.replace('        pair_cmd = [', '''        with socket.create_connection((RDK_HOST, RDK_CONTROL_PORT), timeout=8) as control:
            control.sendall(b'{"op":"switch_pair"}\\n')
            pair_status = json.loads(control.makefile("rb").readline())
        if not pair_status.get("ok"):
            raise RuntimeError(pair_status)
        pair_hashes = [pair_status["models"][role]["sha256"] for role in ("climber", "getup")]
        pair_cmd = [''')
                source = source.replace('env=os.environ.copy(), capture_output=True, text=True)',
                    'env=bpu_env, capture_output=True, text=True, timeout=600)')
                source = source.replace('video_cmd, cwd=PLAYGROUND_ROOT, env=os.environ.copy(),',
                    'video_cmd, cwd=PLAYGROUND_ROOT, env=bpu_env,')
                source = source.replace('print(json.loads(BPU_REPORT.read_text(encoding="utf-8")))',
                    'print(verify_bpu_report(BPU_VIDEO_OUTPUT, BPU_RUN_ID, [BPU_HBM], [BPU_BOARD_SHA]))\n            print("PASS-FRESH-BPU: 本次板端动作已驱动仿真并生成视频。")')
                source = source.replace('    SIM_COMMAND = \'\'\n    print("MuJoCo 回放命令模板:", SIM_COMMAND)', '')
                source = source.replace('print(json.loads(pair_report.read_text(encoding="utf-8")))',
                    'print(verify_bpu_report(BPU_LADDER_PAIR_OUTPUT, BPU_RUN_ID, [BPU_LADDER_CLIMBER_HBM, BPU_LADDER_GETUP_HBM], pair_hashes))\n                    print("PASS-FRESH-BPU: 本次双模型板端动作已驱动仿真并生成视频。")')
                source = source.replace('print(json.loads(report_path.read_text(encoding="utf-8")))',
                    'print(verify_bpu_report(BPU_VIDEO_OUTPUT, BPU_RUN_ID, [BPU_HBM], [BPU_BOARD_SHA]))\n                print("PASS-FRESH-BPU: 本次板端动作已驱动仿真并生成视频。")')
                # Existing lesson versions used either variable name.
                source = source.replace('print(json.loads(bpu_report.read_text(encoding="utf-8")))',
                    'print(verify_bpu_report(BPU_VIDEO_OUTPUT, BPU_RUN_ID, [BPU_HBM], [BPU_BOARD_SHA]))\n                print("PASS-FRESH-BPU: 本次板端动作已驱动仿真并生成视频。")')
            elif "BPU_SMOKE_MODEL" in source:
                source = '''RUN_BPU = os.getenv("MICRODUCK_RUN_BPU", "0") == "1"
if not RUN_BPU:
    print("NOT-RUN-BPU: 本轮只运行工作站推理。")
else:
    print("板端推理结果:")
    if BPU_TASK == "ladder":
        board_report = verify_bpu_report(BPU_LADDER_PAIR_OUTPUT, BPU_RUN_ID,
            [BPU_LADDER_CLIMBER_HBM, BPU_LADDER_GETUP_HBM], pair_hashes)
    else:
        board_report = verify_bpu_report(BPU_VIDEO_OUTPUT, BPU_RUN_ID, [BPU_HBM], [BPU_BOARD_SHA])
    print("PASS-FRESH-BPU: 模型 SHA、运行编号与实际控制步数已核对。")
    print("控制步数:", board_report["steps"])
    print("重置次数:", board_report.get("resets", board_report.get("terminated_or_truncated_resets", 0)))
    print("视频由本次任务策略驱动物理仿真生成。")'''
            if source != cell.source:
                cell.source = source
                cell.outputs = []
                cell.execution_count = None
        book.metadata["bpu_audit_version"] = 1
        nbformat.validate(book)
        nbformat.write(book, path)
        print(path.name)


if __name__ == "__main__":
    patch(Path(__file__).resolve().parent)
