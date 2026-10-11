"""Select verified policies while keeping short training runs as candidates."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = "/home/ubuntu/workspaces/microduck_bpu_models"
BALL_HBM = MODEL_ROOT + "/compiled/ball_balance_restored_int16_normalized_split_expanded/model_output/ball_balance_restored_int16_normalized_split_expanded.bin"
BASKET_HBM = MODEL_ROOT + "/compiled/basketball_recalibrated_int16/model_output/basketball_recalibrated_int16.bin"


def main():
    for path in sorted(ROOT.glob("0[1-7]_*.ipynb")):
        notebook = nbformat.read(path, 4)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            old = cell.source
            source = old.replace("converted/motrix_ball_balance.opset11.onnx", "converted/motrix_ball_balance_restored.opset11.onnx")
            if path.name.startswith("01_"):
                source = source.replace('os.getenv("MICRODUCK_BASKETBALL_SMOKE_ITERATIONS", "5")', 'os.getenv("MICRODUCK_BASKETBALL_SMOKE_ITERATIONS", "10")')
                source = source.replace("        MODEL_PATH = SMOKE_ONNX", "        CANDIDATE_MODEL_PATH = SMOKE_ONNX\n        print('Training candidate:', CANDIDATE_MODEL_PATH, 'Stable inference model:', MODEL_PATH)")
            if path.name.startswith(("03_", "04_")):
                source = source.replace("            MODEL_PATH = DEMO_ONNX", "            CANDIDATE_MODEL_PATH = DEMO_ONNX\n            print('Training candidate:', CANDIDATE_MODEL_PATH, 'Stable inference model:', MODEL_PATH)")
            source = source.replace(MODEL_ROOT + "/compiled/ball_balance/model_output/ball_balance.opset11.bin", BALL_HBM)
            source = source.replace(MODEL_ROOT + "/compiled/basketball/model_output/basketball.opset11.bin", BASKET_HBM)
            source = source.replace('os.getenv("MICRODUCK_GPU_VIDEO_STEPS", "250")', 'os.getenv("MICRODUCK_GPU_VIDEO_STEPS", "1500" if GPU_TASK == "basketball" else "900" if GPU_TASK == "ball_balance" else "500")')
            source = source.replace('os.getenv("MICRODUCK_BPU_VIDEO_STEPS", "250")', 'os.getenv("MICRODUCK_BPU_VIDEO_STEPS", "1500" if BPU_TASK == "basketball" else "900" if BPU_TASK == "ball_balance" else "500")')
            if '"op": "upload_and_switch"' in source:
                marker = '            with socket.create_connection((RDK_HOST, RDK_CONTROL_PORT), timeout=8) as control:'
                normalization = '            transform_path = BPU_HBM.with_suffix(".input_transform.json")\n            if payload and transform_path.is_file():\n                request["input_transform"] = json.loads(transform_path.read_text(encoding="utf-8"))\n'
                if normalization not in source:
                    source = source.replace(marker, normalization + marker)
            if path.name.startswith("05_") and "FASTSAC_ITERATIONS =" in source and "RESTORED_CHECKPOINT =" not in source:
                source = source.replace('os.getenv("MICRODUCK_FASTSAC_ITERATIONS", "5")', 'os.getenv("MICRODUCK_FASTSAC_ITERATIONS", "10")')
                source = source.replace('RUN_FASTSAC_SMOKE =', 'TRAIN_MODE = os.getenv("MICRODUCK_TRAIN_MODE", "pretrained")\nRESTORED_CHECKPOINT = Path(os.getenv("MICRODUCK_FASTSAC_CHECKPOINT", "' + MODEL_ROOT + '/restored_ball_balance/checkpoints/model_0005002.pt"))\nRUN_FASTSAC_SMOKE =', 1)
                source = source.replace('    print("\u5f00\u59cb FastSAC smoke \u8bad\u7ec3:', '    if TRAIN_MODE == "pretrained":\n        import torch\n        starting_step = int(torch.load(RESTORED_CHECKPOINT, map_location="cpu", weights_only=True)["global_step"])\n        train_cmd[-1] = f"algo.trainer.num_learning_iterations={starting_step + FASTSAC_ITERATIONS}"\n        train_cmd.append(f"resume={RESTORED_CHECKPOINT}")\n    elif TRAIN_MODE != "scratch":\n        raise ValueError("MICRODUCK_TRAIN_MODE must be pretrained or scratch")\n    print("\u5f00\u59cb FastSAC smoke \u8bad\u7ec3:')
                # Read the starting step in the task environment, not the notebook kernel.
                source = source.replace('        import torch\n        starting_step = int(torch.load(RESTORED_CHECKPOINT, map_location="cpu", weights_only=True)["global_step"])', '        probe = subprocess.run([str(MOTRIX_PYTHON), "-c", "import torch,sys; print(torch.load(sys.argv[1], map_location=\\\"cpu\\\", weights_only=True)[\\\"global_step\\\"])", str(RESTORED_CHECKPOINT)], check=True, capture_output=True, text=True)\n        starting_step = int(probe.stdout.strip())')
                source = source.replace('    MODEL_PATH = FASTSAC_ONNX', '    CANDIDATE_MODEL_PATH = FASTSAC_ONNX\n    print("Training candidate:", CANDIDATE_MODEL_PATH, "Stable inference model:", MODEL_PATH)')
            if path.name.startswith("05_") and source.startswith("COMPILE_BPU ="):
                source = '''COMPILE_BPU = os.getenv("MICRODUCK_COMPILE_BPU", "0") == "1"
if not COMPILE_BPU:
    print("NOT-RUN-COMPILE: \u672c\u8f6e\u4f7f\u7528\u5df2\u6821\u51c6\u6a21\u578b\u3002")
else:
    BPU_MODEL_ROOT = Path(os.getenv("MICRODUCK_BPU_MODEL_ROOT", "/home/ubuntu/workspaces/microduck_bpu_models"))
    calibration_video = OUTPUT_ROOT / "diagnostics/calibration/ball_balance_gpu.mp4"
    calibration_video.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        [str(MOTRIX_PYTHON), str(MOTRIX_RUNNER), "video", str(PLAYGROUND_ROOT / "scripts/motrix_gpu_inloop_video.py"), "--onnx", str(MODEL_PATH), "--steps", "1000", "--output", str(calibration_video)],
        [str(POLICY_PYTHON), str(PLAYGROUND_ROOT / "scripts/prepare_stability_candidates.py"), "--root", str(BPU_MODEL_ROOT), "--task", "ball_balance", "--onnx", str(MODEL_PATH), "--trace", str(calibration_video.with_suffix(".trace.npz")), "--external-normalization", "--split-weights"],
        ["docker", "run", "--rm", "-v", str(BPU_MODEL_ROOT.parent) + ":/work", os.getenv("MICRODUCK_BPU_DOCKER_IMAGE", "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8"), "hb_mapper", "makertbin", "--model-type", "onnx", "--config", "/work/" + str((BPU_MODEL_ROOT / "ball_balance_restored_int16_normalized_split_config.yaml").relative_to(BPU_MODEL_ROOT.parent))],
    ]
    for command in commands:
        subprocess.run(command, cwd=PLAYGROUND_ROOT, check=True)
    print("PASS-X5-HBM: \u771f\u5b9e\u8f68\u8ff9\u6821\u51c6\u5b8c\u6210\uff0c\u5019\u9009\u6a21\u578b\u4fdd\u5b58\u5230 compiled/ball_balance_restored_int16_normalized_split\u3002")
'''
            if path.name.startswith("06_"):
                source = source.replace("compiled/official_ladder_climber_feature_normalized_all_int16/model_output/official_ladder_climber_feature_normalized_all_int16.bin", "compiled/ladder_climber_recalibrated_int16_normalized_split/model_output/ladder_climber_recalibrated_int16_normalized_split.bin")
                source = source.replace("compiled/official_ladder_getup_feature_combined_all_int16/model_output/official_ladder_getup_feature_combined_all_int16.bin", "compiled/ladder_getup_recalibrated_int16_normalized_split/model_output/ladder_getup_recalibrated_int16_normalized_split.bin")
                source = source.replace('"--seconds", "24"', '"--seconds", "30", "--physics-device", "cpu"')
                source = source.replace('os.getenv("MICRODUCK_LADDER_SECONDS", "24")', 'os.getenv("MICRODUCK_LADDER_SECONDS", "30"), "--physics-device", "cpu"')
            if source != old:
                cell.source = source
                cell.outputs = []
                cell.execution_count = None
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(path.name)


if __name__ == "__main__":
    main()
