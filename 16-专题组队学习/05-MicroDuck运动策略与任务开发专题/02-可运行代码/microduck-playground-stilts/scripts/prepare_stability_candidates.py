"""Prepare real-rollout INT16 calibration without replacing deployed models."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml


def verify_transformed_model(original, candidate, observations, transform):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    baseline = ort.InferenceSession(str(original), sess_options=options, providers=["CPUExecutionProvider"])
    transformed = ort.InferenceSession(str(candidate), sess_options=options, providers=["CPUExecutionProvider"])
    offset = np.asarray(transform["offset"], dtype=np.float32)
    scale = np.asarray(transform["scale"], dtype=np.float32)
    if not len(observations) or not np.isfinite(offset).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Missing calibration observations or invalid normalization")
    maximum = 0.0
    selected = observations[::max(1, len(observations) // 100)]
    for observation in selected:
        raw = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        expected = baseline.run(None, {baseline.get_inputs()[0].name: raw})[0]
        actual = transformed.run(None, {transformed.get_inputs()[0].name: (raw - offset) / scale})[0]
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError("Nonfinite float model output")
        maximum = max(maximum, float(np.abs(actual - expected).max()))
    if not np.isfinite(maximum) or maximum >= 1e-4:
        raise ValueError(f"Transformed float model changed actor outputs: {maximum}")
    return {"samples": len(selected), "max_abs_action_error": maximum,
            "source_onnx_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "candidate_onnx_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=["basketball", "ball_balance", "ladder_climber", "ladder_getup"], required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--feature-scale", action="store_true")
    parser.add_argument("--external-normalization", action="store_true")
    parser.add_argument("--split-weights", action="store_true")
    parser.add_argument("--extra-trace", type=Path)
    args = parser.parse_args()
    if args.feature_scale and args.external_normalization:
        parser.error("Choose one input transform")
    if args.split_weights and not args.external_normalization:
        parser.error("Split weights requires external normalization")
    if not args.trace and args.task != "basketball":
        parser.error("A real rollout trace is required for this task")
    root = args.root.resolve()
    prefix = args.task + "_restored_int16" if args.task == "ball_balance" else args.task + "_recalibrated_int16"
    if args.feature_scale:
        prefix += "_feature"
    if args.external_normalization:
        prefix += "_normalized"
    if args.split_weights:
        prefix += "_split"
    if args.extra_trace:
        prefix += "_expanded"
    calibration = root / "calibration" / prefix
    names = ["obs", "h_in", "c_in"] if args.task == "basketball" else ["obs"]
    shapes = ["1x61", "1x1x256", "1x1x256"] if args.task == "basketball" else ["1x54" if args.task == "ball_balance" else "1x61"]
    for name in names:
        (calibration / name).mkdir(parents=True, exist_ok=True)
    if args.trace:
        trace = np.load(args.trace)
        observations = trace["obs"] if "obs" in trace else trace["observations"].copy()
        if args.task.startswith("ladder"):
            import json
            observations[1:, 34:48] = trace["raw_actions"][:-1]
            observations[0, 34:48] = 0
            report = json.loads(args.trace.with_name(args.trace.name.replace(".trace.npz", ".json")).read_text())
            switch = report["switch_events"][0]["step"]
            observations = observations[:switch] if args.task == "ladder_climber" else observations[switch:]
        if args.extra_trace:
            observations = np.concatenate([observations, np.load(args.extra_trace)["obs"]])
        if args.external_normalization:
            import json
            import onnx
            from onnx import numpy_helper
            model = onnx.load(args.onnx)
            original_onnx = args.onnx
            constants = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
            normalizer_sub = next(n for n in model.graph.node if "normalizer" in n.name and n.op_type == "Sub")
            mean = constants[normalizer_sub.input[1]].reshape(-1)
            normalizer_div = next(n for n in model.graph.node if "normalizer" in n.name and n.op_type == "Div")
            std = constants[normalizer_div.input[1]].reshape(-1)
            normalized = normalizer_div.output[0]
            normalizer_prefix = normalizer_div.name.rsplit("/", 1)[0] + "/"
            kept = [n for n in model.graph.node if not n.name.startswith(normalizer_prefix)]
            for node in kept:
                for index, name in enumerate(node.input):
                    if name == normalized:
                        node.input[index] = "obs"
            del model.graph.node[:]
            model.graph.node.extend(kept)
            if args.split_weights:
                from onnx import helper
                nodes = []
                for node in model.graph.node:
                    if node.op_type != "Gemm":
                        nodes.append(node)
                        continue
                    weight = constants[node.input[1]]
                    step = np.maximum(np.max(np.abs(weight), axis=1, keepdims=True) / 127, 1e-12)
                    coarse = (np.rint(weight / step) * step).astype(np.float32)
                    residual = (weight - coarse).astype(np.float32)
                    coarse_name, residual_name = node.input[1] + "_coarse", node.input[1] + "_residual"
                    zero_name = node.input[2] + "_zero"
                    model.graph.initializer.extend([numpy_helper.from_array(coarse, coarse_name),
                        numpy_helper.from_array(residual, residual_name),
                        numpy_helper.from_array(np.zeros_like(constants[node.input[2]]), zero_name)])
                    high, low = copy.deepcopy(node), copy.deepcopy(node)
                    high.name += "_coarse"
                    low.name += "_residual"
                    high.input[1], low.input[1], low.input[2] = coarse_name, residual_name, zero_name
                    high.output[0], low.output[0] = node.output[0] + "_coarse", node.output[0] + "_residual"
                    nodes.extend([high, low, helper.make_node("Add", [high.output[0], low.output[0]], list(node.output), name=node.name + "_precision_sum")])
                del model.graph.node[:]
                model.graph.node.extend(nodes)
            args.onnx = root / "converted" / (prefix + ".onnx")
            onnx.checker.check_model(model)
            onnx.save(model, args.onnx)
            transform = {"offset": mean.tolist(), "scale": std.tolist()}
            parity = verify_transformed_model(original_onnx, args.onnx, observations, transform)
            args.onnx.with_suffix(".float_parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
            args.onnx.with_suffix(".input_transform.json").write_text(json.dumps(transform), encoding="utf-8")
            observations = (observations - mean) / std
        if args.feature_scale:
            import onnx
            from onnx import helper, numpy_helper
            scales = np.maximum(np.max(np.abs(observations), axis=0), 0.05).astype(np.float32)
            model = onnx.load(args.onnx)
            for node in model.graph.node:
                for index, name in enumerate(node.input):
                    if name == "obs":
                        node.input[index] = "obs_raw"
            model.graph.initializer.append(numpy_helper.from_array(scales, name="input_feature_scales"))
            model.graph.node.insert(0, helper.make_node("Mul", ["obs", "input_feature_scales"], ["obs_raw"], name="restore_input_units"))
            args.onnx = root / "converted" / (prefix + ".onnx")
            onnx.checker.check_model(model)
            onnx.save(model, args.onnx)
            args.onnx.with_suffix(".input_transform.json").write_text(__import__("json").dumps(scales.tolist()))
            observations = observations / scales
        for i, obs in enumerate(observations):
            np.asarray(obs, dtype=np.float32).tofile(calibration / "obs" / f"{i:05d}.bin")
    else:
        sources = [root / "calibration/basketball" / n for n in ["obs_f32", "h_f32", "c_f32"]]
        files = sorted(sources[0].glob("*"))
        # Evenly cover the trajectory and retain large cell states as a matched triplet.
        selected = set(np.linspace(0, len(files) - 1, 480).astype(int).tolist())
        selected.update(np.argsort([np.max(np.abs(np.fromfile(sources[2] / f.name, dtype=np.float32))) for f in files])[-32:].tolist())
        for i, index in enumerate(sorted(selected)):
            for name, source in zip(names, sources):
                shutil.copyfile(source / files[index].name, calibration / name / f"{i:05d}.bin")
    config = yaml.safe_load((root / "basketball_kl512_config.yaml").read_text())
    mount = lambda p: "/work/" + str(p.relative_to(root.parent))
    config["model_parameters"].update(onnx_model=mount(args.onnx.resolve()), output_model_file_prefix=prefix,
                                      working_dir=mount(root / "compiled" / prefix / "model_output"))
    inputs = config["input_parameters"]
    inputs.update(input_name=";".join(names), input_shape=";".join(shapes))
    for key in ["input_type_rt", "input_type_train"]:
        inputs[key] = ";".join(["featuremap"] * len(names))
    for key in ["input_layout_rt", "input_layout_train"]:
        inputs[key] = ";".join(["NCHW"] * len(names))
    inputs["norm_type"] = ";".join(["no_preprocess"] * len(names))
    config["calibration_parameters"].update(calibration_type="max", optimization="set_all_nodes_int16",
        cal_data_dir=";".join(mount(calibration / n) for n in names),
        cal_data_type=";".join(["float32"] * len(names)), preprocess_on=False,
        per_channel=args.task != "basketball")
    path = root / (prefix + "_config.yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
