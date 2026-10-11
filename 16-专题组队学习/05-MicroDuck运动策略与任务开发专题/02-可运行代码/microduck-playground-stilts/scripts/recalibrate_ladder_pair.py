"""Compile per-channel INT16 candidates using the captured ladder trajectories."""

import argparse
from pathlib import Path
import subprocess

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    configs = {
        "climber": "official_ladder_climber_feature_normalized_all_int16.yaml",
        "getup": "config_official_ladder_getup_feature_combined_all_int16.yaml",
    }
    for role, filename in configs.items():
        prefix = "official_ladder_" + role + "_per_channel_int16"
        config = yaml.safe_load((root / filename).read_text())
        config["calibration_parameters"]["per_channel"] = True
        config["model_parameters"].update(output_model_file_prefix=prefix,
            working_dir="/work/" + str((root / "compiled" / prefix / "model_output").relative_to(root.parent)))
        path = root / (prefix + "_config.yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        with (root / (prefix + ".log")).open("w") as log:
            subprocess.run(["docker", "run", "--rm", "-v", str(root.parent) + ":/work",
                "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8", "hb_mapper", "makertbin",
                "--model-type", "onnx", "--config", "/work/" + str(path.relative_to(root.parent))],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        print(root / "compiled" / prefix / "model_output" / (prefix + ".bin"), flush=True)


if __name__ == "__main__":
    main()
