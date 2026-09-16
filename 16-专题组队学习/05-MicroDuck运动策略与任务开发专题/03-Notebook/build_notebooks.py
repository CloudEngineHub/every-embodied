from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def md(value: str):
    return nbf.v4.new_markdown_cell(value.strip())


def code(value: str):
    return nbf.v4.new_code_cell(value.strip())


SETUP = r'''
from pathlib import Path
import json
import os
import platform
import shlex
import socket
import subprocess


def locate_topic_root():
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if (base / "02-可运行代码").is_dir() and (base / "01-任务资料").is_dir():
            return base
        group = base / "16-专题组队学习"
        if group.is_dir():
            matches = list(group.glob("*/02-可运行代码"))
            if matches:
                return matches[0].parent
    configured = os.getenv("MICRODUCK_TOPIC_ROOT")
    if configured and Path(configured).is_dir():
        return Path(configured).resolve()
    raise RuntimeError("找不到专题目录，请从 03-Notebook 启动 Jupyter，或设置 MICRODUCK_TOPIC_ROOT。")


TOPIC_ROOT = locate_topic_root()
PLAYGROUND_ROOT = TOPIC_ROOT / "02-可运行代码" / "microduck-playground-stilts"
OUTPUT_ROOT = TOPIC_ROOT / "03-Notebook" / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("专题目录:", TOPIC_ROOT)
print("可运行代码:", PLAYGROUND_ROOT)
print("输出目录:", OUTPUT_ROOT)
print("Jupyter 工作站:", socket.gethostname(), platform.system(), platform.machine())
print("RDK 地址:", os.getenv("RDK_HOST", "192.168.8.128"), "用户: sunrise")
'''


def task_cells(model_hint: str | None, task_id: str | None, sim_command: str, direct_connect: bool = False, video_hint: str | None = None, keyframe_hint: str | None = None):
    hint = repr(model_hint) if model_hint else "None"
    task = repr(task_id) if task_id else "None"
    if direct_connect:
        stage3_md = md("""
## 3. 导航策略：直接接入已有模型

导航/部署入口不重复训练，直接使用已有 ONNX。默认读取本 Notebook 目录下的 `outputs/microduck_demo_latest.onnx`；也可以通过 `MICRODUCK_NAV_ONNX` 指定其他已经训练好的导航策略。
""")
        stage3_code = code(r'''
NAV_ONNX = os.getenv("MICRODUCK_NAV_ONNX", "")
default_nav_onnx = OUTPUT_ROOT / "microduck_demo_latest.onnx"
if NAV_ONNX:
    MODEL_PATH = Path(NAV_ONNX).expanduser().resolve()
elif default_nav_onnx.exists():
    MODEL_PATH = default_nav_onnx
print("导航策略直接接入:", MODEL_PATH if MODEL_PATH and MODEL_PATH.exists() else "未找到")
if MODEL_PATH is None or not MODEL_PATH.exists():
    print("请设置 MICRODUCK_NAV_ONNX=/path/to/navigation.onnx 后重新运行本单元。")
''')
    else:
        stage3_md = md("""
## 3. 固定演示模型：恢复并 smoke 训练 10 个 iteration

为了让直播每次都得到同一份可追踪产物，本单元默认从已验证的行走 checkpoint 恢复，在 GPU 上用 64 个并行环境继续训练 10 个 PPO iteration，然后覆盖 `outputs/microduck_demo_latest.pt` 和 `outputs/microduck_demo_latest.onnx`。原始 `model_5999.pt` 不会被修改。

10 个 iteration 不是“重新训练出一个新能力”，而是验证 `checkpoint -> MuJoCo/Warp -> PPO 更新 -> checkpoint -> ONNX` 全链路。每个 iteration 默认采集 `64 × 24 = 1536` 条环境步，所以本次约更新 15360 条 transition。
""")
        stage3_code = code(r'''
import shutil
import sys

TRAIN_ROOT = Path(os.getenv("MICRODUCK_WORKSPACE", "/home/ubuntu/workspaces/microduck_rl"))
TRAIN_TASK = os.getenv("MICRODUCK_TRAIN_TASK", "Mjlab-Velocity-Flat-MicroDuck")
TRAIN_SOURCE_RUN = os.getenv(
    "MICRODUCK_TRAIN_RUN",
    "2026-09-03_20-05-56_every-embodied-4096x6000",
)
TRAIN_SOURCE_CHECKPOINT = os.getenv("MICRODUCK_TRAIN_CHECKPOINT", "model_5999.pt")
TRAIN_ENVS = int(os.getenv("MICRODUCK_TRAIN_ENVS", "64"))
TRAIN_ITERATIONS = int(os.getenv("MICRODUCK_TRAIN_ITERATIONS", "10"))
RUN_TRAIN_SMOKE = os.getenv("MICRODUCK_RUN_TRAIN_SMOKE", "1") == "1"
DEMO_PT = OUTPUT_ROOT / "microduck_demo_latest.pt"
DEMO_ONNX = OUTPUT_ROOT / "microduck_demo_latest.onnx"

source_path = TRAIN_ROOT / "logs" / "rsl_rl" / "velocity" / TRAIN_SOURCE_RUN / TRAIN_SOURCE_CHECKPOINT
print("训练工作区:", TRAIN_ROOT)
print("恢复 checkpoint:", source_path)
print("固定输出:", DEMO_PT, DEMO_ONNX)

if not RUN_TRAIN_SMOKE:
    print("已跳过 smoke 训练：设置 MICRODUCK_RUN_TRAIN_SMOKE=1 后重新运行本单元。")
elif not TRAIN_ROOT.is_dir():
    print("找不到 Ubuntu 训练工作区；请在 Ubuntu Jupyter 中运行，或设置 MICRODUCK_WORKSPACE。")
elif not source_path.is_file():
    print("找不到恢复 checkpoint；请设置 MICRODUCK_TRAIN_RUN / MICRODUCK_TRAIN_CHECKPOINT。")
else:
    train_cmd = [
        sys.executable, "-m", "mjlab.scripts.train", TRAIN_TASK,
        "--env.scene.num-envs", str(TRAIN_ENVS),
        "--agent.max-iterations", str(TRAIN_ITERATIONS),
        "--agent.resume", "True",
        "--agent.load-run", TRAIN_SOURCE_RUN,
        "--agent.load-checkpoint", TRAIN_SOURCE_CHECKPOINT,
        "--agent.experiment-name", "velocity",
        "--agent.run-name", "demo-latest-10-step",
        "--agent.save-interval", str(TRAIN_ITERATIONS),
        "--video", "False",
    ]
    train_env = os.environ.copy()
    train_env["WANDB_MODE"] = "offline"
    print("开始 GPU smoke 训练:", " ".join(shlex.quote(x) for x in train_cmd))
    train_result = subprocess.run(train_cmd, cwd=TRAIN_ROOT, env=train_env, text=True)
    if train_result.returncode != 0:
        print("训练失败，未覆盖固定演示模型。returncode:", train_result.returncode)
    else:
        demo_runs = sorted(
            (TRAIN_ROOT / "logs" / "rsl_rl" / "velocity").glob("*_demo-latest-10-step"),
            key=lambda p: p.stat().st_mtime,
        )
        if not demo_runs:
            raise FileNotFoundError("训练成功但没有找到 demo-latest-10-step 输出目录。")
        demo_run = demo_runs[-1]
        checkpoints = sorted(
            demo_run.glob("model_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"训练输出目录没有 checkpoint: {demo_run}")
        latest_checkpoint = checkpoints[-1]
        shutil.copy2(latest_checkpoint, DEMO_PT)
        export_cmd = [
            sys.executable, "scripts/export.py", TRAIN_TASK,
            "--checkpoint-file", str(latest_checkpoint),
            "--onnx-file", str(DEMO_ONNX),
        ]
        print("导出固定 ONNX:", " ".join(shlex.quote(x) for x in export_cmd))
        export_result = subprocess.run(export_cmd, cwd=TRAIN_ROOT, env=train_env, text=True)
        if export_result.returncode != 0:
            print("ONNX 导出失败，保留 checkpoint，returncode:", export_result.returncode)
        else:
            MODEL_PATH = DEMO_ONNX
            print("PASS-demo-model:", MODEL_PATH)
            print("本次实际使用 checkpoint:", latest_checkpoint)
''')
    return [
        md("""
## 1. 模型契约与本地检查

目标契约是 actor `obs[1, 61] -> action[1, 14]`。如果本任务没有随专题提交 checkpoint/ONNX，Notebook 会明确显示“模型未提供”，不会用别的任务模型冒充当前任务结果。
"""),
        code(f'''
TASK_ID = {task}
MODEL_HINT = {hint}
MODEL_PATH = Path(os.getenv("MICRODUCK_ONNX", "")) if os.getenv("MICRODUCK_ONNX") else None
if MODEL_PATH is None and MODEL_HINT:
    candidate = TOPIC_ROOT / MODEL_HINT
    if candidate.exists():
        MODEL_PATH = candidate
if MODEL_PATH is None and not MODEL_HINT and not TASK_ID:
    candidates = sorted(TOPIC_ROOT.glob("01-任务资料/**/*.onnx"))
    MODEL_PATH = candidates[0] if candidates else None
print("任务:", TASK_ID or "接口/网页演示")
print("ONNX:", MODEL_PATH if MODEL_PATH else "未提供")
if TASK_ID and MODEL_PATH is None:
    print("该任务尚未随专题提交任务专属 ONNX；请设置 MICRODUCK_ONNX 后再运行模型检查。")
'''),
        code(r'''
def inspect_onnx(path):
    try:
        import onnx
    except ImportError:
        print("缺少 onnx：请在 microduck-playground 环境中启动 Jupyter。")
        return None
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    def shape(value):
        return [d.dim_value if d.dim_value else (d.dim_param or "?") for d in value.type.tensor_type.shape.dim]
    result = {
        "inputs": [(item.name, shape(item)) for item in model.graph.input],
        "outputs": [(item.name, shape(item)) for item in model.graph.output],
        "nodes": len(model.graph.node),
        "metadata": {item.key: item.value for item in model.metadata_props},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

contract = inspect_onnx(MODEL_PATH) if MODEL_PATH else None
'''),
        md(f"""
## 2. ONNX 导出入口

训练得到的是 RSL-RL checkpoint；导出时要把 actor 和观测归一化一起固化到 ONNX。下面是本任务命令模板。`--video` 会在本机录制 MuJoCo 回放，完整视频写入 `outputs/`，不提交到 Git。

```bash
uv run python scripts/export.py {task_id or '<TASK_ID>'} \\
  --checkpoint-file <CHECKPOINT.pt> \\
  --onnx-file outputs/policy.onnx \\
  --num-envs 1 --video --video-length 250
```
"""),
        code(r'''
EXPORT_COMMAND = "uv run python scripts/export.py " + (TASK_ID or "<TASK_ID>") + " --checkpoint-file <CHECKPOINT.pt> --onnx-file outputs/policy.onnx --num-envs 1 --video --video-length 250"
print(EXPORT_COMMAND)
print("导出前后检查：观测维度、动作顺序、归一化和 action clip。")
'''),
        stage3_md,
        stage3_code,
        md("""
## 4. 本地 ONNX 基准推理

这一步只验证 ONNX 图能在本机运行，以及输出形状和数值是否有限；它不是训练效果评测，也不是 BPU 验收。
"""),
        code(r'''
def onnx_local_smoke(path):
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print("缺少本地推理依赖:", exc)
        return None
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {}
    for item in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in item.shape]
        feeds[item.name] = np.zeros(shape, dtype=np.float32)
    outputs = session.run(None, feeds)
    info = {"providers": session.get_providers(), "inputs": [(x.name, x.shape) for x in session.get_inputs()], "outputs": [(x.name, x.shape) for x in session.get_outputs()], "finite": all(np.isfinite(x).all() for x in outputs)}
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    return outputs

if MODEL_PATH:
    local_outputs = onnx_local_smoke(MODEL_PATH)
else:
    print("跳过：尚未提供该任务 ONNX。")
'''),
        md("""
## 5. MuJoCo 展示

直播默认展示短视频或关键帧，避免网页 WebAssembly、浏览器 GPU 或 OpenGL 窗口打断讲解。需要交互时，再启动专题中的网页服务；需要连续闭环时，使用任务自己的回放脚本。
"""),
        code(f'''
VIDEO_OVERRIDE = os.getenv("MICRODUCK_VIDEO")
generated_video = OUTPUT_ROOT / "microduck_demo_latest.mp4"
video_candidates = [Path(VIDEO_OVERRIDE)] if VIDEO_OVERRIDE else []
VIDEO_HINT = {video_hint!r}
KEYFRAME_HINT = {keyframe_hint!r}
if not VIDEO_OVERRIDE and VIDEO_HINT:
    candidate = TOPIC_ROOT / VIDEO_HINT
    if candidate.exists():
        video_candidates.append(candidate)
if {direct_connect!r} and not VIDEO_OVERRIDE and not video_candidates and generated_video.exists():
    video_candidates.append(generated_video)
keyframes = []
if KEYFRAME_HINT:
    candidate = TOPIC_ROOT / KEYFRAME_HINT
    if candidate.exists():
        keyframes.append(candidate)
try:
    from IPython.display import Image, Video, display
    if video_candidates and video_candidates[0].exists():
        if video_candidates[0].suffix.lower() == ".gif":
            display(Image(filename=str(video_candidates[0])))
        else:
            display(Video(str(video_candidates[0]), embed=True))
        print("展示任务视频:", video_candidates[0])
    elif keyframes:
        display(Image(filename=str(keyframes[0])))
        print("该任务暂无视频，展示任务关键帧:", keyframes[0])
    else:
        print("该任务暂无视频；请设置 MICRODUCK_VIDEO，或先运行该任务自己的回放脚本。")
except Exception as exc:
    print("展示失败:", exc)
SIM_COMMAND = {sim_command!r}
print("MuJoCo 回放命令模板:", SIM_COMMAND)
'''),
        md("""
## 6. RDK X5 / BPU 探测与验收

当前仓库已有的 RDK 服务器默认是 `CPUExecutionProvider`。本单元只在板端真实连通并且存在 BPU 工具时继续；不会把 CPU 推理结果写成 BPU 成功。真实验收需要板端 HBM 模型和与 SDK 版本匹配的命令。
"""),
        code(r'''
import time

RDK_HOST = os.getenv("RDK_HOST", "192.168.8.128")
BPU_SMOKE_MODEL = os.getenv(
    "RDK_BPU_SMOKE_MODEL",
    "/opt/tros/humble/lib/dnn_benchmark_example/config/X5/mobilenetv1_224x224_nv12.bin",
)
BPU_SMOKE_INPUT_BYTES = int(os.getenv("RDK_BPU_SMOKE_INPUT_BYTES", "75264"))
BPU_HBM = os.getenv("RDK_BPU_HBM", "")
print("默认 BPU 样例模型:", BPU_SMOKE_MODEL)
print("默认 BPU 样例输入:", BPU_SMOKE_INPUT_BYTES, "bytes uint8")
print("MicroDuck 策略 HBM:", BPU_HBM or "未提供（先运行默认 BPU 样例）")
probe_script = "for x in hrt_model_exec hb_mapper hrt_bin_dump hrt_bin_info python3; do if command -v $x >/dev/null 2>&1; then echo $x=$(command -v $x); else echo $x=MISSING; fi; done"
remote = "sunrise@" + RDK_HOST
probe = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", remote, "bash -lc " + shlex.quote(probe_script)], capture_output=True, text=True)
if probe.returncode:
    print("RDK 暂不可达，跳过板端验收:", probe.stderr.strip() or "ssh failed")
else:
    print("RDK 已连通:", RDK_HOST)
    print(probe.stdout)
    if not BPU_HBM:
        print("未提供策略 HBM；使用上面的默认 BPU 样例验证板端运行时链路。")
        smoke_model = BPU_SMOKE_MODEL
        smoke_bytes = BPU_SMOKE_INPUT_BYTES
        smoke_info_cmd = "hrt_model_exec model_info --model_file " + shlex.quote(smoke_model)
        smoke_info = subprocess.run(["ssh", remote, "bash -lc " + shlex.quote(smoke_info_cmd)], capture_output=True, text=True)
        print("BPU 样例 model_info exit:", smoke_info.returncode)
        print(smoke_info.stdout[-4000:])
        print(smoke_info.stderr[-1000:])
        if smoke_info.returncode == 0:
            import numpy as np
            local_input = OUTPUT_ROOT / "rdk_bpu_runtime_smoke.bin"
            np.zeros(smoke_bytes, dtype=np.uint8).tofile(local_input)
            remote_dir = "/tmp/microduck_notebook"
            remote_input = remote_dir + "/rdk_bpu_runtime_smoke.bin"
            subprocess.run(["ssh", remote, "mkdir", "-p", remote_dir], check=True)
            subprocess.run(["scp", str(local_input), remote + ":" + remote_input], check=True)
            dump_dir = remote_dir + "/runtime_dump"
            infer_cmd = "rm -rf " + shlex.quote(dump_dir) + " && mkdir -p " + shlex.quote(dump_dir) + " && hrt_model_exec infer --model_file " + shlex.quote(smoke_model) + " --input_file " + shlex.quote(remote_input) + " --enable_dump true --dump_format txt --dump_path " + shlex.quote(dump_dir)
            started = time.perf_counter()
            infer = subprocess.run(["ssh", remote, "bash -lc " + shlex.quote(infer_cmd)], capture_output=True, text=True)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print("BPU runtime sample exit:", infer.returncode, "round-trip ms:", round(elapsed_ms, 2))
            print(infer.stdout[-4000:])
            print(infer.stderr[-1000:])
            if infer.returncode == 0:
                print("PASS-rdk-bpu-runtime-sample: X5 BPU 样例推理已返回 0。")
    else:
        info_cmd = "hrt_model_exec model_info --model_file " + shlex.quote(BPU_HBM)
        info = subprocess.run(["ssh", remote, "bash -lc " + shlex.quote(info_cmd)], capture_output=True, text=True)
        print("HBM model_info exit:", info.returncode)
        print(info.stdout[-6000:])
        print(info.stderr[-2000:])
        if info.returncode == 0:
            # hrt_model_exec consumes binary tensors; this zero observation is only
            # a transport/BPU smoke input, not a task-quality evaluation.
            import numpy as np
            local_input = OUTPUT_ROOT / "zero_obs_61_f32.bin"
            np.zeros((1, 61), dtype=np.float32).tofile(local_input)
            remote_dir = "/tmp/microduck_notebook"
            remote_input = remote_dir + "/zero_obs_61_f32.bin"
            subprocess.run(["ssh", remote, "mkdir", "-p", remote_dir], check=True)
            subprocess.run(["scp", str(local_input), remote + ":" + remote_input], check=True)
            dump_dir = remote_dir + "/dump"
            infer_cmd = "rm -rf " + shlex.quote(dump_dir) + " && mkdir -p " + shlex.quote(dump_dir) + " && hrt_model_exec infer --model_file " + shlex.quote(BPU_HBM) + " --input_file " + shlex.quote(remote_input) + " --enable_dump true --dump_format txt --dump_path " + shlex.quote(dump_dir)
            started = time.perf_counter()
            infer = subprocess.run(["ssh", remote, "bash -lc " + shlex.quote(infer_cmd)], capture_output=True, text=True)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print("BPU infer exit:", infer.returncode, "round-trip ms:", round(elapsed_ms, 2))
            print(infer.stdout[-6000:])
            print(infer.stderr[-2000:])
            if infer.returncode == 0:
                print("PASS-rdk-bpu: hrt_model_exec infer 已返回 0。")
            else:
                print("未通过：请确认 HBM 输入是否确实为 [1, 61] float32，以及模型是否为单输入模型。")
'''),
        md("""
## 7. 直播结论

- `PASS-local-onnx`：ONNX checker 和本地 ONNX Runtime 通过。
- `PASS-mujoco`：使用相同观测/动作契约完成 MuJoCo 回放。
- `PASS-rdk-bpu-runtime-sample`：板端已安装的 X5 BPU 样例推理退出码为 0。
- `PASS-rdk-bpu`：SSH 连通、板端明确选择 BPU、MicroDuck HBM 推理命令退出码为 0，并记录输入输出形状与延迟。

在 `PASS-rdk-bpu` 之前，不能把 RDK 的 CPUExecutionProvider 或“模型文件能复制到板子”称为 BPU 部署。
"""),
    ]


TASKS = [
    ("01_篮球平衡_PPO_ONNX_BPU_MuJoCo.ipynb", "篮球平衡 / PPO", "Mjlab-Basketball-MicroDuck", None, "", False, "01-任务资料/01-MicroDuck篮球平衡强化学习/assets/preview.gif", None),
    ("02_浏览器物理扰动_回放与接口.ipynb", "浏览器物理扰动 / MuJoCo Web", None, None, "python 任务资料/02 的 serve_mjswan.py", False, "01-任务资料/02-mjswan-MicroDuck浏览器物理扰动/assets/microduck_official.gif", "01-任务资料/02-mjswan-MicroDuck浏览器物理扰动/assets/mjswan_microduck_manual_drag_demo_keyframes.jpg"),
    ("03_高跷行走_课程与ONNX_BPU.ipynb", "高跷行走 / 形态课程", "Mjlab-Stilt-Flat-MicroDuck", None, "uv run python scripts/infer_policy.py --walking <STILT.onnx> --new-cmd-obs", False, None, "01-任务资料/03-MicroDuck高跷行走强化学习复现/assets/microduck_stilts_25cm_reproduced_keyframes.jpg"),
    ("04_摆动旋转_ONNX_BPU_MuJoCo.ipynb", "摆动旋转 / 自激摆动", "Mjlab-SwingPump-MicroDuck", "01-任务资料/04-MicroDuck摆动旋转强化学习复现/assets/microduck_swing_alpha050.onnx", "uv run python scripts/infer_policy.py --walking outputs/policy.onnx", False, None, "01-任务资料/04-MicroDuck摆动旋转强化学习复现/assets/microduck_swing_alpha050_local_keyframes.jpg"),
    ("05_球平衡_FastSAC_ONNX_BPU.ipynb", "球平衡 / FastSAC", "microduck-ball-balance", None, "uv run python scripts/infer_policy.py --walking <BALL_BALANCE.onnx> --new-cmd-obs", False, None, "01-任务资料/05-MotrixLab-MicroDuck球平衡与FastSAC/assets/motrix_microduck_ball_balance_local_5000iter_keyframes.jpg"),
    ("06_梯面攀爬_接触与部署模板.ipynb", "梯面攀爬 / 接触课程", "Mjlab-Video-Ladder-Footstep-MicroDuck", None, "uv run python scripts/infer_policy.py --walking <LADDER.onnx> --new-cmd-obs", False, None, "01-任务资料/06-MicroDuck梯面攀爬强化学习导读/assets/microduck_ladder_v2_bootstrap_preview_keyframes.jpg"),
    ("07_RDK网页与多策略_BPU验收.ipynb", "导航 / RDK网页 / 多策略部署", "Mjlab-Velocity-Flat-MicroDuck", None, "uv run python scripts/infer_policy.py --walking <WALKING.onnx> --new-cmd-obs", True, None, "01-任务资料/07-RDK端侧与网页部署/assets/local_videos/microduck_4096env_6000iter_walk_keyframes.jpg"),
]


def build(filename: str, title: str, task_id: str | None, model_hint: str | None, sim_command: str, direct_connect: bool, video_hint: str | None, keyframe_hint: str | None):
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}
    notebook.cells = [
        md(f"# MicroDuck 直播 Notebook：{title}\n\n把任务、策略、ONNX、MuJoCo 和 RDK X5/BPU 验收串成一条可复用流程。"),
        md("## 0. 运行说明\n\n建议在 `02-可运行代码/microduck-playground-stilts` 的 Python 环境中启动 Jupyter。训练模型和板端 HBM 不随 Git 提交；通过 `MICRODUCK_ONNX`、`RDK_BPU_HBM` 和 `RDK_HOST` 注入。"),
        code(SETUP),
    ] + task_cells(model_hint, task_id, sim_command, direct_connect=direct_connect, video_hint=video_hint, keyframe_hint=keyframe_hint)
    nbf.write(notebook, ROOT / filename)


for item in TASKS:
    build(*item)
print(f"generated {len(TASKS)} notebooks in {ROOT}")
