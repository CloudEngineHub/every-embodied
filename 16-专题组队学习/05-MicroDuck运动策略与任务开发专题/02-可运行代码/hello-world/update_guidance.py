"""Update notebook prose without discarding previously executed cell outputs."""

from pathlib import Path

import nbformat


ENTRY = (
    "第一次运行请先打开 [MuJoCo 策略控制 HelloWorld](./00_MuJoCo策略控制_HelloWorld.ipynb)："
    "自动下载行走模型，用 Python 下发前进、转向和停止命令，完整执行观测、推理、关节目标和物理更新，再进入当前任务。"
)
MODEL_CHECK = """## 4. 检查 ONNX 的输入与动作输出

先确认模型的输入名称、维度和动作输出，为接入仿真准备接口。下方使用零输入做一次模型检查。
随后在本任务闭环单元中读取实时观测，重新计算动作、下发执行器并更新物理状态。
第一次接触这条流程，请先运行 [MuJoCo 策略控制 HelloWorld](./00_MuJoCo策略控制_HelloWorld.ipynb)，
亲手执行前进、转向和停止，并查看 `data.ctrl` 与 `mujoco.mj_step()` 怎样连接策略和仿真。"""
RESULTS = """## 8. 阅读本次运行结果

先查看模型接口，再查看任务闭环与生成的视频，最后对照板端报告。

| 输出标记 | 对应流程 |
| --- | --- |
| `PASS-local-onnx` | 模型结构、输入输出形状和有限值检查通过；随后进入任务闭环 |
| `PASS-GPU-VIDEO` | GPU 策略连接物理仿真，完成本次视频与报告生成 |
| `PASS-BPU-VIDEO` | 当前连接 RDK，逐步取得对应 HBM 的动作，驱动工作站仿真并生成视频 |
| `BPU-CACHED-VIDEO` | 播放之前保存的板端演示；连接 RDK 后重跑板端闭环单元可生成新视频 |
| `PASS-rdk-bpu-runtime-sample` | 开发板的 MobileNet 样例完成推理，运行时检查通过 |
| `PASS-rdk-bpu` | 当前任务 HBM 的板端推理完成，输入输出与耗时保存在输出中 |

任务效果结合位移、速度、稳定时间、接触或登桌指标与视频一起阅读。
修改模型、命令或场景后重新运行对应闭环单元，保存自己的实验记录。"""


def update(topic):
    for path in sorted((Path(topic) / "03-Notebook").glob("0[1-9]_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        changed = False
        first = notebook.cells[1]
        if "00_MuJoCo策略控制_HelloWorld" not in first.source:
            first.source += "\n\n" + ENTRY
            changed = True
        for cell in notebook.cells:
            if cell.cell_type != "markdown":
                continue
            if "这一步只验证 ONNX 图能在本机运行" in cell.source:
                cell.source = MODEL_CHECK
                changed = True
            if cell.source.startswith("## 5. GPU 推理视频") and "climber/get-up" in cell.source:
                cell.source = cell.source.replace("GPU 推理视频", "本任务策略闭环与 GPU 视频", 1)
                cell.source += """

下方调用 `scripts/bpu_official_ladder_pair_video.py` 的 GPU 分支：
每一步读取当前观测，按登桌切换条件选择 climber 或 get-up，推理动作，
再经 `recorder.step(action)` 更新 MuJoCo 执行器、接触与物理状态。
下一步从更新后的状态继续推理，视频在这个循环中逐帧录制。
若先体验 Python 的前进、转向和停止接口，可运行 [HelloWorld](./00_MuJoCo策略控制_HelloWorld.ipynb)。
"""
                changed = True
            if cell.source.startswith("## 8. 直播结论"):
                cell.source = RESULTS
                changed = True
            if "GPU 推理视频" in cell.source and "MuJoCo/MotrixSim" in cell.source:
                heading = cell.source.splitlines()[0].replace("GPU 推理视频", "本任务策略闭环与 GPU 视频")
                cell.source = heading + """

运行下方单元会启动当前任务的策略与仿真器：每个控制周期读取机器人实时观测，
调用当前任务的 actor，取得关节动作，再交给仿真环境的 `env.step(actions)` 处理执行器与物理更新。
下一周期使用更新后的状态重新推理；视频记录这一段完整运行过程。
工作站完成物理计算、渲染与编码，GPU 执行当前任务的策略网络。

脚本位于 `02-可运行代码/microduck-playground-stilts/scripts/`。
行走、扰动、高跷、篮球和摆动使用 `gpu_inloop_video.py`；球平衡使用 `motrix_gpu_inloop_video.py`；
爬梯使用 `bpu_official_ladder_pair_video.py` 的 GPU 分支与爬梯/起身双策略。
下方模型路径需要指向当前任务的权重；各任务教程说明模型下载与环境准备。

若想逐行修改速度命令并直接看到关节目标下发，先运行 [HelloWorld](./00_MuJoCo策略控制_HelloWorld.ipynb)。
RDK 推理放在下一节，接收同一控制流程中的观测并返回动作。
"""
                changed = True
        if changed:
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            print(path.name)


if __name__ == "__main__":
    update(Path(__file__).resolve().parents[2])
