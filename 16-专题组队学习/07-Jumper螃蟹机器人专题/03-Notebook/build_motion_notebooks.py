"""Build the Jumper jump and dance teaching notebooks from their evidence files.

Run this generator from any directory; execute the resulting notebooks from their
own directory so relative media and evidence links resolve. Existing notebooks
are overwritten, so rerun them after regenerating to save their outputs.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
M = nbf.v4.new_markdown_cell
C = nbf.v4.new_code_cell

SETUP = """## 运行前提

本文记录的上游代码来自 [KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper)，提交 `b2e58669848a3ba4f209597cd6bebdfe9d0533a4`。完整环境检查和旧实验见 [Ubuntu 工作站复现记录](../01-学习资料/Ubuntu工作站复现记录.md)。下方代码单元读取已经保存的结果并显示视频，不会自动安装依赖或启动训练。

工作站代码在 `/home/ubuntu/projects/robotics_shared/repos/jumper-everyembodied`，Python 为 `/data/Data14TB/envs/jumper-everyembodied/bin/python`。下方命令在仓库根目录执行；`warp:cuda` 用于 GPU 训练，`native:cpu` 和 OSMesa 用于无窗口录像。训练日志、checkpoint、视频和评估数据来自不同步骤，应在报告中分别记录。具体版本和结果以本专题保存的证据为准。
"""

LOAD = """import json
from pathlib import Path
from IPython.display import Video, display

evidence_dir = Path('../01-学习资料')
media_dir = Path('../assets')
"""

def result_cell(filename):
    return C(f'''result_path = evidence_dir / {filename!r}
if result_path.is_file():
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print("尚未找到实际评估数据:", result_path)
''')

def video_cell(filename):
    return C(f'''video_path = media_dir / {filename!r}
if video_path.is_file():
    display(Video(str(video_path), embed=True, width=640))
else:
    print("尚未找到实际视频:", video_path)
''')

books = {
    "06_跳跃_参考动作训练与评估.ipynb": [
        M("# 06｜跳跃：参考动作、强化学习与评估\n\n用上游已公开的跳跃参考数据与 PPO 任务，在仿真中训练和回放跳跃。结果及视频读取本专题保存的实测文件；没有运行的数据不会被写成成果。"),
        M(SETUP),
        M("## 1. 开源内容和动作机制\n\n[训练任务与奖励](https://github.com/KingKongRobotics/jumper/tree/b2e58669848a3ba4f209597cd6bebdfe9d0533a4/tasks/jumper/jump) 使用仓库内 `tasks/jumper/jump/ref/high_jump_flat.npz`。策略输出是在参考关节动作上的残差；学习目标包含参考关节、机身高度、速度及接触跟踪。上游 PPO 网络与超参数见该目录的 `rl_cfg.py`。参考轨迹不等于当前训练出的策略；不要把参考动画或官方动作包当作这次 checkpoint 的结果。"),
        M("## 2. 列出任务并训练\n\n```bash\ncd /home/ubuntu/projects/robotics_shared/repos/jumper-everyembodied\nPY=/data/Data14TB/envs/jumper-everyembodied/bin/python\n\"$PY\" scripts/train.py --task jumper.jump --dry-run\nOMP_NUM_THREADS=4 \"$PY\" scripts/train.py --task jumper.jump --backend warp --device cuda:0 --num_envs 4096 --max-iterations 1000 --headless --no-tensorboard\n```\n\n以训练实际日志为准记录源码 commit、环境、任务、后端、环境数、迭代、随机种子和 checkpoint。上游默认训练可能超过此处示例的 1000 次；短训用于验证学习与运行链路，不自动表示收敛。"),
        M("## 3. 阅读训练过程\n\n除了总奖励，关注 `Episode_Reward/ref_base_height`、`ref_joint_pos`、`ref_contact` 以及 `Metrics/jump/apex_rise`、`flight_s`、`takeoff_vz`。训练日志中的批量统计与单次视频指标是不同测量。需要检查机器人实际离地、落地及失败终止，才能描述回放效果。"),
        M("## 4. 回放和录像\n\n将 `CKPT` 替换成实际文件，注意保留所用版本。`--physics-hz 200` 对齐训练物理步频；观测噪声默认开启。`--stop-on-done` 使终止后的自动重置不会被误认为一次连续动作。\n\n```bash\nCKPT=logs/jumper/jumper.jump/<实际运行目录>/model_<迭代>.pt\nMUJOCO_GL=osmesa OMP_NUM_THREADS=2 \"$PY\" scripts/play.py --task jumper.jump --backend native --device cpu --checkpoint \"$CKPT\" --headless --steps 500 --physics-hz 200 --stop-on-done --measure --video jump.mp4 --speed 0\n```\n\n回放应记录实际时长、是否因终止提前结束、跳跃高度和空中时间的测量口径；如果没有可用统计，只描述视频及日志中可确认的行为。"),
        M("## 5. 本专题实测数据"), C(LOAD), result_cell("jump-result.json"),
        M("## 6. 自训策略视频\n\n下方视频和评估文件应对应同一 checkpoint 与明确的回放条件。"), video_cell("jump-trained.mp4"),
        M("## 延伸阅读\n\n- [上游教程：三种技能训练与回放](https://github.com/KingKongRobotics/jumper/blob/b2e58669848a3ba4f209597cd6bebdfe9d0533a4/docs/TUTORIAL.md)\n- [专题首页](../README.md) · [三足步态训练](03_动作训练与评估.ipynb) · [舞蹈任务](07_舞蹈_参考动作训练与评估.ipynb)\n\n数据和源码的第三方许可见上游 `NOTICE`。仿真效果不直接代表真机效果。"),
    ],
    "07_舞蹈_参考动作训练与评估.ipynb": [
        M("# 07｜舞蹈：动作参考、策略训练与仿真回放\n\n使用上游开放的舞蹈轨迹和训练代码，学习动作跟踪与训练评估；下方视频为本专题本次运行保存的结果。"),
        M(SETUP),
        M("## 1. 已公开的内容\n\n[舞蹈任务](https://github.com/KingKongRobotics/jumper/tree/b2e58669848a3ba4f209597cd6bebdfe9d0533a4/tasks/jumper/dance) 包含参考动作 `media/demo.npz`、任务配置与 PPO 训练代码。首次构建时会上游转换和缓存参考数据。仓库中的 `demo.mp4` 是脸部动画素材，不是本次自训机器人视频。训练不需要音乐；输出配乐成片依赖上游未提供的音乐，这里仅使用无声仿真回放。"),
        M("## 2. PPO 跟踪参考动作\n\n策略基于机器人状态和参考动作信息输出关节控制；奖励衡量对参考运动的跟踪及运动质量。上游使用动作跟踪 runner，记录位置、姿态等指标；对舞蹈而言，能否持续完成参考片段和是否出现终止也应结合视频判断。模型、参考文件和代码版本应一同记录。"),
        M("## 3. 在工作站训练\n\n```bash\ncd /home/ubuntu/projects/robotics_shared/repos/jumper-everyembodied\nPY=/data/Data14TB/envs/jumper-everyembodied/bin/python\n\"$PY\" scripts/train.py --task jumper.dance --dry-run\nOMP_NUM_THREADS=4 \"$PY\" scripts/train.py --task jumper.dance --backend warp --device cuda:0 --num_envs 4096 --max-iterations 1000 --headless --no-tensorboard\n```\n\n该命令是可调整迭代数的工作站入口；实际使用的命令和结果应以本专题的训练日志及 JSON 为准。上游配置默认 10000 次迭代。"),
        M("## 4. 检查与录像\n\n观察训练日志里的参考动作跟踪、位置和方向误差以及终止信息；高训练奖励不等于完整表演。录像前确认 checkpoint、参考素材和任务一致，用有限步数与 `--stop-on-done` 保留失败终止。\n\n```bash\nCKPT=logs/jumper/jumper.dance/<实际运行目录>/model_<迭代>.pt\nMUJOCO_GL=osmesa OMP_NUM_THREADS=2 \"$PY\" scripts/play.py --task jumper.dance --backend native --device cpu --checkpoint \"$CKPT\" --headless --steps 1000 --physics-hz 200 --stop-on-done --measure --video dance.mp4 --speed 0\n```\n\n确认视频实际时长和是否提前终止。无音乐不妨碍检查关节动作；播放过的官方动作包应明确与自训模型区分。"),
        M("## 5. 本专题实测数据"), C(LOAD), result_cell("dance-result.json"),
        M("## 6. 自训策略视频\n\n视频和评估文件需标注同一 checkpoint、轨迹与回放条件。"), video_cell("dance-trained.mp4"),
        M("## 延伸阅读\n\n- [舞蹈参考数据说明](https://github.com/KingKongRobotics/jumper/blob/b2e58669848a3ba4f209597cd6bebdfe9d0533a4/tasks/jumper/dance/media/README.md)\n- [上游动作包训练教程](https://github.com/KingKongRobotics/jumper/blob/b2e58669848a3ba4f209597cd6bebdfe9d0533a4/docs/TUTORIAL.md)\n- [专题首页](../README.md) · [跳跃任务](06_跳跃_参考动作训练与评估.ipynb) · [官方动作包演示](05_官方动作包_跳跃与舞蹈.ipynb)\n\n第三方素材许可见上游 `NOTICE`；仿真结果与真机执行分别记录。"),
    ],
}

jump = books["06_跳跃_参考动作训练与评估.ipynb"]
jump[3].source += "\n\n本次计划 1000 次，但工作站与其他任务共享 GPU，在保存 model_300 后主动中止；评估使用 model_300，上游默认 10000 次。"
jump[5].source = """## 4. 回放、录像与上游评估

跳跃训练物理频率 400 Hz、控制频率 200 Hz；回放使用相同物理频率。观测噪声默认开启，`--stop-on-done` 避免将重置后的画面误认为连续动作。

```bash
CKPT=logs/jumper/jumper.jump/2026-10-29_14-48-03/model_300.pt
MUJOCO_GL=osmesa OMP_NUM_THREADS=1 "$PY" scripts/play.py --task jumper.jump --backend native --device cpu --checkpoint "$CKPT" --headless --steps 700 --physics-hz 400 --stop-on-done --measure --video jump.mp4 --video-width 640 --video-height 480 --speed 0
"$PY" tools/checks/jump_eval.py --ckpt "$CKPT" --num-envs 1 --episodes 3
```

上游评估脚本关闭观测噪声，以 1 N 接触力阈值判断全足离地，报告高度与累计腾空时间；它在自动重置后采样，且把一次 done 波计为一个 episode，本例用单环境解释其输出。训练批量指标与单次回放指标不同。
"""
dance = books["07_舞蹈_参考动作训练与评估.ipynb"]
dance[4].source = dance[4].source.replace('OMP_NUM_THREADS=4', 'OMP_NUM_THREADS=2').replace('--num_envs 4096 --max-iterations 1000', '--num_envs 1024 --max-iterations 500')
dance[4].source += "\n\n本次与其他 GPU 任务共享工作站，采用 1024 环境，改变了 PPO 批量；上游 GPU 示例为 4096 环境。"
dance[5].source = dance[5].source.replace('logs/jumper/jumper.dance/<实际运行目录>/model_<迭代>.pt', 'logs/jumper/jumper.dance/2026-10-29_14-54-46/model_200.pt').replace('OMP_NUM_THREADS=2', 'OMP_NUM_THREADS=1').replace('--video dance.mp4 --speed 0', '--video dance.mp4 --video-width 640 --video-height 480 --speed 0')
dance[5].source += "\n\n上游参考舞蹈约 235.7 秒；本片 20 秒、无配乐，属于开头片段。第 50 和 100 次 checkpoint 的回放在第 2 和 3 步终止，第 200 次的一次回放连续运行 1000 步。训练日志中的关节位置误差是关节向量 L2 范数（rad），不是逐关节平均或 RMS。"
dance.extend([
    M("## 7. 500 次训练后的最终模型\n\n使用最终 `model_499.pt` 重新回放，同样在 200 Hz 物理、50 Hz 控制及观测噪声开启时完成 1000 步、20 秒，没有提前终止。训练日志中的第 499 次关节位置误差 L2 为 0.3505 rad，刚体平均位置误差为 0.0123 m；这些是训练批量指标，不是录像回放误差。"),
    result_cell("dance-final-result.json"),
    video_cell("dance-final.mp4"),
    M("[完整训练与评估记录](../01-学习资料/跳跃与舞蹈实跑记录.md) · [训练日志](../01-学习资料/dance-training.txt) · [最终回放日志](../01-学习资料/dance-final-replay.txt) · [最终 checkpoint](../02-可运行代码/checkpoints/dance-model-499.pt)\n\n最终模型重放命令与第 200 次一致，将 `CKPT` 改为 `logs/jumper/jumper.dance/2026-10-29_14-54-46/model_499.pt`。本次使用上游开源参考与算法；没有公开的配乐不包含在成果中。"),
])

for name, cells in books.items():
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbf.validate(notebook)
    nbf.write(notebook, HERE / name)
