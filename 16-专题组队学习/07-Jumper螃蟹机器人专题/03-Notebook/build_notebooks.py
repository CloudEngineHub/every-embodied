"""Generate the Jumper learning notebooks using nbformat."""
from pathlib import Path
import nbformat as n

HERE = Path(__file__).resolve().parent
M = n.v4.new_markdown_cell
C = n.v4.new_code_cell

common = """## 使用方式
按顺序阅读并执行代码单元。默认单元只检查本机和演示原理，不会安装依赖、下载仓库或开始训练。下面的命令是供配置完成后手动执行的步骤；上游有更新时以对应版本文档为准。
"""
books = {
"01_环境检查与项目地图.ipynb": [
M("# 01｜环境检查与项目地图\n\n目标：了解 Jumper 学习路线，记录本机环境和上游版本。"), M(common),
M("## 1. 检查环境\n\nPython 和 Git 用于阅读与运行工具；GPU 检查仅报告当前环境，具体训练依赖以项目指南为准。"),
C("import sys, platform, shutil, subprocess\nprint('Python:', sys.version.split()[0])\nprint('系统:', platform.platform())\nfor name in ('git', 'nvidia-smi'):\n    path = shutil.which(name)\n    print(name, path or '未找到')\n    if name == 'nvidia-smi' and path:\n        p = subprocess.run([path, '--query-gpu=name,driver_version', '--format=csv,noheader'], capture_output=True, text=True)\n        print(p.stdout.strip() or p.stderr.strip())"),
M("## 2. 准备上游仓库\n\n选择自己的工作目录后，在终端执行：\n\n```bash\ngit clone https://github.com/KingKongRobotics/jumper.git\ngit clone https://github.com/KingKongRobotics/jumper-design.git\ncd jumper\ngit rev-parse HEAD\n```\n\n依照 [项目指南](https://github.com/KingKongRobotics/jumper/blob/main/docs/PROJECT_GUIDE.md) 创建隔离环境、安装依赖。指南列出 Python 3.10–3.13，GPU 版 PyTorch 与 CPU 原生后端分别配置。不要把此处环境检查当作安装完成。"),
M("## 3. 项目地图\n\n- `jumper`：模型、任务、训练、回放、打包。\n- `jumper-design`：外观、场景及内容包。\n- [教程](https://github.com/KingKongRobotics/jumper/blob/main/docs/TUTORIAL.md) 将训练、回放、打包连接起来。\n\n**检查**：记录系统、Python、GPU、上游 commit；接着打开 02。" )],
"02_模型与六足步态.ipynb": [
M("# 02｜模型与六足步态\n\n目标：理解关节、接触、观测与动作之间的关系，并阅读上游机器人模型。"), M(common),
M("## 1. 六足与 22 自由度\n\nJumper 有六条腿，整机 22 个自由度。自由度总数不等于腿数；阅读模型时记录每个关节名称、运动范围、执行器、身体和足端，不要假设所有关节都参与行走。参见 [上游仓库](https://github.com/KingKongRobotics/jumper)。"),
M("## 2. 三足步态的概念模型\n\n下面仅演示两组交替的抽象相位，A/B 是占位组别，不代表 Jumper 实际的腿部命名、触地时间或训练策略。实际分组应查看上游任务实现及回放。"),
C("for step in range(8):\n    support = 'A' if step % 2 == 0 else 'B'\n    swing = 'B' if support == 'A' else 'A'\n    print(f'相位 {step}: 支撑组 {support}, 摆动组 {swing}')"),
M("## 3. 源码阅读任务\n\n在本地 Jumper 仓库查看 `assets/jumper` 和 `tasks/jumper`，找到模型、任务配置与动作/观测定义；文件名与组织结构可能随版本变化。对照 [项目指南](https://github.com/KingKongRobotics/jumper/blob/main/docs/PROJECT_GUIDE.md) 和 [训练教程](https://github.com/KingKongRobotics/jumper/blob/main/docs/TUTORIAL.md)。\n\n填写：关节顺序、观测组成、动作维度、控制周期、足端接触和终止条件分别在哪里定义？这些信息应来自所用版本的代码。\n\n**检查**：画出“状态 → 观测 → 策略 → 关节目标 → 物理仿真 → 状态”的闭环，说明支撑组切换可能影响稳定性的原因。")],
"03_动作训练与评估.ipynb": [
M("# 03｜动作训练、回放与评估\n\n目标：从公开三足步态入手，理解训练、checkpoint、回放与评估的区别。"), M(common),
M("## 1. 训练过程\n\n策略接收观测并输出动作；物理仿真产生下一状态，奖励和终止条件决定学习目标。三足步态任务重点关注速度指令跟踪。跳跃和舞蹈任务涉及参考动作，具体定义请阅读[上游教程](https://github.com/KingKongRobotics/jumper/blob/main/docs/TUTORIAL.md)。"),
M("## 2. 配置完成后在 Jumper 根目录运行\n\n```bash\npython scripts/train.py --list\npython scripts/train.py --task jumper.tripod\npython scripts/play.py --task jumper.tripod\n```\n\n命令来自 [项目指南](https://github.com/KingKongRobotics/jumper/blob/main/docs/PROJECT_GUIDE.md)。先确认环境与显卡配置；训练可能耗时。回放默认选择的 checkpoint 和可用参数应以当前版本文档为准，记录具体文件与 commit。上游也提供 Colab 训练入口。"),
M("## 3. 设计可复核评估\n\n固定 checkpoint、任务、场景、种子集合、回合时长和成功判据；记录跌倒、速度跟踪误差及视频。多组实验比较时保持评估条件一致。以下是演示如何统计，不是实际实验结果。"),
C("demo_success = [True, False, True, True, False]\nprint('示例成功率:', sum(demo_success), '/', len(demo_success), '=', sum(demo_success)/len(demo_success))"),
M("## 4. 从回放到部署\n\n阅读上游教程的动作包部分，理解策略、控制器及运行平台的对应关系；先在仿真验证，再按上游指南进行设备部署。\n\n**检查**：保存命令、环境、checkpoint、视频和评估结果到 [实验记录模板](../01-学习资料/实验记录模板.md)。短训成功启动只说明链路可运行，不能代替性能评估。")],
"04_外观场景与作品.ipynb": [
M("# 04｜外观、场景与专题作品\n\n目标：理解 `.skin` 和 `.map` 的用途，设计可分享、可检查的作品。"), M(common),
M("## 1. 创建内容\n\n[Jumper Design](https://github.com/KingKongRobotics/jumper-design) 提供外观和场景的工作流、示例、格式定义与校验工具。外观导出为 `.skin`，场景导出为 `.map`；新几何可能需要外部建模工具。按照该仓库的快速开始操作，记录版本和生成文件。"),
M("## 2. 设计一个作品\n\n选择一个目标：为跳跳设计沙漠游侠外观，或设计一个有障碍和目标点的室内场景。先写主题、使用素材、交互设想、评价方式；随后检查包结构并在支持的查看器中观察效果。记录外部素材来源和许可证。"),
C("idea = {'主题': '螃蟹的客厅', '任务': '绕过障碍到达目标点', '评估': '到达次数、碰撞与回放视频'}\nfor key, value in idea.items():\n    print(f'{key}: {value}')"),
M("## 3. 创作与训练的边界\n\n根据 [项目指南](https://github.com/KingKongRobotics/jumper/blob/main/docs/PROJECT_GUIDE.md)，皮肤当前用于展示，导入场景用于回放而非训练；确认所用版本是否改变。外观展示成功不代表实体外壳可装配，场景回放成功也不代表策略已经在该场景训练。\n\n**检查**：保存内容包、上游版本、来源、校验结果和截图；小组评审可讨论视觉效果、交互设计及下一步实现。")]
}
books["05_官方动作包_跳跃与舞蹈.ipynb"] = [
    M("# 05｜上游动作包：跳跃与螃蟹舞\n\n目标：查看现成控制器如何切换动作，理解训练策略与上游预训练动作包的区别。以下视频在 Ubuntu 工作站以 native CPU 和 OSMesa 实际录制。素材使用上游提交 `b2e58669848a3ba4f209597cd6bebdfe9d0533a4` 自带 `out/bundle_example/jumper.app`，该包 SHA-256 为 `454762cb9018a3ef91e29ef06dd350a44e7c934941a25cf289dec1cd2bcebc0b`。它不是本专题自训模型。"),
    M("## 跳跃演示\n\n仿真 2.00 秒注入 Space，控制器从 locomotion 切换到 jump，3.79 秒返回 locomotion。画面可见离地。"),
    C("from IPython.display import Video, display\ndisplay(Video('../assets/official-jump.mp4', embed=True, width=480))"),
    M("## 螃蟹舞演示\n\n仿真 2.00 秒注入 Ctrl+1，控制器切换到 dance_crab；画面后段可见挥爪。"),
    C("display(Video('../assets/official-dance.mp4', embed=True, width=480))"),
    M("## 在工作站重放\n\n脚本使用上游内部 `AppPlayer._person` 接口模拟按键，绑定上述提交，接口随上游版本可能变化。将 [跳跃脚本](../02-可运行代码/official_jump_demo.py) 上传到工作站 `/tmp/official_jump_demo.py`，再从 Jumper 源码目录运行：\n\n```bash\ncd /data/Data14TB/robotics_shared/repos/jumper-everyembodied\nMUJOCO_GL=osmesa OMP_NUM_THREADS=1 /data/Data14TB/envs/jumper-everyembodied/bin/python /tmp/official_jump_demo.py\n```\n\n脚本输出 `/tmp/jumper-official-jump.mp4`；[舞蹈脚本](../02-可运行代码/official_dance_demo.py) 同法运行，输出 `/tmp/jumper-official-dance.mp4`。两段视频各 9.03 秒。查看 [跳跃日志](../01-学习资料/official-jump.txt) 与 [舞蹈日志](../01-学习资料/official-dance.txt)，寻找模式切换时间。上游包内容与许可见 [Jumper](https://github.com/KingKongRobotics/jumper)。")
]
books["05_官方动作包_跳跃与舞蹈.ipynb"][1:1] = [
    M("## 行走演示：前进指令与实际位移\n\n这是上游预训练动作包的独立回放。在第 2–9 秒按住 W，指令逐渐增至 0.8 m/s；固定镜头便于观察位移，跟随镜头便于观察腿部。两段是独立回放，不能把两条轨迹合并。"),
    C("from IPython.display import Video, display\ndisplay(Video('../assets/official-walk.mp4', embed=True, width=480))\ndisplay(Video('../assets/official-walk-follow.mp4', embed=True, width=480))"),
    M("### 用采样轨迹计算位移\n\n脚本每 0.05 秒采样一次机身世界坐标；取最接近 2 秒和 9 秒的点，计算水平直线位移及时间平均位移速度。该速度不是逐时速度均值。"),
    C("import csv, math\nfrom pathlib import Path\nfor filename in ('official-walk.csv', 'official-walk-follow.csv'):\n    with (Path('../01-学习资料') / filename).open(encoding='utf-8', newline='') as f:\n        rows = list(csv.DictReader(f))\n    a, b = [min(rows, key=lambda r: abs(float(r['time_s']) - t)) for t in (2, 9)]\n    dist = math.hypot(float(b['world_x_m']) - float(a['world_x_m']), float(b['world_y_m']) - float(a['world_y_m']))\n    duration = float(b['time_s']) - float(a['time_s'])\n    print(f'{filename}: {duration:.2f} s, {dist:.3f} m, {dist/duration:.3f} m/s')"),
    M("[固定镜头脚本](../02-可运行代码/official_walk_demo.py) · [跟随镜头脚本](../02-可运行代码/official_walk_follow_demo.py) · [固定镜头数据](../01-学习资料/official-walk.csv) · [跟随镜头数据](../01-学习资料/official-walk-follow.csv)\n\n脚本在上游源码根目录运行，使用内部键盘注入接口；输出写入 `/tmp`。此演示使用上游动作包，不是自训策略。"),
]
books["03_动作训练与评估.ipynb"].insert(1, M("**实跑结果**：Ubuntu 工作站 GPU 从零训练 1000 次迭代，得到会前进的三足策略。最终模型在 0.3 m/s 指令下，两次 10 秒回放的平均前向速度为 0.2780 和 0.2759 m/s，均无重置。视频、代码、checkpoint 和评估数据见文末第 7 节；第 6 节保留一次没有走起来的续训对照。"))
books["03_动作训练与评估.ipynb"].extend([
    M("## 5. Ubuntu 工作站实跑：300 次 CPU 三足训练\n\n2026-10-29，Jumper 提交 `b2e58669848a3ba4f209597cd6bebdfe9d0533a4`；`native:cpu`、64 环境、4 物理线程，300 次迭代、460,800 环境步，日志所报训练时间 4 分 25 秒。最终 mean reward -5.29，线速度跟踪奖励 0.3666，课程等级 0；这些是训练日志指标，不是独立成功率。\n\n[完整日志](../01-学习资料/tripod-training.txt) · [详细复现记录](../01-学习资料/Ubuntu工作站复现记录.md) · [零动作基线视频](../assets/zero-baseline.mp4) · [训练后视频](../assets/tripod-299.mp4)"),
    M("### 回放与测量\n\n用最终 `model_299.pt`，固定机体前向指令 0.1 m/s，500 控制步、10 秒，观测噪声开启；单次回放没有 episode reset。平均前向速度约 -0.0013 m/s，平均绝对误差约 0.1013 m/s。该策略在此条件下基本没有跟踪前进指令，需要继续训练和评估。回放物理 1000 Hz、控制 50 Hz；仅是单次测量。"),
    C("from IPython.display import Video, display\ndisplay(Video('../assets/tripod-299.mp4', embed=True, width=480))"),
    C("import json\nfrom pathlib import Path\nevaluation = json.loads(Path('../01-学习资料/evaluation.json').read_text(encoding='utf-8'))\nfor key in ('command', 'control_steps', 'simulated_seconds', 'episode_resets', 'actual_velocity_mean', 'mean_absolute_error', 'observation_noise'):\n    print(f'{key}: {evaluation[key]}')"),
    M("![关节与接触测量](../assets/measure.png)\n\n[测量 CSV](../01-学习资料/measure.csv) · [评估 JSON](../01-学习资料/evaluation.json) · [回放配置](../01-学习资料/replay-info.json)\n\n训练和回放的精确命令、checkpoint 路径与哈希见复现记录。视频呈现一次仿真轨迹，不能替代多种子评估。"),
    M("## 6. GPU 续训与实际排查\n\n从 CPU `model_299.pt` 恢复，使用 Warp CUDA、4096 个环境再训练 1000 次迭代；日志记录 98,304,000 个环境步、11 分 29 秒。最后 mean reward 28.79，课程等级 1。\n\n对最终 `model_1298.pt` 以 0.3 m/s、200 Hz 物理、50 Hz 控制、观测噪声开启回放 10 秒：没有 reset，但平均前向速度 -0.00080 m/s，前向平均绝对误差 0.30080 m/s，策略仍未有效前进。"),
    C("display(Video('../assets/tripod-gpu-final.mp4', embed=True, width=480))"),
    C("gpu_result = json.loads(Path('../01-学习资料/gpu-final-evaluation.json').read_text(encoding='utf-8'))\nfor key in ('command', 'physics_hz', 'control_steps', 'episode_resets', 'actual_velocity_mean', 'mean_absolute_error', 'observation_noise'):\n    print(f'{key}: {gpu_result[key]}')"),
    M("中间 checkpoint 在 200/1000 Hz、CPU/GPU 后端下均接近原地；命令管理器检查显示它收到 0.3 m/s，但不能据此认定整个观测链都没有问题。[复现记录与诊断](../01-学习资料/Ubuntu工作站复现记录.md)包含命令、数据和 checkpoint 哈希。[官方预训练动作包演示](05_官方动作包_跳跃与舞蹈.ipynb)展示实际行走、跳跃和舞蹈，来源与本次训练不同。"),
    M("## 7. GPU 从零训练：自训三足策略行走\n\n保持上游任务配置，从随机初始化以 Warp CUDA、4096 环境训练。`model_200.pt` 在 CPU 原生后端、200 Hz 物理、50 Hz 控制、观测噪声开启的 10 秒回放中，收到 0.3 m/s 前向指令，平均前向速度 **0.2724 m/s**，没有 reset。录视频的另一次独立回放为 0.2878 m/s；1000 Hz 物理回放为 0.2793 m/s。回放未固定种子，结果随运行变化。"),
    C("display(Video('../assets/tripod-fresh200.mp4', embed=True, width=480))"),
    C("for name in ('fresh200-eval.json', 'fresh200-video-evaluation.json', 'fresh200-cpu1000-03.json'):\n    result = json.loads((Path('../01-学习资料') / name).read_text(encoding='utf-8'))\n    print(name, 'physics Hz:', result['physics_hz'], 'vx:', round(result['actual_velocity_mean'][0], 4), 'resets:', result['episode_resets'])"),
    M("### 1000 次迭代后的最终模型\n\n完成 1000 次迭代，共 98,304,000 环境步。下方视频和数据来自最终 `model_999.pt`，使用 CPU 原生后端、200 Hz 物理、观测噪声开启；每次回放 10 秒。"),
    C("display(Video('../assets/tripod-fresh999.mp4', embed=True, width=480))"),
    C("for name in ('fresh999-repeat.json', 'fresh999-video.json', 'fresh999-fast.json'):\n    result = json.loads((Path('../01-学习资料') / name).read_text(encoding='utf-8'))\n    print(name, 'target:', result['command'][0], 'vx:', round(result['actual_velocity_mean'][0], 4), 'MAE:', round(result['mean_absolute_error'][0], 4), 'resets:', result['episode_resets'])"),
    M("![最终策略关节与接触测量](../assets/fresh999-measure.png)\n\n[训练与回放命令、checkpoint 哈希](../01-学习资料/Ubuntu工作站复现记录.md) · [最终自训 checkpoint](../02-可运行代码/checkpoints/tripod-gpu-fresh999.pt) · [测量 CSV](../01-学习资料/fresh999-measure.csv)。多次回放没有固定种子，结果说明策略在这些指定仿真条件下能行走；真机效果尚未测量。"),
])
for filename, cells in books.items():
    book = n.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}})
    n.validate(book)
    n.write(book, HERE / filename)
