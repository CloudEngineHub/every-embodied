# BladeMaster 与 DexLab：运行柔体切割仿真

本章使用 DexLab 的 CPU 切割实现，运行豆腐状软块对半切和香蕉连续三刀，检查独立评分并生成回放。DexLab 参考了 BladeMaster 的 TLMPM 和持久切割历史，材料参数为合成设置，尚未标定真实食材。

## 1. 先看项目演示

- [豆腐对半切：表面视频](https://github.com/huangkiki/Dexlab/blob/main/docs/evidence/cutting/tofu/surface.mp4) · [粒子视频](https://github.com/huangkiki/Dexlab/blob/main/docs/evidence/cutting/tofu/particles.mp4) · [实验说明](https://github.com/huangkiki/Dexlab/blob/main/docs/site/zh/cutting.md)。下刀、撤刀后，施加诊断冲量检查两块材料的独立运动。
- [香蕉连续三刀：表面视频](https://github.com/huangkiki/Dexlab/blob/main/docs/evidence/cutting/banana/surface.mp4) · [粒子视频](https://github.com/huangkiki/Dexlab/blob/main/docs/evidence/cutting/banana/particles.mp4) · [实验说明](https://github.com/huangkiki/Dexlab/blob/main/docs/site/zh/banana-cutting.md)。三次切割后，独立评分器检查是否形成四个连通分量。
- [BladeMaster 项目页](https://jango6324.github.io/blademaster/)展示原论文的机器人切割、推移和交叉切割演示；论文实现采用 GPU，DexLab 是独立的 CPU 实现。

## 2. Windows 实测：从仿真到视频

本次在 Windows 11、AMD Ryzen 7 5800H 上使用现有 Python 3.11 环境运行 DexLab 切割代码；求解在 CPU 上进行。项目完整安装声明 Python 3.12，本次使用 `PYTHONPATH` 直接加载切割模块，未安装完整的 DexLab 任务依赖。实测版本：NumPy 1.26.4、SciPy 1.11.4、Numba 0.64.0、Matplotlib 3.10.6、imageio 2.37.0、imageio-ffmpeg 0.6.0、VTK 9.6.0、Pillow 10.4.0。

下面在 PowerShell 中从源码创建环境并运行；需要 Git 和 Conda，预留下载依赖的空间。`$Tutorial` 指向本仓库本章目录，按实际位置修改。每次实验输出与渲染目录都必须是新路径。

```powershell
conda create -n dexlab-cutting python=3.11 -y
conda activate dexlab-cutting
python -m pip install numpy==1.26.4 scipy==1.11.4 numba==0.64.0 matplotlib==3.10.6 imageio==2.37.0 imageio-ffmpeg==0.6.0 vtk==9.6.0 pillow==10.4.0
git clone https://github.com/huangkiki/Dexlab.git
cd Dexlab
git checkout c7d958fe8bc1a2eb51429c95ae02f729f9176ef5
$env:PYTHONPATH = Join-Path (Get-Location) "src"
$Tutorial = "C:\path\to\every-embodied\10-具身智能其他仿真工具及仿真前沿\13-BladeMaster与DexLab柔体切割"
$Out = Join-Path $HOME "dexlab-cutting-output"
New-Item -ItemType Directory -Force $Out | Out-Null
python -c "import numpy, scipy, numba, imageio, imageio_ffmpeg, vtk; print(vtk.vtkVersion.GetVTKVersion(), hasattr(vtk, 'vtkWin32OpenGLRenderWindow'))"
```

最后一行应输出 VTK 版本和 `True`。以下两次仿真顺序执行；程序按模拟时间输出进度，结束时给出评分。然后单独运行评分器再次检查：

```powershell
$Tofu = Join-Path $Out "tofu-run-01"
$Banana = Join-Path $Out "banana-run-01"
python -m dexlab.tofu_cut --output $Tofu
python -m dexlab.cutting_score $Tofu
python -m dexlab.tofu_cut --task banana --output $Banana
python -m dexlab.cutting_score $Banana
```

确认两次评分 `passed` 为 `true`，豆腐最终为 2 个分量、香蕉最终为 4 个分量。`manifest.json`、`states.npz`、`trace.npz` 和 `score.json` 保存在各自输出目录。

官方渲染器使用 EGL；Windows VTK 提供 `vtkWin32OpenGLRenderWindow`。本章 [windows_render.py](windows_render.py) 在启动渲染器前将窗口类适配为 Windows 的离屏窗口，不改仿真状态和评分。对两份记录分别渲染：

```powershell
python (Join-Path $Tutorial "windows_render.py") $Tofu --output (Join-Path $Out "tofu-media-01")
python (Join-Path $Tutorial "windows_render.py") $Banana --output (Join-Path $Out "banana-media-01") --stride 20 --fps 5
```

香蕉采用每 20 个保存帧采样一次、5 fps 输出以减少渲染耗时，仍覆盖完整 8 秒；豆腐使用默认每 5 帧采样、20 fps。每个媒体目录包含 `surface.mp4`、`particles.mp4`、对应 GIF、海报图和测量文件。渲染是逐帧的，运行结束后再播放 MP4；如果输出目录已存在，请改用新目录。此 Windows 流程直接运行，不使用下节 Linux 的 systemd 资源限制与共享锁；在共享机器上应自行安排运行窗口。

### 本机结果与回放

本机豆腐实验的独立评分通过，最终为两个连通分量，各占质量 50%，相对位移约 3.221 mm；切后运动包含诊断冲量。香蕉默认 `cut-push` 实验也通过独立评分，三刀后分量依次为 2、3、4，刀面推移检查通过，受推段相对位移约 10.209 mm。以下视频由本次仿真状态和 Windows 离屏适配生成。

[豆腐表面视频](assets/tofu-surface.mp4) · [豆腐粒子视频](assets/tofu-particles.mp4) · [豆腐评分结果](assets/tofu-score.json)

[香蕉表面视频](assets/banana-surface.mp4) · [香蕉粒子视频](assets/banana-particles.mp4) · [香蕉评分结果](assets/banana-score.json)

## 3. Linux / WSL 官方环境

以下命令面向 Linux 或 WSL，使用 Python 3.12、Git，以及能够访问 PyPI 的网络。官方有界执行器需要运行 systemd、cgroup 限额和无需交互的 sudo；WSL 需先启用 systemd。仿真使用 CPU；生成表面回放还需安装 `cutting-render` 依赖。确保有足够磁盘空间，首次安装包括 PyTorch 等依赖。所有命令在 DexLab 仓库根目录执行：

```bash
git clone https://github.com/huangkiki/Dexlab.git
cd Dexlab
git checkout c7d958fe8bc1a2eb51429c95ae02f729f9176ef5
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cutting,cutting-render]'
```

这个提交固定了本章对应的源码；若需使用后续版本，请同时核对其官方实验说明。安装后运行 `python -c "import dexlab, numpy, scipy, numba, vtk; print('dependencies OK')"`，出现 `dependencies OK` 表示这些依赖可以导入。若找不到 `python3.12`，先安装 Python 3.12 及其 venv 支持；若渲染导入失败，检查 `vtk` 的安装输出。

## 4. Linux：运行豆腐对半切

DexLab 提供有界执行器和共享研究锁。先指定本机输出目录、该目录所在的块设备，以及共享锁路径；`/dev/sda` 仅是示例，应通过 `findmnt -T "$CUT_ROOT"` 和 `lsblk` 查明实际设备。所有共用同一研究资源的进程应使用同一个锁路径；每次 `RUN`、资源计划和收据要使用新的路径。先确认 `systemctl is-system-running` 能连接 systemd，且 `sudo -n true` 成功。下面采用项目实验页的 4 核、600 秒设置：

```bash
CUT_ROOT="$HOME/dexlab-cutting"
mkdir -p "$CUT_ROOT"
findmnt -T "$CUT_ROOT"
lsblk
IO_DEVICE=/dev/sda                    # 改为实际块设备
LOCK="$CUT_ROOT/research.lock"
RUN="$CUT_ROOT/tofu-full-01"          # 每次换一个新路径
python scripts/bounded_run.py --profile adaptive \
  --resource-plan "$CUT_ROOT/tofu-plan.json" --cpu-cores 4 --io-device "$IO_DEVICE" \
  --data-dir "$CUT_ROOT" --timeout 600 --receipt "$CUT_ROOT/tofu-resources.json" -- \
  python scripts/research_guard.py run --lock "$LOCK" \
  --kind qualification --receipt "$CUT_ROOT/tofu-window.json" -- \
  .venv/bin/python -m dexlab.tofu_cut --output "$RUN"
python -m dexlab.cutting_score "$RUN"
```

项目记录中的豆腐基线包含 3 秒模拟过程：下刀、撤刀，然后向一侧施加诊断冲量。运行完成后查看 `$RUN/manifest.json`、`states.npz`、`trace.npz` 和 `score.json`。以独立评分器退出状态及 `score.json` 为准：完整切割应形成两个连通分量，并满足项目定义的位移、接触和数值检查。运行器成功退出只表明记录过程结束。本机使用现有 Windows Python 环境直接运行豆腐求解器和独立评分，得到两个分量且评分通过；下述 Linux 有界执行链尚未在本机验证。

## 5. Linux：运行香蕉连续三刀

继续在同一个环境和仓库根目录下，使用新的输出目录。该任务比豆腐耗时更长；实际耗时取决于机器和任务版本。

```bash
RUN="$CUT_ROOT/banana-full-01"
python scripts/bounded_run.py --profile adaptive \
  --resource-plan "$CUT_ROOT/banana-plan.json" --cpu-cores 4 --io-device "$IO_DEVICE" \
  --data-dir "$CUT_ROOT" --timeout 600 --receipt "$CUT_ROOT/banana-resources.json" -- \
  python scripts/research_guard.py run --lock "$LOCK" \
  --kind qualification --receipt "$CUT_ROOT/banana-window.json" -- \
  .venv/bin/python -m dexlab.tofu_cut --task banana --output "$RUN"
python -m dexlab.cutting_score "$RUN"
```

固定版本中 `--task banana` 默认采用 `cut-push`：三刀后继续用刀面推移，完整模拟时长 8 秒。若要对应上方历史香蕉实验的诊断冲量版本，可额外指定 `--motion-protocol impulse`，该流程为 7 秒。检查评分中的切割分量及任务结果。项目还提供 `--mode uncut` 和 `--mode partial` 对照，用于检查未切开或第三刀浅切时评分是否拒绝完整四段任务。每个对照都要使用新的输出目录。

## 6. Linux：生成并查看回放

以下以刚完成的香蕉记录为例。渲染读取已有物理状态，不重新运行仿真；官方建议在单独的受限后处理窗口执行。当前源码使用 `vtkEGLRenderWindow`，需要带 EGL 支持的 VTK 和可用的 EGL 图形环境。可以先运行 `python -c "import vtk; print(hasattr(vtk, 'vtkEGLRenderWindow'))"` 检查；若输出 `False`，需安装支持 EGL 的 VTK 环境后再渲染。Windows 用户使用第 2 节的窗口适配。渲染结束后在输出目录查找视频和图片，与第 1 节官方粒子视频对照：

```bash
MEDIA="$CUT_ROOT/banana-media-01"
python scripts/bounded_run.py --profile adaptive \
  --resource-plan "$CUT_ROOT/render-plan.json" --cpu-cores 4 --io-device "$IO_DEVICE" \
  --data-dir "$CUT_ROOT" --timeout 600 --receipt "$CUT_ROOT/render-resources.json" -- \
  python scripts/research_guard.py run --lock "$LOCK" \
  --kind qualification --receipt "$CUT_ROOT/render-window.json" -- \
  .venv/bin/python -m dexlab.cutting_showcase "$RUN" --output "$MEDIA"
find "$MEDIA" -type f
```

颜色表示切割历史；判断材料是否真的分离还要看评分器重建的连接关系和切后运动。豆腐实验的时间步敏感性结果中反力峰值变化明显，尚未建立切割力收敛；它适合学习数值机制，不宜直接作为真实食材切割力。

## 7. 切口如何保留下来

刀具经过材料时，求解器为粒子记录切割侧别；粒子与网格传递动量时，切面两侧不再发生不相容的耦合。这样撤刀后切口仍能保持，同时材料接触允许切面再次接触和滑动。刀具与材料的双向作用让材料反力影响刀具运动。

```mermaid
flowchart LR
    A[刀具运动和材料粒子] --> B[记录切割侧别]
    B --> C[按相容关系传递动量]
    C --> D[材料接触和刀具反力]
    D --> E[保存状态并独立评分]
```

参考：[BladeMaster 论文](https://arxiv.org/abs/2609.27342)、[DexLab 仓库](https://github.com/huangkiki/Dexlab)、[豆腐实验与完整资源配置](https://github.com/huangkiki/Dexlab/blob/main/docs/site/zh/cutting.md)、[香蕉实验](https://github.com/huangkiki/Dexlab/blob/main/docs/site/zh/banana-cutting.md)。
