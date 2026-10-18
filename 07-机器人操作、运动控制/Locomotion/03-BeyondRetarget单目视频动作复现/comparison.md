# BeyondRetarget 与两阶段重定向：怎么比、优势在哪里

> 更新：2026-10-18。本页的数值和对比演示来自作者论文及项目页；本教程尚未重新运行四条基线。我们已跑通的内容仍是[六秒在线推理](./results/validation.md)。下面将论文结果解读与同视频复跑流程分别说明。

## 1. 先分清“换机器人”和“换方法”

同一动作驱动八款机器人，展示的是跨本体适配能力。要比较算法，先固定为同一款 **Unitree G1、29 个关节**，让各条流水线接收同一段视频：

| 方法 | 视频到机器人动作的路径 | 主要比较点 |
| --- | --- | --- |
| GVHMR → GMR | 视频 → 人体运动估计 → 优化式重定向 | 人体估计误差如何影响几何匹配 |
| GVHMR → NMR | 同一份 GVHMR 人体估计 → 学习式重定向 | 固定人体估计，只换重定向方法 |
| WHAM → GMR | 视频 → 另一种人体运动估计 → 优化式重定向 | 固定重定向方法，只换人体估计 |
| WHAM → NMR | 同一份 WHAM 人体估计 → 学习式重定向 | 检查结果是否依赖某一种人体估计器 |
| BeyondRetarget | 视频特征 → 共享运动表示 → G1 解码与后处理 | 直接生成机器人轨迹的整体效果 |

这里的 [GVHMR](https://github.com/zju3dv/GVHMR) 和 [WHAM](https://github.com/yohanshin/WHAM) 负责恢复人体运动；[GMR](https://github.com/YanjieZe/GMR) 和 [NMR](https://github.com/NJU3DV-HumanoidGroup/MakeTrackingEasy) 负责把人体动作转成机器人动作。不能拿 GVHMR 单独输出的人体动画，与 BeyondRetarget 的机器人动画当成完整方法对比。

这个二乘二设计有助于定位问题：换人体估计器后改善，提示上游重建可能是瓶颈；输入人体估计保持不变而换重定向器后改善，提示映射方式有影响。但这些观察仍受模型、配置和后处理等因素影响，不能据单个片段直接作因果结论。

## 2. 先看作者已经公布的比较

### 2.1 定量结果：同一 G1 上比较

以下摘录自 [BeyondRetarget 论文 v1 表 1](https://arxiv.org/html/2609.29850v1#S4.T1)，**不是本教程重新测得的成绩**。MAMPJPE 为形态对齐后的关键点误差，RTE 为根轨迹误差，FS 为支撑帧的水平脚滑；前三项越小越好。执行成功率使用作者的 SONIC 跟踪评测。

| RGB 输入的方法 | MAMPJPE ↓（mm） | RTE ↓（mm） | FS ↓（mm/帧） | 成功片段 / 总片段 ↑ |
| --- | ---: | ---: | ---: | ---: |
| GVHMR → NMR | 39.19 | 701.13 | 2.28 | 340 / 362 |
| GVHMR → GMR | 39.46 | 637.33 | 3.30 | 313 / 362 |
| WHAM → NMR | 43.15 | 849.36 | 6.64 | 342 / 362 |
| WHAM → GMR | 45.88 | 217.76 | 13.30 | 295 / 362 |
| **BeyondRetarget** | **26.36** | **80.74** | **1.71** | **347 / 362** |

在这四条 RGB 基线中，最低 MAMPJPE 为 39.19 mm；BeyondRetarget 的 26.36 mm 比它低约 **32.7%**，计算为 `(39.19 - 26.36) / 39.19`。执行成功率从最好的 342/362 到 347/362，即 **94.48% → 95.86%，增加约 1.38 个百分点**。它们比较的是不同指标各自的最佳基线，不能把所有提升都归给同一个对手。

论文还设置了两条 **GT 人体运动 → 重定向**的诊断基线，用于绕过人体估计误差；它们有额外真值信息，不进入 RGB 方法排名：

| 使用人体真值的诊断方法 | MAMPJPE ↓（mm） | RTE ↓（mm） | FS ↓（mm/帧） | 成功片段 / 总片段 ↑ |
| --- | ---: | ---: | ---: | ---: |
| GT → NMR | 33.26 | 837.62 | 1.30 | 346 / 362 |
| GT → GMR | 32.02 | 36.01 | 0.82 | 347 / 362 |

因此应把结论限定为：**在作者的 RGB 输入评测中，BeyondRetarget 的这几项结果优于四条两阶段基线；它没有在全部设置和指标上胜出。** 例如 GT → GMR 的根轨迹误差和脚滑更低，执行成功数与 BeyondRetarget 相同。较低的形态对齐姿态误差，也不能被解释为其从视频恢复了比人体真值更准确的人体动作。

### 2.2 官方演示：带着问题看

打开以下作者提供的同步对比视频；这些是**官方定性演示，不是我们这次生成的基线结果**：

- [叉腰与摆臂](https://bear-ty.github.io/Beyondretarget_page/assets/videos/main-result-waist-arm-swing.mp4)：看肘部、手部是否保持对应位置，大幅运动时是否出现姿态跳变。
- [双手放在背后](https://bear-ty.github.io/Beyondretarget_page/assets/videos/main-result-behind-back.mp4)：看手究竟位于躯干前还是躯干后；只看正面轮廓容易漏掉深度错误。
- [大幅绕臂](https://bear-ty.github.io/Beyondretarget_page/assets/videos/main-result-large-arm-circle.mp4)：看肩肘朝向与运动连续性，而非仅看机器人是否“在动”。
- [举手与下蹲](https://bear-ty.github.io/Beyondretarget_page/assets/videos/main-result-hand-raise-squat-five-methods.mp4)：同时观察上肢、腰部与双脚，避免只挑手臂好看的一帧。

链接与说明来自[官方项目页](https://bear-ty.github.io/Beyondretarget_page/)。这些是作者挑选的案例，能解释失效形式，不能替代完整测试集的统计结果。我们随章的六秒低分辨率输入动作较简单，也不适合单凭观感突出某种方法的优势。

## 3. 为什么直接预测机器人动作可能更好

两阶段路线先拟合人体，再把这个结果交给重定向器。第二阶段通常看不到原始图像：如果上游把背后的手估到了身前，后续几何匹配可能忠实地复现这个错误。即使人体姿态正确，人体与机器人的肢段比例、关节结构和可达空间也不同，仍可能出现映射误差。

BeyondRetarget 把最终机器人输出作为学习目标，使视觉表示可以服务于目标本体；机器人专属解码头与接触修正再处理形态和支撑问题。这提供了减少跨阶段误差积累的机制解释，但“少一个中间表示”本身并不保证优于任何两阶段方法。优势仍需在相同任务、输入与计算条件下检验。

接触后处理也有取舍。论文表 3 中，去掉 refinement 的 MAMPJPE 为 25.73 mm，完整模型为 26.36 mm；同时 FS 从 4.51 降至 1.71 mm/帧，接触违例率从 22.18% 降至 6.17%。这说明修正会牺牲一些姿态贴合，换取更稳定的接触，不能把“关节更像”与“更容易执行”视为同一目标。[来源：论文消融表](https://arxiv.org/html/2609.29850v1#S4.T3)

## 4. 自己重跑：先接通 GVHMR → GMR

这是最适合先补的本地基线，因为 GMR 已提供读取 GVHMR 输出的正式入口。下面是按上游源码核对的步骤，**本教程尚未执行该完整流程**。

### 4.1 独立安装并记录版本

分别按 [GVHMR 安装文档](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md) 和 [GMR 安装文档](https://github.com/YanjieZe/GMR#installation) 准备独立环境、推理权重与授权的人体模型。它们不是 BeyondRetarget 环境里默认已有的依赖。GVHMR 的人体模型和 GMR 要求的文件扩展名、目录可能不同，应依上游说明配置。

本页核查的版本如下；这只是命令核查版本，并非论文作者当时使用的全部基线版本：

| 项目 | 核查提交 |
| --- | --- |
| GVHMR | `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` |
| GMR | `bb1bbe40774794fceb2a7c579a3464a28e68c844` |
| NMR | `ba87ad6541f6512df9a370acdd2c2d95da7ae419` |

对每个项目，在其仓库根目录使用 `git checkout <对应提交>` 固定版本，并保存 `git rev-parse HEAD`、环境依赖列表、权重哈希和模型哈希。上游版本会变化，不应在展示结果时只写“最新版”。

### 4.2 使用完全相同的视频

在 GVHMR 项目根目录、激活其环境后执行。`-s` 表示静态相机，下面假定输入也确实来自固定机位：

```bash
python tools/demo/demo.py --video=/absolute/path/person_6s.mp4 -s
```

应得到 `outputs/demo/person_6s/hmr4d_results.pt`。切换到 GMR 项目根目录并激活其环境后：

```bash
python scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file /absolute/path/GVHMR/outputs/demo/person_6s/hmr4d_results.pt \
  --robot unitree_g1 \
  --save_path /absolute/path/runs/comparison/gvhmr_gmr_g1.pkl \
  --record_video
```

此入口会打开查看器，需有可用图形环境。核查版本的视频路径由脚本生成在 `videos/` 下，并没有独立的 `--video_path` 参数；批量实验每次都应在运行后归档对应视频，避免同名输出覆盖。若要测处理速度，不要开启人为限速的 `--rate_limit`。[来源：GMR 的 GVHMR 入口](https://github.com/YanjieZe/GMR/blob/bb1bbe40774794fceb2a7c579a3464a28e68c844/scripts/gvhmr_to_robot.py)

### 4.3 两个会让比较失真的导出细节

在上述 GMR 版本的 `gvhmr_to_robot.py` 中：

- MuJoCo 中 `qpos` 使用 wxyz，但保存到 PKL 的 `root_rot` 转成了 **xyzw**。BeyondRetarget 的 `root_rot_quat` 为 **wxyz**，适配时应显式调整，不能只改字段名。GMR 的关节顺序也需要按模型核对。
- 主循环先将 `i` 从 0 加到 1，再处理该帧，因而跳过第 0 帧。正式比较应修正循环为处理全部帧并记录改动，或依据真实时间戳明确标记缺失的首帧；不能把少一帧的输出从时间 0 开始播放后直接对齐。也不要用补零、循环播放或隐式截短掩盖缺失。

这些是入口实现细节，不能据此推断论文的 GMR 评测也使用了相同脚本或存在相同问题。本章的 `inspect_motion.py` 只支持说明过的 NPZ/ZIP 结构，**不能直接读取 GMR 的 PKL**；导出适配器还应验证单位、关节名、四元数和时间戳，之后再用于共同查看器。

## 5. 如何扩展另外三条基线

| 扩展路线 | 可以复用什么 | 仍需确认什么 |
| --- | --- | --- |
| GVHMR → NMR | 同一份 GVHMR 人体估计 | 将世界系人体参数转换为 NMR 接受的坐标、姿态与帧率 |
| WHAM → GMR | 同一原视频，以及 GMR 的目标机器人 | WHAM 人体表示与 GMR SMPL-X 输入之间的适配 |
| WHAM → NMR | 同一份 WHAM 人体估计 | SMPL/SMPL-X 语义、关节维度、坐标及时间轴适配 |

[NMR 官方入口](https://github.com/NJU3DV-HumanoidGroup/MakeTrackingEasy)接受人体运动文件，不直接接受 RGB 视频。核查日官方 Space 的 `/predict` 也要求动作文件。用任意 AMASS 示例跑通 NMR，只能验证重定向器，不能称为“同一视频的 GVHMR → NMR 对比”。

对已经正确转换的输入，在 NMR 环境中执行：

```bash
python inference.py --src /absolute/path/person_motion.npz --output-dir /absolute/path/runs/comparison/nmr
```

NMR 文档区分 Z-up 的 AMASS 字段 `trans/root_orient/pose_body` 与 Y-up 的标准字段 `transl/global_orient/body_pose`；输出 bmimic NPZ 为 **50 FPS**。不能仅靠重命名键来做坐标转换，也不能把 50 FPS 输出逐帧拼到 30 FPS 视频上。适配验证后，应按共同的秒级时间轴重采样，四元数使用合适的旋转插值，并保留原始输出。[来源：NMR 输入输出说明](https://github.com/NJU3DV-HumanoidGroup/MakeTrackingEasy/blob/ba87ad6541f6512df9a370acdd2c2d95da7ae419/README.md)

WHAM 的官方视频入口是 `python demo.py --video /absolute/path/person_6s.mp4 --visualize`，但生成了人体结果，并不等于已经接上 GMR/NMR。本页未验证 WHAM 适配器，也不提供虚构的 `wham_to_robot.py` 一键命令。[来源：WHAM 官方仓库](https://github.com/yohanshin/WHAM)

## 6. 对比视频与指标怎样才公平

先做“原视频 + 五种方法的同款 G1”同步画面；再用同一控制器做动力学跟踪。八机器人展示应另作适配演示，不与 G1 方法排名混在一起。

| 要固定的条件 | 具体要求 |
| --- | --- |
| 输入 | 同一原文件及哈希、同一裁剪、分辨率、起止时间和帧率；失败片段保留 |
| 本体 | 相同 G1 关节集合、模型尺度和限制；检查不同仓库模型的质量、碰撞与关节排列 |
| 时间与坐标 | 保留时间戳；统一米/弧度、世界轴、根朝向和四元数顺序，记录重采样及固定对齐变换 |
| 画面 | 相同视角、相机距离、地面和播放速度；保留完整动作，可加侧面视角看深度 |
| 后处理 | 记录每条方法的原始和推荐后处理；额外增加的统一滤波另列结果 |
| 物理执行 | 同一个跟踪控制器、权重、步长、增益、初始状态及终止判据；报告失败数 |
| 耗时 | 同硬件分别记录检测、人体恢复、重定向/直接预测、后处理与渲染；网络排队单独列 |

对齐时不能给每一帧任意平移或旋转来消除根轨迹错误；需要形态/刚体对齐的指标，应严格遵守其定义，并同时保留能显示全局漂移的回放。

没有动作真值时，可以做明确标注的定性比较、关节限制检查或共同定义的代理指标；不能凭输入视频直接算出论文的 MAMPJPE/RTE。脚滑也应使用相同的支撑标注或预先固定的共同判定规则，不能让每种方法自行删掉“不好”的支撑帧。静止输出可能很平滑，低抖动必须和动作保真度一起看。

现有六秒输入只有 **180 帧**，小于论文 Fail 指标的 200 帧窗口，也不足论文物理执行评测的 600 帧片段。在线演示还限制最多 10 秒，不能把这个六秒示例直接套用成论文测试集的失败率或成功率。

## 7. 适合教程的结论

目前能讲清楚的优势是：作者在同一 G1 视觉输入任务上的误差与执行结果更好；直接面向机器人学习，为减少人体估计到重定向的误差传递提供了一条路线。接触修正有助于改善物理合理性，但存在精度与稳定性的取舍。

下一步可先完成同视频的 GVHMR → GMR，再补学习式重定向和另一种人体估计器。届时将本地成绩与论文表分开，保留优势、平局和失败案例。八本体支持列表也应以各项目的固定版本为准；例如 **Galaxea R1 Pro 与 Unitree R1 是不同机器人**，不能因名称都含 R1 就视为共同支持。

[返回复现教程](./README.md)
