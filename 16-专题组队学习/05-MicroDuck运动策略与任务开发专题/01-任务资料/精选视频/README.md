# MicroDuck 技能串联精选视频

当前文件夹保存一版可公开使用的阶段组合演示：

- `microduck_skill_suite_v0.mp4`：统一为 1280×720、30 FPS 的组合视频；
- `clips/`：章节片段和构建时使用的障碍绕行源视频；
- `clips/source_ladder_v3_model_1100.mp4`：工作站最新 V3 checkpoint 的 9 秒审计回放，当前仍停在梯脚附近，未并入主片；
- `build_skill_suite.ps1`：在 Windows 上重新生成组合视频的脚本。

## 片段说明

1. **绕障行走**：使用 Ubuntu 工作站生成的 MicroDuck 障碍物行走回放，是真实策略在 MuJoCo 障碍几何中的运行结果。
2. **物理扰动**：mjswan 浏览器端拖拽演示，外力来自 MuJoCo 物理引擎，不把人工拖拽写成策略已经学会的恢复动作。
3. **球面平衡**：MotrixLab/FastSAC 球面平衡回放。
4. **摆动与接触**：MicroDuck 摆动任务回放，用于展示动力学接触和姿态变化。
5. **梯面接触**：当前梯面任务的阶段性物理回放。它证明场景和接触链路已经运行，但不代表连续换档或顶部落台已经完成。最新 V3 `model_1100.pt` 的独立评测仍是首个物理横档状态，`MAX_PHYSICAL_RUNG_TARGET=1`，所以没有把它剪成“成功爬梯”。

## 当前边界

绕障、拾取和梯面攀爬不能只靠把多个 policy 名称串起来就算完成。当前 `microduck-rl-lab` 的五技能 pipeline 已完成环境 smoke；GroundPick 的完整 checkpoint、连续梯面 V3 以及顶部落台仍在训练和验收中。本片把已验证结果与训练中内容分开标注，后续拿到完整策略后再替换对应章节。

重新生成：

```powershell
Set-Location 'C:\Users\kewei\Documents\2025\04资料整理\03具身教程编写\every-embodied\07-机器人操作、运动控制\Locomotion\精选视频'
powershell -ExecutionPolicy Bypass -File .\build_skill_suite.ps1
```
