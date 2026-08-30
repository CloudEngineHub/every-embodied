# 机械臂和机器人设计

本目录整理机器人机械结构、代码建模、CAD 生成、机器人描述文件和学习驱动机器人设计相关教程。建议大家先从参数化建模入门，再进入面向机器人项目的 Text-to-CAD、ForgeCAD、URDF/SDF 工作流和价值梯度机器人设计。

## 教程列表

- [Build123d 代码建模入门](01Build123d代码建模入门/README.md) - 用 Python/build123d 生成 STEP、STL 和预览图，理解代码 CAD 的基础语法。
- [Text-to-CAD 工程化建模入门](02Text-to-CAD工程化建模入门/README.md) - 用代码智能体、CAD skill 和 CAD Explorer 串起源文件、派生文件和几何检查。
- [ForgeCAD 官方 3D 打印机、键盘与灵巧手案例复现](03ForgeCAD视觉逆向工程入门/README.md) - 对齐 ForgeCAD public kit 的 `3dprinter-gpt52codex` benchmark，保存官方 GIF，并复刻 3D 打印机、视频键盘和可动灵巧手的参数化装配、渲染 GIF 与 STEP/STL/3MF 导出。
- [Shape Your Body 价值梯度机器人设计导读](04-Shape-Your-Body价值梯度机器人设计导读/README.md) - 讲解如何训练多具身 policy/value function，并用冻结 critic 的价值梯度优化连续机器人身体参数。

## 推荐学习顺序

1. 先学习 Build123d，理解参数、基元、布尔操作和文件导出。
2. 再学习 Text-to-CAD，理解代码智能体如何参与 CAD 源文件修改和几何检查。
3. 然后学习 ForgeCAD，体验 JavaScript/TypeScript 生态下的 code-first CAD、官方 benchmark 复刻和复杂装配展示流程。
4. 最后阅读 Shape Your Body，理解 CAD/URDF/MJCF 参数如何进一步进入“身体参数 -> 仿真控制 -> 价值梯度 -> 设计诊断”的学习型机器人设计链路。
