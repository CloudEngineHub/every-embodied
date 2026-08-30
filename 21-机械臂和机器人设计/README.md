# 机械臂与机器人设计

本章从参数化几何建模出发，逐步连接代码生成、机器人描述文件、装配检查和学习驱动的身体设计。读者将理解自然语言需求如何转化为可执行建模代码，以及几何参数如何进一步进入仿真与控制优化。

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

## 学习成果

完成本章后，读者应能区分建模源代码、机器人描述文件和导出网格，能够运行至少一个参数化建模案例，并能说明几何设计、仿真动力学和策略价值之间的关系。
