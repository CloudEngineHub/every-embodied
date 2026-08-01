# SLAM ATEC2026 公开参考方案

本目录不复制外部仓库的大型源码，只保存归因和复现入口。原始实现由原作者维护，请以原仓库最新内容和许可证为准。

## 原始链接

- [GitHub 仓库](https://github.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge)
- [Task B demo](https://github.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge/tree/main/demo)
- [Task B 文档](https://github.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge/tree/main/doc)
- [Task B EE lock 文档](https://raw.githubusercontent.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge/main/doc/task_b_ee_lock_pipeline.md)
- [Task B object classifier 文档](https://raw.githubusercontent.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge/main/doc/task_b_object_classifier.md)

## 获取

```bash
bash ../../code/fetch_reference_repos.sh "${TMPDIR:-$HOME/.cache/atec2026-reference}"
```

## 归因

该公开仓库当前包含 MIT 风格 LICENSE。本文只引用其链接和公开设计，不将其完整代码、模型或数据集重新打包到 Datawhale 归档。若后续复制任何实质代码，应同时保留原版权声明和许可证全文。

## 复现时的重点

1. 先读 `task_b_ee_lock_pipeline.md`，理解头部相机搜索到末端相机锁定的交接。
2. 再读 `task_b_object_classifier.md`，区分离线标注和运行时合法 RGB 输入。
3. 检查 `solution_task_b_nav_only.py` 的默认路线是否只是触碰/服务式处理。
4. 用官方评测器单独核验抓取、提起和放置，不把公开脚本的存在当成最终分数证明。
