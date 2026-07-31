# ATEC2026 Task B 中文 Workspace Memory

更新时间：2026-07-31

## 目的

本文件是后续继续维护 Task B 公开教程时的事实记忆和工作边界。它不保存私有机器路径、凭据、模型权重或原始比赛日志。

## 已确认事实

- Task B 是 Garbage Collection，平台是 B2-Piper。
- 任务闭环不止导航，还包括目标搜索、接近、机械臂和夹爪操作、抓取验证、搬运、垃圾桶放置和官方评测确认。
- 官方公开观测主要包含 proprioception、LiDAR、头部 RGB-D 和末端 RGB-D。
- 没有证据表明一个全局第三方环境相机属于默认正式 policy observation。
- 环境相机可以做本地录像、全局排障和调试对齐，但不能提供运行时目标坐标。
- 物体真值位姿、对象 ID、仿真内部对象列表和评测器内部状态只能用于 debug-only 检查。
- 接近、触碰和接触代理指标不能替代物理夹持、提起和放置验证。

## 最值得参考的公开方案

- SLAM fork：[cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge](https://github.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge)。公开了 Task B 导航、末端相机锁定、机械臂校准、分类器和训练文档，是当前最接近完整 Task B 结构的公开方案。
- Clear fork：[StevenLiudw/Clear_ATEC2026_Simulation_Challenge](https://github.com/StevenLiudw/Clear_ATEC2026_Simulation_Challenge)。更适合作为 B2-Piper locomotion、下蹲和动作映射参考。
- Lhy fork：[Lhy6900/ATEC2026_Simulation_Challenge](https://github.com/Lhy6900/ATEC2026_Simulation_Challenge)。重点是 G1/GR00T adapter，不能直接替换 B2-Piper Task B 控制接口。
- ZSN2024 和 yma867 的链接已经在上一级 [公开方案索引](../references/README.md) 中登记，分别用于训练骨架和 RGB-D 感知参考。

## 不能写成结论的内容

- 公开 demo 的存在不等于官方最终成绩。
- 自报排名不等于官方排行榜证明。
- 视频里“碰到了”不等于物体被夹住。
- 环境相机看到了物体不等于策略合法地看到了物体。
- 其他机器人的关节顺序、相机外参、动作维度和奖励阈值不能直接迁移。

## 后续维护顺序

1. 以官方仓库当前 checkout 重新核对 observation key、action space、runner 和评分器。
2. 在单环境下做合法观测 smoke test，再做头部搜索和末端锁定。
3. 单独验证抓取后的相对位姿、提起高度和垃圾桶放置判定。
4. 每次实验保存 commit、seed、环境版本、官方分数、视频和失败阶段。
5. 新增外部代码前先检查 LICENSE；优先链接和固定版本，不直接复制大仓库。
6. 提交前运行 `code/scan_public_release.sh` 和 `code/build_public_manifest.py`。

## 发布边界

GitHub 放教程、轻量脚本、配置、结果边界和外部链接；Datawhale HF 放与教程一致的小型公开归档。官方资产、数据、checkpoint、原始视频和私有日志只有在明确获得再分发许可后才能进入公开仓库。

## 2026-07-31 公开归档与清理状态

- Datawhale HF：`Datawhale/atec2026-task-b-reproducibility`，公开分支已包含 Task B 文档、脚本、媒体清单、5 个视频、4 张截图和官方源码快照。
- 官方源码快照：`source/ATEC2026_Simulation_Challenge_20260518.zip`，大小 `431182199` bytes，SHA-256 为 `84ccfdc3903e4e03a5de8a7dedd90314b15ed09382c136bbeee6a858dae802d9`。
- 视频和截图均为脱敏 debug/evaluation preview；其中 top-down/global 画面使用 privileged trace，只用于解释行为，不是正式 policy 输入或官方成功证明。
- 官方选手指南归档在 HF `official_docs/`；历史 Task E XSA 插件归档在 HF `historical_task_e/`。
- HF 归档可复现公开教程、源码快照和证据阅读；完整仿真仍需官方匹配版本的 Isaac Sim/Isaac Lab、GPU 驱动和 Python 环境，不应写成“只靠下载 HF 就能直接运行”。
- 原始 ATEC 工作目录、官方压缩包、指南副本、XSA 副本和本地 Task B 视频在远端 SHA/大小核验通过后删除；原始 Workspace Memory 不上传，保留本文件作为脱敏替代。
