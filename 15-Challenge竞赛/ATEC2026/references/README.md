# 外部方案索引

本目录不复制第三方仓库的完整代码，而是记录上游链接、固定 commit、许可证和迁移边界。这样可以保留作者署名和版本可追溯性，也避免把不同机器人、不同仿真版本的代码误当成 Datawhale 的可直接提交方案。

## Logic-TARS

- 上游仓库：[Logic-TARS/ATEC2026](https://github.com/Logic-TARS/ATEC2026)
- 固定 commit：`b78c4afd1b84302fe8f88bcfd287eac64c33692c`
- 许可证：MIT，以上游仓库 `LICENSE` 为准
- 重点文件：`README.zh-CN.md`、`submission/solution.py`、`scripts/train/`、`scripts/export/`
- 适用方向：Task D B2W 推箱越障
- 可迁移内容：61D/16D 接口、16D→24D adapter、阶段状态机、LiDAR 箱体检测、航向修正、卡死恢复
- 不可直接假定：checkpoint 可用、评分阈值永远不变、B2Piper 与 B2wPiper 接口相同

## ZSN2024

- 上游仓库：[ZSN2024/ATEC2026_Simulation_Challenge](https://github.com/ZSN2024/ATEC2026_Simulation_Challenge)
- 固定 commit：`ee4e0eb97928754d9404a3acd5d644020ac7794c`
- 适用方向：Task B B2wPiper Stage1 训练、导出和 adapter
- 注意：统一 demo 中的垃圾收集部分仍属于基础启发式控制，不是已证明的高分闭环

## yma867

- 上游仓库：[yma867/ATEC2026_Simulation_Challenge_RIL](https://github.com/yma867/ATEC2026_Simulation_Challenge_RIL)
- 固定 commit：`e56a2a9e39c5231a91c0a8b1cce8ab1bc0e72403`
- 重点目录：`taskb_perception/`
- 适用方向：Task B YOLO、ByteTrack、RGB-D 深度反投影和世界坐标
- 注意：其操作入口明确保留 zero-action placeholder，需要另行实现底盘与机械臂控制
