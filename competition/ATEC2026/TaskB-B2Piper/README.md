# ATEC2026 Task B：B2-Piper 垃圾收集

这是一个面向复现者的中文教程和问题复盘。Task B 不是“拿到一个物体坐标，然后走到坐标附近”这么简单；完整闭环应当包含感知、定位、接近、机械臂与夹爪控制、抓取成功确认、搬运、垃圾桶放置和评测状态确认。

## 先看结论

| 问题 | 结论 |
| --- | --- |
| 这是机器狗捡垃圾吗？ | 是。官方任务名称为 Garbage Collection，平台为 B2-Piper。 |
| 是否存在可直接读取的全局环境相机？ | 官方公开观测契约中没有默认的全局环境相机。头部 RGB-D、末端 RGB-D、LiDAR 和 proprioception 才是主要观测来源。 |
| 是否能直接读取物体坐标？ | 正式运行不能依赖物体真值位姿、随机种子或仿真内部对象列表。应从合法传感器观测估计目标位置。 |
| 导航到目标位置就算完成吗？ | 不算。接近、触碰、夹持、提起和放入垃圾桶都需要单独确认。距离足够近可能只是接触代理分数，不能当成物理抓取成功。 |
| 是否可以搬运其他项目？ | 可以搬运状态机、视觉伺服、全身协调、校准和诊断方法；不能直接搬运机器人资产、关节顺序、相机外参或奖励假设。 |
| 当前是否有可核验的完整公开 Task B 冠军实现？ | 本归档没有找到带官方可核验最终分数和权重的完整公开实现。最接近完整流程的是 SLAM 公开 fork，详见 [外部方案对照](./docs/05_外部优秀方案对照.md)。 |

## 推荐阅读顺序

1. [比赛定义与观测边界](./docs/01_比赛定义与观测边界.md)
2. [问题清单与解决方案](./docs/02_问题清单与解决方案.md)
3. [实现路线与阶段状态机](./docs/03_实现路线与阶段状态机.md)
4. [环境相机与调试视角](./docs/04_环境相机与调试视角.md)
5. [外部优秀方案对照](./docs/05_外部优秀方案对照.md)
6. [公开发布与可复现检查](./docs/06_公开发布与可复现检查.md)
7. [中文 Workspace Memory](./中文_Workspace_Memory.md)

## 公开代码

| 文件 | 用途 |
| --- | --- |
| [`code/inspect_task_b_observation.py`](./code/inspect_task_b_observation.py) | 不启动仿真，检查脱敏观测样例的键、类型和形状 |
| [`code/build_public_manifest.py`](./code/build_public_manifest.py) | 为公开目录生成带 SHA-256 的文件清单 |
| [`code/scan_public_release.sh`](./code/scan_public_release.sh) | 检查凭据、机器绝对路径和大文件 |
| [`code/fetch_reference_repos.sh`](./code/fetch_reference_repos.sh) | 拉取外部公开参考仓库到仓库外的缓存目录 |
| [`configs/observation_contract.yaml`](./configs/observation_contract.yaml) | Task B 观测边界和禁止依赖的声明性配置 |

这些脚本是教程和发布工具，不冒充完整得分策略。真正运行比赛代码时，必须使用官方 runner、官方环境和官方评测器。

## 快速检查

```bash
cd /path/to/every-embodied/competition/ATEC2026/TaskB-B2Piper
bash code/scan_public_release.sh .
python3 code/build_public_manifest.py . --output public_manifest.json
```

如需检查一个脱敏 JSON 观测样例：

```bash
python3 code/inspect_task_b_observation.py --input observation_sample.json
```

如需获取公开参考实现，使用仓库外目录：

```bash
bash code/fetch_reference_repos.sh "${TMPDIR:-.reference-cache}"
```

## 评估证据

当前归档优先保留可核验边界，而不是用一段视频或一次近距离分数宣称成功。实验记录见 [`results/experiment_ledger.csv`](./results/experiment_ledger.csv)，证据说明见 [`results/evidence_boundary.md`](./results/evidence_boundary.md)。

## Hugging Face

公开归档说明见 [`hf/README.md`](./hf/README.md)。HF 数据集仓库承载本专题文档、发布清单、经检查的官方公开源码快照、调试媒体和必要的历史资料；不上传私有权重、私有日志和未经授权的外部大文件。完整上传映射与本地删除清单见 [归档对账与本地清理](../docs/05_归档对账与本地清理.md)。
