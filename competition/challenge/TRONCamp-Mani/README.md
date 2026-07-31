# TRONCamp Mani：RoboTwin ACT 竞赛赛后复盘

这是一份真实参赛工程的公开复盘，不是“高分秘诀”，也不把失败的线上成绩包装成成功案例。内容覆盖数据构建、ACT 训练、闭环评测、GPU/环境迁移、提交打包、线上结果和最终失败原因，目标是让后来者少走一轮弯路。

## 结果先说

本队的 T1、T2、T3 顺序门槛均通过，T4 是唯一计分任务，但最终线上 T4 graded 为 **0.2533333333333333（25.3/100）**，未通过预期门槛。

最终有效线上任务只有一个：正确账号尾号 `a40Cbw` 的第一版提交，官方回执为 `#719 T4`。第二版 exact550 e2000 在截止时间后被服务端 `403` 拒绝，没有进入队列，也没有第二个成绩。

这次结果最重要的结论不是“多跑几个 epoch 就一定会好”，而是：

1. 公开 seed 的少量成功不能替代完整、稳定、与线上完全一致的评测。
2. 训练损失下降、checkpoint 能加载、评测能跑完，都不能直接等价为 T4 能通过。
3. 训练成熟度、数据覆盖、动作时间对齐、评测进程拓扑、提交包边界和官方环境必须作为一个整体验收。
4. 比赛截止前应保护一份已通过 T1/T2/T3 的最小可提交候选，同时为 T4 保留可回退的正式评测证据。

## 项目结构

```text
TRONCamp-Mani/
├── README.md                         # 总览、结果、快速复现路线
├── docs/
│   ├── 01_完整工程复盘.md              # 从数据到提交的时间线
│   ├── 02_问题清单与解决方案.md         # 失败模式、根因和修复
│   ├── 03_评测协议与模型选择.md         # 官方口径、门禁和证据边界
│   └── 04_提交与开源检查清单.md         # 提交包、脱敏和发布流程
├── configs/
│   ├── t4_official_act.yaml            # 脱敏后的主线配置
│   └── t4_exact550_manifest.json        # 数据集身份摘要，不含数据
├── code/
│   ├── audit_processed_t4.py            # 16-D / 相机 / 对齐严格审计
│   ├── build_public_manifest.py         # 公开目录 SHA256 清单
│   └── scan_public_release.sh           # 发布前敏感文件与大文件扫描
└── results/
    ├── final_score.json                 # 最终线上结果摘要
    ├── experiment_ledger.csv            # 关键候选与门禁记录
    └── failure_taxonomy.md              # 按阶段归类的失败模式
```

## 任务与官方边界

TRONCamp Mani 使用 RoboTwin / Tron2 双臂操作任务，任务链为：

| 赛道 | 任务 | 线上作用 |
|---|---|---|
| T1 | `adjust_bottle` | 顺序解锁门槛 |
| T2 | `grab_roller` | 顺序解锁门槛 |
| T3 | `stack_bowls_two` | 顺序解锁门槛 |
| T4 | `stack_bowls_three` | 主榜计分，三层末态分级 |

T4 的分数按最终堆叠层数给分，每层约占三分之一；线上私有 seed、官方 simulator、success predicate 和实际门槛由主办方后端决定。公开 seed 只能做本地筛选，不能当作私有榜单的保证。

官方资料：

- [TRONCamp 参赛文档](https://limx-troncamp.github.io/troncamp-web-mani/doc.html)
- [官方提交入口](https://submit.troncamp-mani.limxdynamics.com/)
- [RoboTwin 项目](https://github.com/RoboTwin-Platform/RoboTwin)
- [Every Embodied](https://github.com/datawhalechina/every-embodied)

## 推荐复现路线

公开仓库不包含官方 Tron2 专有资产和大数据。准备好官方授权的 RoboTwin/Tron2 工作区后，按以下顺序复现：

### 1. 先做环境门禁

确认 Python、PyTorch/CUDA、SAPIEN、MPlib、cuRobo、ffmpeg 和官方 ACT 代码版本一致。不要先改动作桥、TCP、相机顺序、success predicate 或 simulator 参数来“让结果变好”。

### 2. 先审计数据，再处理 ACT 数据

每条 HDF5 轨迹至少需要满足：

- 动作和 `qpos` 都是有限值；
- 动作维度为 16，左右臂各 7-D；
- `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三路相机齐全且帧数一致；
- 数据集文件集合与 manifest 完全一致；
- 如果采用 next-state 语义，逐帧验证 `action[t] == qpos[t+1]`；
- 每个 layout 的来源、采集配置、专家 profile 和是否 rescue 都可追溯。

`code/audit_processed_t4.py` 提供不依赖项目路径的严格审计脚本。它只输出摘要和每条轨迹的 SHA，不会自动修复数据。

### 3. 用官方 ACT 做基线

主线配置摘要见 [`configs/t4_official_act.yaml`](configs/t4_official_act.yaml)。重要参数是 ResNet18、4 层 encoder、7 层 decoder、hidden 512、FFN 3200、8 heads、chunk 50、KL 权重 10、batch 8、学习率和 backbone 学习率均为 `1e-5`、seed 120、action shift 0、gripper loss weight 1。

训练时要保存：代码 SHA、原始 manifest SHA、处理审计 SHA、stats SHA、checkpoint SHA、完整 epoch、训练/验证 loss、每 epoch 时间、GPU 型号和显存占用。不要只保存一个“best”文件名而不保存其 lineage。

### 4. 评测要分级

建议固定成：

```text
fresh3  -> 逐条核查层数 / 阶段 / 完整三碗 success
fresh10 -> 仅在 fresh3 通过预设门槛后运行
fresh20 -> 仅在 fresh10 重复稳定后运行
full/public -> 用固定 seed、固定 wrapper、固定进程拓扑
online   -> 使用官方上传包和官方私有评测结果
```

普通 runner、trace/monkey-patch runner、cold process 和 persistent process 的结果不能混成一个“最好成绩”。诊断脚本可以帮助定位阶段，但不能替换正式评测判据。

### 5. 提交前做包边界验收

T2/T3/T4 代码上传时，提交根应是最小的 `external/robotwin_local` 镜像，而不是整个工程根目录、原始数据目录或实验日志目录。代码包不应包含软链接、硬链接、wandb 运行目录、缓存、额外依赖或未使用的备份文件。checkpoint 包只应包含 `policy_last.ckpt` 和同目录的 `dataset_stats.pkl`。

`code/build_public_manifest.py` 只用于公开材料的可追溯清单；正式线上提交仍应使用官方 `submit/submit.py` 和官方 preflight。

## 本次公开的内容与不公开的内容

### 公开

- 数据审计规则、训练配置、评测协议、提交包检查方法；
- 脱敏后的实验表和线上结果；
- 失败模式、根因、修复动作以及哪些“修复”没有证据；
- 可独立运行的公开材料扫描和 SHA256 清单脚本；
- 后续上传 Hugging Face 时应采用的目录与校验约定。

### 不直接放入 Git

- 550 条原始/processed HDF5 和其他大规模数据；
- ACT checkpoint、stats、提交 zip 和视频大文件；
- Tron2 专有机器人 mesh、URDF/MJCF、官方内部资产；
- 私有服务器 IP、端口、PID/PGID、磁盘快照和个人路径；
- access token、SSH 私钥、云平台凭据和浏览器会话；
- 原始运行日志中可能包含的个人信息、内部 URL 或令牌片段。

如后续确认拥有再分发许可，大文件应放在 Datawhale Hugging Face 的独立模型/数据集仓库，并在本目录 README 中补充公开 URL、版本、文件清单和 SHA256。GitHub 只保留轻量代码和文档。

## 许可说明

本目录中的新写文档和通用审计脚本随 Every Embodied 仓库采用仓库的 CC BY 4.0 许可。原官方 RoboTwin、LimX Tron2、cuRobo 和其他第三方内容仍受各自许可证约束；本目录不重新授权官方专有资产，也不暗示主办方对本复盘背书。
