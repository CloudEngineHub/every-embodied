# 失败分类

| 类别 | 证据 | 影响 | 后续动作 |
|---|---|---|---|
| 训练成熟度 | exact550 正式 e1250/e1500/e1750 均未完成三碗 | 不能证明 6000 epoch 目标已达到 | 预留完整计算窗口，分段落盘并评测 |
| 数据分布 | 550 条中 rescue=257；clean50 与 exact550表现不同 | 可能存在 profile / pose / joint coverage 偏差 | 做 stock / retry / rescue 单变量 paired ablation |
| 时间对齐 | processed audit 的 next-state max error=0，但 loader 修改会改变目标 | 错位会让 loss 和闭环都失真 | 固定官方语义，单独命名 alignment 实验 |
| 评测拓扑 | 普通 runner 与 trace runner 对同权重给出不同结果 | 小样本分数不可直接比较 | 最终使用 persistent official runner |
| public/private gap | 本地控制 fresh3=2/3，线上 T4=25.3/100 | public 3 seed 不能代表私有 100 seed | 增加固定、未看过的 seed panel |
| 资源可用性 | 远程 4090 欠费关机；PRO 6000 有磁盘/GPU 归属约束 | 训练和同步窗口被压缩 | 持久盘、余额告警、可恢复 checkpoint |
| 提交流程 | 错账号 403；第二版截止后 403 | 只有一份有效线上 T4 | 提交前检查解锁状态并保存 queue id |
| 发布边界 | 私有日志含路径、进程、凭据和大文件 | 直接复制会泄露秘密或超大仓库 | 只发布脱敏摘要和可复现脚本 |
