# ATEC2026 赛后复盘 Workspace Memory

更新时间：2026-07-31

## 工作边界

- GitHub 公开入口：`competition/ATEC2026/`。
- 原始 ATEC 工程、训练缓存、环境和大 checkpoint 不放入 Every Embodied Git history。
- 大文件、过滤后的数据、视频和完整源码快照进入 Datawhale Hugging Face，并记录版本和 SHA256。
- 不能把外部选手仓库的自报成绩写成官方成绩。

## 任务事实

- Task A：越野徒步。
- Task B：垃圾收集。
- Task D：推箱越障。
- Task E：桌面整理。
- Task D 的研究重点是箱体接触、坑/平台通过、状态机、LiDAR 修正和 score-aware recovery。
- Task B 的研究重点是 RGB-D 感知、目标选择、底盘接近、抓取/推动、运输和垃圾桶投放闭环。

## 当前最值得参考的外部方案

- Logic-TARS：Task D 首选参考。固定 commit：`b78c4afd1b84302fe8f88bcfd287eac64c33692c`。
- ZSN2024：Task B B2wPiper Stage1 训练/导出/适配参考。固定 commit：`ee4e0eb97928754d9404a3acd5d644020ac7794c`。
- yma867：Task B YOLO + ByteTrack + RGB-D 深度反投影参考。固定 commit：`e56a2a9e39c5231a91c0a8b1cce8ab1bc0e72403`。

## 重要边界

- 全局相机可以用于 viewer、录像和离线检查；正式 policy 输入必须遵守官方 observation contract。
- B2Piper 与 B2wPiper 的动作维度不同，不能只复制 `solution.py`。
- 训练 reward 上升不等于官方 score 上升；必须保存 score trace、视频和失败阶段。
- `yma867` 的感知层不是完整垃圾收集控制器；其操作入口包含 zero-action placeholder。
- Logic-TARS 的 `32/100` 是 README 自报结果，尚无官方独立佐证。

## 实验纪律

1. 先确认任务 ID、机器人构型、观测 key 和动作维度。
2. 先做 Python 语法、模型加载、单环境播放和 adapter smoke test。
3. 每次只改变一个变量，保护已知最好 checkpoint。
4. 评分、视频和失败分类优先于训练 loss。
5. 外部代码先检查许可证和固定 commit，再做接口迁移。
6. 发布前扫描 token、绝对路径、私有 IP、缓存、权重和未清理日志。
