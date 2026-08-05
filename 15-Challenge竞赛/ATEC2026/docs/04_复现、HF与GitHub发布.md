# 4. 复现、Hugging Face 与 GitHub 发布

## 4.1 发布分层

公开教程需要同时满足可读性、可复现性和安全性。推荐把资源拆成三层：

| 位置 | 放什么 | 不放什么 |
|---|---|---|
| GitHub `15-Challenge竞赛/ATEC2026/` | Markdown、轻量 Python、配置模板、许可证、固定 commit、问题复盘 | 权重、数据集、缓存、私有日志、token |
| Datawhale Hugging Face | 完整源码快照、模型权重、过滤后数据、视频、日志归档、SHA256 | token、私有 IP、未清理的完整服务器目录 |
| 本地 / 服务器 | Isaac Sim、conda 环境、原始训练输出、缓存和中间 checkpoint | 不作为教程默认路径 |

GitHub 文档中的命令使用 `$PROJECT_ROOT`、`$DATA_ROOT`、`$MODEL_ROOT` 等变量，不能出现作者机器上的绝对路径。

## 4.2 外部仓库复现

从本目录执行：

```bash
export ATEC_REFERENCE_ROOT=/path/to/atec-reference-repos
bash code/external_repo_audit/clone_references.sh "$ATEC_REFERENCE_ROOT"
```

预期结果：

```text
$ATEC_REFERENCE_ROOT/logic-tars-atec
$ATEC_REFERENCE_ROOT/zsn2024-atec2026
$ATEC_REFERENCE_ROOT/yma867-atec2026-ril
```

这个步骤只验证公开仓库和固定 commit 可获取，不代表 Isaac Lab 已安装，也不代表 policy 可以在当前机器上直接运行。

## 4.3 最小 smoke test

### 代码层

```bash
python3 -m py_compile code/taskd_reference_adapter.py
```

它只能证明 Python 语法和文件结构正确，不能证明机器人行为正确。

### 接口层

```bash
python3 code/taskd_reference_adapter.py
```

它应打印 16D 输入被适配成 24D 输出，以及不同分数对应的 Task D 阶段。它不启动 Isaac Sim，也不代替官方评分器。

### 仿真层

在已经配置好 Isaac Sim / Isaac Lab 的环境中，先用一个环境播放：

```bash
python tools/atec/list_envs.py
python tools/atec/play_task.py \
  --task ATEC-TaskD-B2wPiper \
  --headless --num_envs 1 --debug
```

具体环境名以官方仓库当前版本为准。第一次只检查窗口、机器人、相机、动作 shape 和日志，不进行长训练。

## 4.4 Hugging Face 上传内容

建议创建一个公开 Datawhale 数据集仓库，例如：

```text
Datawhale/atec2026-simulation-challenge-reproducibility
```

推荐目录：

```text
README.md
manifests/
  external_repos.yaml
  checksums.sha256
source/
  taskd_reference_adapter.py
  external_repo_audit/
references/
  logic-tars/
  zsn2024/
  yma867/
logs/
  redacted/
videos/
```

第三方源码快照必须包含原许可证和上游 commit；如果仓库规模较大，也可以只上传 manifest 和差异补丁，在 GitHub 教程中链接上游固定 commit。

## 4.5 发布前检查

```bash
rg -n -i \
  'hf_[A-Za-z0-9]{20,}|token|password|secret|api[_-]?key|/[h]ome/|/[d]ata/' \
  15-Challenge竞赛/ATEC2026
```

检查结果中：

- token、密码和私钥必须为零；
- 机器绝对路径只能出现在说明“替换成自己的路径”的示例中；
- 本地绝对路径改成 `$PROJECT_ROOT`、`$DATA_ROOT` 或 `/path/to/...`；
- 日志中删除私有服务器名、内网 IP、订阅 URL 和账号信息；
- 大文件使用 Git LFS 或 Hugging Face，不直接进入普通 Git history；
- 每个权重和数据分片都有 SHA256；
- 第三方仓库的许可证、作者、上游链接和固定 commit 不能遗漏。

## 4.6 教程读者应如何理解结果

教程中把证据分成三类：

1. **已验证**：源码、语法检查、文件 hash、固定命令输出或本地视频直接支持；
2. **作者自报**：公开仓库 README 中的排名、训练时间或精度；
3. **待复现**：需要 Isaac Sim、GPU、checkpoint 或官方评分器，当前只能给出复现路径。

这样可以避免把“代码能下载”“policy 能加载”“机器人能移动”“任务得分高”混写成一个结论。
