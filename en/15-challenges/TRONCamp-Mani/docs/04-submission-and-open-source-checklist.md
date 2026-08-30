# Submission and Open Source Checklist

## 1. Before online submission

```text
[ ] 账号已按 T1 -> T2 -> T3 -> T4 顺序解锁
[ ] token 只从文件或环境变量读取，不写入命令历史和日志
[ ] 权重、dataset_stats.pkl、代码树和 deploy config 已验 SHA
[ ] checkpoint 与 stats 位于约定目录，文件名与 config 一致
[ ] code root 是最小 external/robotwin_local 镜像
[ ] 代码包内没有软链接、硬链接、wandb、缓存、数据和备份目录
[ ] import smoke 只依赖官方线上环境已有包
[ ] fixed fresh3/fresh10/fresh20 结果已保存，未挑高合并
[ ] 已记录 server response、queue id、server SHA 和时间
[ ] 截止时间前得到明确回执；仅有上传进程不算入队
[ ] 不覆盖 protected candidate
```

## 2. Before Git / Hugging Face release

```text
[ ] `git status` 已确认没有把用户已有改动混入提交
[ ] 仓库中没有 token、私钥、cookie、内部服务器 URL
[ ] 没有个人绝对路径、PID/PGID、端口和未脱敏日志
[ ] 没有 HDF5、checkpoint、pickle、safetensors、视频和大缓存
[ ] 没有专有 Tron2 mesh、URDF/MJCF 或未确认许可的第三方资产
[ ] 结果表标明 local / diagnostic / official-online 的边界
[ ] 所有 SHA 对应公开可获得的文件；否则只作为 private archive reference
[ ] README 说明复现所需的官方资产和许可证
[ ] Hugging Face 资源使用独立模型/数据集仓库并可恢复、可校验
[ ] 先在干净 clone 中运行扫描脚本和文档链接检查
```

## 3. The public boundaries for this submission

This public package only includes:

- Summary of training configuration;
- Data audit scripts and public manifest tools;
- Experimental results and failure analysis;
- Final online results (25.3/100);
- Resource migration, testing topology, submission packages, and masking experience.

It does not contain any secret materials that can restore account access or directly run the proprietary simulator. If large files are uploaded later, clearly state in the Hugging Face README: source license, data/model version, number of files, total size, SHA256, restoration command, and known limitations.
