# Hugging Face 发布建议

本目录用于记录公开发布的组织方式，不包含模型权重、原始轨迹、视频、访问令牌或私有运行日志。

## 推荐拆分

如果后续要发布到 Hugging Face，建议分成两个仓库：

1. **代码仓库**：发布 `competition/TRONCamp-Mani` 下的配置、审计脚本、评测协议和复盘文档。
2. **数据或模型仓库**：只有在确认比赛规则、数据许可和第三方资产许可后，才发布脱敏后的 manifest、权重或可公开样例。

不要把比赛原始数据、完整 HDF5、checkpoint、评测服务凭据、机器地址、SSH 密钥或内部日志直接上传到公共仓库。大文件应使用 Hugging Face 官方的 Git LFS / 分片机制，并在仓库 README 中记录 SHA256、版本、许可和恢复方法。

## 发布前检查

在本地完成以下检查，并使用环境变量或交互式登录提供令牌，不要把令牌写进脚本：

```bash
cd competition/TRONCamp-Mani
bash code/scan_public_release.sh .
python3 code/build_public_manifest.py . --output /tmp/troncamp-manifest.json
```

发布后应把生成的公开 manifest 和对应 commit SHA 保存到版本说明中，方便读者复核文件是否发生漂移。当前仓库已经提供可复用的工程复盘和严格 processed-data 审计脚本，但没有代替数据许可审核或官方成绩认证。
