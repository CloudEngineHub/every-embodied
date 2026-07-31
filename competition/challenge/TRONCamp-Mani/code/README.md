# 公开代码说明

这里的脚本是与赛事工作区解耦的公开工具，不是官方 RoboTwin/Tron2 资产的替代品。

## 严格审计 processed ACT 数据

```bash
python code/audit_processed_t4.py \
  --dataset-dir /path/to/processed_data \
  --expected 550 \
  --raw-manifest-sha256 eeca954a7e64dda818a0cca86534e34f6ffa1a8055bd37810f9781a264199901 \
  --output /tmp/processed_audit.json
```

审计失败就停止，不要在脚本里自动删除、补帧、重排相机或修改 action。

## 生成公开材料 SHA 清单

```bash
python code/build_public_manifest.py competition/TRONCamp-Mani \
  --output /tmp/troncamp_public_manifest.json
```

该脚本会拒绝把 checkpoint、HDF5、pickle、视频和符号链接纳入公开清单。它不替代 Hugging Face 上传后的远端校验。

## 发布前扫描

```bash
bash code/scan_public_release.sh competition/TRONCamp-Mani
```

扫描通过后，仍应人工检查 Markdown、CSV 和 JSON；自动扫描不能识别所有个人信息、专有资产或错误的实验结论。
