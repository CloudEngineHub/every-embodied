# 清理 pip 下载与构建缓存

`pip` 会把下载的安装包和构建结果保存在用户缓存目录。安装被中断后，这些文件仍会保留。优先使用 `pip` 自带的缓存命令，不要直接删除整个 `$HOME/.cache`，因为其他工具也可能使用该目录。

## 查看缓存

```bash
python -m pip cache dir
python -m pip cache info
```

第一条命令显示实际缓存目录，第二条命令显示文件数量和占用空间。

## 清理缓存

清理全部 `pip` 缓存：

```bash
python -m pip cache purge
```

只删除指定包的缓存：

```bash
python -m pip cache remove torch
```

## 验证与排错

再次运行 `python -m pip cache info`，占用空间应明显下降。若提示当前版本不支持 `cache` 子命令，先执行 `python -m pip install --upgrade pip`；若权限不足，确认缓存目录属于当前用户，不要用 `sudo pip` 混合写入用户环境。
