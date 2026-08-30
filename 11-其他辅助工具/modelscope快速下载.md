# 使用 ModelScope 按目录下载模型或数据集

## 安装依赖

```bash
pip install modelscope

pip install "datasets>=2.16.0,<3.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 编写下载脚本

```python
from modelscope.hub.snapshot_download import snapshot_download
import os

# 1. 设置下载参数
repo_id = 'agibot_world/GenieSim3.0-Dataset'
# 目标本地存放路径
local_dir = os.path.expandvars("$MODEL_ROOT/organize_items")

# 2. 执行下载
# allow_patterns 用于匹配你想要的子目录或特定文件
# 这样只会下载 organize_items 文件夹下的内容，节省时间和空间
print(f"正在从 {repo_id} 下载 checkpoints...")
snapshot_download(
    repo_id,
    local_dir=local_dir,
    allow_patterns=[
        'checkpoints/organize_items/**',  # 匹配该目录下所有文件
    ],
    repo_type='dataset'  # 明确告诉它这是一个数据集仓库
)

print(f"下载完成！文件已存放在: {local_dir}")
```

运行前设置输出根目录：

```bash
export MODEL_ROOT=/path/to/models
python download.py
```
