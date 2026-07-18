# 其他辅助工具

本目录整理具身智能实验中常用的辅助工具。这里的内容不一定是机器人算法本身，但会影响环境配置、模型下载、数据处理、GPU 算子性能和日常开发效率。

## 高速计算与算子库

- [FlashLib 高速机器学习库](FlashLib高速机器学习库.md)：面向现代 GPU 的 classical ML operator library，覆盖 KMeans、KNN、ANN、PCA、SVD、UMAP、t-SNE、HDBSCAN、GEMM 等，适合大规模 embedding、轨迹特征和检索/聚类分析。
- [flash-attn 快速安装](flash-attn快速安装.md)：按 PyTorch、CUDA、Python 和 ABI 匹配 FlashAttention wheel，避免从源码长时间编译。

## 模型与数据下载

- [Hugging Face 快速下载并避免无关分支文件](hf快速下载同时避免下载不需要的其他分支文件.md)
- [ModelScope 快速下载](modelscope快速下载.md)
- [Xget 库](Xget库.md)
- [Go 语言安装和 S3 下载内容安装](go语言安装和s3下载内容安装.md)

## 环境与系统工具

- [conda / mamba / pip / uv 等包管理工具](conda_mamba_pip_uv等包管理工具/)
- [Docker 相关](docker相关/)
- [Git 相关使用](git相关使用/)
- [VS Code 使用相关](vscode使用相关/)
- [WSL](wsl/)
- [WSA](wsa/)
- [Linux 快速安装 micromamba](linux快速安装micromamba.md)
- [Ubuntu 安装中文输入法](ubuntu安装中文输入法.md)
- [Vulkan 重要环境配置](vulkan重要环境配置.md)

## 设备与网络排障

- [JoyCon 连接 X5](joycon连接x5.md)
- [网络测试工具](网络测试工具.md)
- [虚拟网卡上网问题相关](虚拟网卡上网问题相关.md)
- [解决远程服务器内容系统变化](解决远程服务器内容系统变化.md)
- [Linux 命令行复制为 Markdown 带字体颜色内容](linux命令行复制为markdown带字体颜色内容.md)
