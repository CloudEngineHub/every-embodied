# FlashLib：面向现代 GPU 的高速经典机器学习算子库

> 项目仓库：https://github.com/FlashML-org/flashlib
> 项目主页与技术博客：https://flashml-org.github.io/
> PyPI 包名：`flashlib`
> 开源协议：Apache-2.0
> 适合放置位置：`11-其他辅助工具`

FlashLib 是 FlashML-org 开源的 GPU classical machine learning operator library。它不是 VLA、不是机器人控制框架，也不是世界模型；它更像一个高速底层工具库，把 KMeans、KNN、PCA、SVD、UMAP、t-SNE、HDBSCAN、ANN 检索、回归、分类、GEMM 等经典机器学习和线性代数算子重新写成更适合现代 GPU 的版本。

在具身智能教程里，它适合作为“其他辅助工具”收录。原因很直接：具身智能项目里经常会处理大规模 embedding、点云特征、视觉特征、轨迹特征、检索库、聚类结果和降维可视化。很多时候真正卡住 pipeline 的不一定是大模型推理，而是周边的检索、聚类、PCA/SVD、KNN、t-SNE 或 HDBSCAN。如果这些算子从 CPU / scikit-learn / 常规 GPU 路径换成更高效的 GPU primitive，数据整理、特征分析和在线检索链路都会更轻。

## 1. 它解决什么问题

传统经典机器学习算子有一个尴尬点：算法本身很常见，但很多实现仍然按离线批处理时代设计。比如：

- KMeans 会反复计算点到所有聚类中心的距离；
- KNN / ANN 会在大规模 embedding 库上做 top-k；
- PCA / SVD 会构造 Gram matrix 或 covariance matrix，再做特征分解；
- t-SNE / UMAP / HDBSCAN 常用于可视化和无监督分析，但在大数据量上很慢；
- GEMM、QR、eigh、polar 等线性代数 primitive 是很多上层算法的核心瓶颈。

这些算子如果直接按朴素公式写，容易产生巨大的中间矩阵、频繁读写 HBM、atomic contention 或不适合 TensorCore / Hopper 架构的访存模式。FlashLib 的目标就是把这些经典算子重新组织成更适合现代 GPU 的 fused / tiled / hardware-aware 实现。

项目主页给出的定位是：把 classical ML operators 从离线工具变成可以进入 agentic AI、检索、验证、压缩和 orchestration loop 的在线 primitive。这个视角对具身智能也成立。机器人系统里大量中间环节都不是 Transformer 本体，而是：

- 从视觉特征里检索相似 demo；
- 对轨迹 embedding 做聚类；
- 用 PCA / SVD 压缩高维状态或视觉特征；
- 对点云或场景对象做近邻搜索；
- 用 UMAP / t-SNE 可视化策略表示；
- 用 KMeans/HDBSCAN 做数据清洗、异常轨迹发现、失败模式分组。

FlashLib 适合加速的正是这些“模型周边但很关键”的算子。

## 2. 覆盖哪些算子

官方 README 写到当前 release 覆盖 18 个 high-level primitives，主要分成几类：

| 类别 | 算子 |
| :--- | :--- |
| Clustering | `flash_kmeans`、`flash_dbscan`、`flash_hdbscan`、`flash_spectral_clustering` |
| Nearest neighbors | `flash_knn`、`flash_ivf_flat`、`flash_ivf_pq`、`flash_cagra` |
| Decomposition | `flash_pca`、`flash_truncated_svd` |
| Manifold learning | `flash_umap`、`flash_tsne` |
| Regression | `flash_linear_regression`、`flash_ridge`、`flash_logistic_regression` |
| Classification | `flash_multinomial_nb`、`flash_random_forest` |
| Preprocessing | `flash_standard_scaler` |
| Linear algebra | `cov_gemm`、`gram_gemm`、`ab_gemm`、`eigh`、`polar`、`msign`、`cholqr2`、`split_basis`、多精度 GEMM 变体 |

API 设计上，每个 primitive 通常同时提供两种形式：

- 顶层 `flash_*` 函数，适合在 PyTorch pipeline 里直接调用；
- sklearn-style class，例如 `KMeans`、`PCA`、`HDBSCAN`、`IVFFlat`、`IVFPQ`、`CAGRA`，适合 `fit(...)` / `kneighbors(...)` 风格使用。

## 3. 基本安装

最简单的安装方式是：

```bash
pip install flashlib
```

从源码安装：

```bash
git clone https://github.com/FlashML-org/flashlib.git
cd flashlib
pip install -e .
```

建议在 CUDA 可用的 PyTorch 环境里测试。FlashLib 面向 GPU tensor，典型输入是 `torch.Tensor(device="cuda")`。如果机器没有 NVIDIA GPU，仍然可以阅读源码和使用 `flashlib.info` 成本估计接口，但主算子加速价值主要来自 GPU。

## 4. 最小 KMeans 示例

下面这个例子直接来自官方 README 的用法风格。它在 GPU 上生成 100 万个 128 维向量，并用 `flash_kmeans` 聚成 1024 类。

```python
import torch
from flashlib import flash_kmeans

x = torch.randn(1_000_000, 128, device="cuda", dtype=torch.float32)
labels, centroids, n_iter = flash_kmeans(
    x,
    n_clusters=1024,
    max_iters=20,
)

print(labels.shape)
print(centroids.shape)
print(n_iter)
```

这个 smoke test 只证明三件事：

1. `flashlib` 能正确 import；
2. PyTorch CUDA tensor 能进入 FlashLib primitive；
3. KMeans 的 forward pipeline 能跑完并返回 labels / centroids / iteration count。

它不证明某个真实任务一定更快。要比较速度，需要固定输入规模、dtype、GPU 型号、warmup 次数和 baseline，例如 cuML、FAISS、scikit-learn 或 PyTorch 自写版本。

## 5. ANN 检索：IVF-Flat、IVF-PQ 和 CAGRA

FlashLib 也提供 GPU approximate nearest neighbor primitives。官方 README 里重点提到三类：

- `IVFFlat`：先聚类建倒排表，再按 `nprobe` 查若干列表；
- `IVFPQ`：在 IVF 基础上加入 product quantization，用更小内存保存向量；
- `CAGRA`：构建 proximity graph，并用 fused greedy traversal 做高吞吐查询。

IVF-Flat 示例：

```python
import torch
from flashlib import IVFFlat

db = torch.randn(1_000_000, 128, device="cuda")
queries = torch.randn(10_000, 128, device="cuda")

index = IVFFlat(nlist=1024, nprobe=16).fit(db)
distances, indices = index.kneighbors(queries, n_neighbors=10)
```

`nprobe` 是主要 recall knob。提高 `nprobe` 会扫描更多倒排列表，通常 recall 更高、速度更慢。这个参数在 RAG 检索、相似 demo retrieval、轨迹 embedding 检索里都很常见。

IVF-PQ 示例：

```python
from flashlib import IVFPQ

index = IVFPQ(nlist=1024, m=16, nprobe=16).fit(db)
distances, indices = index.kneighbors(queries, n_neighbors=10)
print(index.compression_ratio)
```

如果原始向量是 128 维 fp32，每个向量约 512 bytes；`m=16` 的 PQ code 约 16 bytes，理论上能做到约 32 倍压缩。它适合向量库很大、显存是主要瓶颈的场景，但要接受 recall ceiling 和量化误差。

CAGRA 示例：

```python
from flashlib import CAGRA

index = CAGRA(graph_degree=32, itopk_size=64).fit(db)
distances, indices = index.kneighbors(queries, n_neighbors=10)
```

`itopk_size` 和 `graph_degree` 是 CAGRA 的主要速度/召回调节项。在线小 batch 查询时，图搜索类方法可能比 IVF 更合适；大 batch 且极高 recall 时，IVF-Flat 的矩阵乘 fine-scan 可能更占优。教程里不要把某一个 index 写成绝对最优，应该按 batch size、召回要求、库规模和显存预算选择。

## 6. `flashlib.info`：不跑 GPU 也能估计成本

FlashLib 一个很有意思的设计是 `flashlib.info`。官方 README 里说它可以在纯 CPU 上预测 primitive 的 runtime、FLOPs 和 HBM bytes，耗时约 5 微秒，不 import torch、triton 或 cutlass。

示例：

```python
import flashlib.info as info

est = info.estimate(
    "kmeans",
    shape=(100_000, 64),
    params={"K": 256, "max_iters": 20},
    device="H200",
)

print(est.summary_line())
```

这个接口对 agentic workflow 很有用。比如一个自动化数据处理 agent 在决定“先做 PCA 还是先做 KMeans”之前，可以先估计每一步的成本，再选择是否把任务下发到 GPU。对具身数据处理也类似：如果要处理几百万条视觉 embedding 或轨迹 embedding，可以先用 `info.estimate(...)` 估一个数量级，避免直接开跑后发现显存或时间预算不够。

## 7. 为什么它和具身智能有关

FlashLib 不是机器人专用库，但在具身智能项目里有几个直接使用点。

### 7.1 大规模 demo / trajectory 聚类

收集 DROID、BridgeData、RoboTwin、LIBERO 或自建真机数据后，经常需要把示教按任务、场景、动作模式或失败类型分组。可以先用 VLM / policy encoder 提取 episode embedding，再用 KMeans 或 HDBSCAN 聚类。

```text
robot episodes -> episode embeddings -> flash_kmeans / flash_hdbscan -> cluster labels -> 数据清洗 / 失败模式分析
```

### 7.2 相似示教检索

很多 VLA fine-tuning 或 prompt retrieval 会找“当前任务最像哪些历史 demo”。这本质上是 nearest neighbor search。小规模可以直接 `flash_knn`；大规模可以用 `IVFFlat`、`IVFPQ` 或 `CAGRA`。

```text
task embedding / visual embedding -> ANN index -> top-k demos -> prompt / adapter warm start / replay sampling
```

### 7.3 表征降维和可视化

训练 VLA、世界模型或导航策略时，常常需要看 embedding space 是否按任务、场景、动作阶段分开。PCA、t-SNE、UMAP 是常用分析工具。FlashLib 可以让这些可视化前处理更快，尤其是在样本数较大时。

### 7.4 点云与 3D 特征处理

3D 场景重建、点云分割、局部几何匹配和 grasp proposal filtering 经常需要 KNN、PCA 或聚类。FlashLib 的 KNN/PCA/KMeans 可以作为原型加速组件，但具体能否替换已有 3D pipeline，还要看数据格式、精度要求和 batch 组织方式。

## 8. 设计思想：不是简单“把 sklearn 搬到 GPU”

FlashLib 项目主页把设计原则总结成四类，这里翻译成更工程化的说法。

第一，**数学等价重写**。很多算子的朴素写法会生成大中间矩阵。FlashLib 尽量把公式改成 GPU-friendly 形式，例如 KMeans assignment 不显式 materialize `N x K` 距离矩阵，而是在寄存器里保留局部最小值。

第二，**硬件感知 kernel**。同一个算子在不同 shape、不同 GPU、不同 batch size 下最优实现不同。FlashLib 使用 Triton 和 CuteDSL 构建多个 kernel variant，并根据 workload 做 heuristic dispatch。

第三，**tolerance-driven dispatch**。有些科学计算要求严格精度，有些 embedding 聚类或检索任务可以接受小误差换速度。FlashLib 暴露 `tol` 这样的精度预算，让 dispatcher 在满足残差约束的前提下选择更快路径。

第四，**成本可预测 API**。`flashlib.info` 把 runtime、FLOPs、HBM bytes 和 bound regime 暴露出来，让上层 pipeline 或 LLM agent 在执行前就能估算成本。

这也是它比单纯“快一点的 KMeans”更值得收录的原因：它把 classical ML operator 做成了可组合、可预算、可由 agent 调度的基础 primitive。

## 9. 使用时要注意什么

FlashLib 还比较新，教程里建议保守使用。

1. **先做数值对齐**。把同一批输入拿给 FlashLib 和 baseline，比较 labels、distances、indices、explained variance、embedding 或 classification result。对于聚类和 t-SNE/UMAP 这种非唯一输出，要比较目标指标和下游效果，而不是逐元素完全一致。
2. **区分 exact 和 approximate**。`flash_knn`、`IVFFlat`、`IVFPQ`、`CAGRA` 的语义不同。ANN index 的 recall 需要显式评估。
3. **固定硬件和版本再报告速度**。官方 benchmark 主要在 H200、CUDA 13.0、PyTorch 2.11、Triton 3.6、cuML 25.10 上比较。换到 A100、4090、L40S 或不同 CUDA/PyTorch，速度结论要重新测。
4. **注意 JIT warmup**。Triton/CuteDSL 路径首次调用会有编译开销，benchmark 要丢掉 first call 或单独报告 cold start。
5. **不要用它替代所有库**。如果只是几千条数据，CPU sklearn 或 PyTorch 可能已经够快；FlashLib 更适合大规模 GPU-resident tensor。

## 10. 推荐 smoke test

下面是一个适合写进实验记录的最小检查流程。

```bash
python - <<'PY'
import torch
from flashlib import flash_kmeans

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())

x = torch.randn(100_000, 64, device="cuda", dtype=torch.float32)
labels, centroids, n_iter = flash_kmeans(x, n_clusters=256, max_iters=5)

print("labels", tuple(labels.shape), labels.dtype, labels.device)
print("centroids", tuple(centroids.shape), centroids.dtype, centroids.device)
print("n_iter", n_iter)
PY
```

预期现象：

- `cuda available` 为 `True`；
- `labels` 形状是 `(100000,)`；
- `centroids` 形状是 `(256, 64)`；
- 输出 tensor 在 CUDA device 上；
- `n_iter` 不超过 `max_iters`。

如果这里失败，优先检查：

- 当前 Python 环境里是否安装了 `flashlib`；
- PyTorch 是否能看到 CUDA；
- 输入 tensor 是否真的在 GPU 上；
- CUDA / PyTorch / Triton 版本是否和当前平台兼容；
- 是否在没有 NVIDIA GPU 的机器上误跑了 GPU primitive。

## 11. 在本教程库里的定位

FlashLib 应该和 `flash-attn快速安装.md`、`hf快速下载同时避免下载不需要的其他分支文件.md`、`Xget库.md`、`modelscope快速下载.md` 这类内容放在一起。它不是具身智能主线算法，但能显著影响实验效率。

如果后续要扩展，可以补三类实测小实验：

- 用 `flash_kmeans` 对机器人 episode embedding 聚类；
- 用 `IVFFlat` / `CAGRA` 做相似 demo 检索；
- 用 `flash_pca` / `flash_tsne` 可视化 VLA hidden states 或轨迹 embedding。

## 参考链接

- FlashLib GitHub：https://github.com/FlashML-org/flashlib
- FlashLib 项目主页与技术博客：https://flashml-org.github.io/
- PyPI 安装入口：https://pypi.org/project/flashlib/
