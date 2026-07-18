# L0 B2Piper option-MoE 提交代码骨架

这个目录保存 ATEC2026 线上赛·赛道1 L0「机器人徒步」的一个可公开提交代码骨架。

文件说明：

- `solution.py`：官网调用入口。
- `solution_rl.py`：B2Piper 低层策略加载、速度命令 schedule、manual/rough option-MoE 逻辑。
- `requirements.txt`：示例为空，表示不额外安装 pip 包，依赖官方镜像内置环境。

注意：

1. 这里不附带 `policy.pt`。请从官方 baseline、自己的训练导出结果或比赛允许公开的权重中补入同目录。
2. 该代码默认加载同目录 `policy.pt`，并假设它是 `45 -> 12` 的 TorchScript locomotion actor。
3. `rough option-MoE` 是赛后复盘中的工程尝试：保留保护版 gait prior，通过 score 段和 progress gate 在 rough 段切换谨慎推进/脱困命令。
4. 它不是冠军方案，也没有承诺满分；更适合当作“如何把比赛策略逻辑封装到 `solution.py`”的参考。

提交目录结构示例：

```text
solution.py
solution_rl.py
policy.pt
requirements.txt
```

本地 smoke：

```bash
python - <<'PY'
import torch
m = torch.jit.load("policy.pt", map_location="cpu")
print(m(torch.zeros(1, 45)).shape)
PY
```

期望输出：

```text
torch.Size([1, 12])
```
