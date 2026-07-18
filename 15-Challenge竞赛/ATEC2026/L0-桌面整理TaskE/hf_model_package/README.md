# Hugging Face 模型上传说明

ATEC2026 L0 Task E 最终保护权重已上传到 Hugging Face，而不是直接提交到 GitHub。

## 模型仓库

```text
https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best
```

上传 commit：

```text
https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best/commit/4583bda9e3118e75987a0f91de62c155a70a2bed
```

## 需要上传的文件

已从本机暂存目录上传：

```text
/data/Data14TB/01Proj/ATEC2026/hf_task_e_act_seed1_best
```

原始来源：

```text
/home/ubuntu/Documents/01Proj/13atec/ATEC2026_Simulation_Challenge/submissions/task_e_act_seed1_best_20260608_upload/policy_act.pt
/home/ubuntu/Documents/01Proj/13atec/ATEC2026_Simulation_Challenge/submissions/task_e_act_seed1_best_20260608_upload.zip
```

也可以上传完整目录：

```text
/home/ubuntu/Documents/01Proj/13atec/ATEC2026_Simulation_Challenge/submissions/task_e_act_seed1_best_20260608_upload/
```

## SHA256

```text
policy_act.pt
282614de9673dc01e229557448e380b6a02c196b5c3953cec77b2aebd2305a8e
```

完整 zip 的 SHA256 见：

```text
task_e_act_seed1_best_20260608_upload.zip.sha256
```

## 网页上传

如果已经在浏览器登录 Datawhale Hugging Face 账号，可以直接进入模型仓库网页更新文件，不需要 CLI token。

## CLI 上传

命令行更新需要 token。示例：

```bash
pip install -U huggingface_hub
huggingface-cli login

huggingface-cli upload Datawhale/atec2026-task-e-act-seed1-best \
  /home/ubuntu/Documents/01Proj/13atec/ATEC2026_Simulation_Challenge/submissions/task_e_act_seed1_best_20260608_upload/policy_act.pt \
  policy_act.pt \
  --repo-type model

huggingface-cli upload Datawhale/atec2026-task-e-act-seed1-best \
  /home/ubuntu/Documents/01Proj/13atec/ATEC2026_Simulation_Challenge/submissions/task_e_act_seed1_best_20260608_upload.zip \
  task_e_act_seed1_best_20260608_upload.zip \
  --repo-type model
```

如需重新上传，请核对 README 链接和 SHA256：

```text
../README.md
```
