# Hugging Face Model Upload Instructions

The final protection weight for the ATEC2026 L0 Task E has been uploaded to Hugging Face, rather than being submitted to GitHub directly.

## Model Repository

```text
https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best
```

Upload commit:

```text
https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best/commit/4583bda9e3118e75987a0f91de62c155a70a2bed
```

## Files to be uploaded

Uploaded from the local temp directory:

```text
$DATA_ROOT/ATEC2026/hf_task_e_act_seed1_best
```

Original source:

```text
$PROJECT_ROOT/submissions/task_e_act_seed1_best_20260608_upload/policy_act.pt
$PROJECT_ROOT/submissions/task_e_act_seed1_best_20260608_upload.zip
```

You can also upload the entire directory:

```text
$PROJECT_ROOT/submissions/task_e_act_seed1_best_20260608_upload/
```

## SHA256

```text
policy_act.pt
282614de9673dc01e229557448e380b6a02c196b5c3953cec77b2aebd2305a8e
```

For the SHA256 of the complete zip file, see:

```text
task_e_act_seed1_best_20260608_upload.zip.sha256
```

## Webpage Upload

If you have logged in to the Datawhale Hugging Face account in the browser, you can directly access the model repository website to update files without needing a CLI token.

## CLI Upload

Command line updates require a token. Example:

```bash
pip install -U huggingface_hub
huggingface-cli login

huggingface-cli upload Datawhale/atec2026-task-e-act-seed1-best \
  $PROJECT_ROOT/submissions/task_e_act_seed1_best_20260608_upload/policy_act.pt \
  policy_act.pt \
  --repo-type model

huggingface-cli upload Datawhale/atec2026-task-e-act-seed1-best \
  $PROJECT_ROOT/submissions/task_e_act_seed1_best_20260608_upload.zip \
  task_e_act_seed1_best_20260608_upload.zip \
  --repo-type model
```

If you need to upload again, please check the README link and SHA256:

```text
../README.md
```
