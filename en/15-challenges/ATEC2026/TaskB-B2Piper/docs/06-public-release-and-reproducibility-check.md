# 6. Public Publication and Reproduction Check

## 6.1 The Division of Labor between GitHub and Hugging Face

GitHub save:

- Chinese tutorials and problem reviews.
- Lightweight inspection scripts, configurations, and result boundaries.
- Original links, licenses, and migration instructions for external solutions.
- Code that can be read and audited without competition assets.

Hugging Face save:

- List of public archives corresponding to the tutorial.
- Small anonymization examples and re-distributable configurations.
- Large file entries that will be added after authorization.

Official simulation assets, raw videos, checkpoints, and private logs do not automatically gain re-distribution permissions just because they are "placed on HF". It is necessary to confirm the source, license, and file size before uploading.

## 6.2 Pre-release Checklist

- [ ] No token, password, SSH private key, or authenticated URL.
- [ ] No machine user directory, mount point, internal network address, or private runner path.
- [ ] No official assets, private logs, or unauthorized weights.
- [ ] No writing of environment camera, truth object pose, or object ID into formal observations.
- [ ] All experimental results are annotated with evidence type and constraints.
- [ ] External code includes repository links, author attribution, and license information.
- [ ] Large files do not bypass Git LFS or directly use ordinary Git blob.
- [ ] Paths in documentation commands use variables such as `$ATEC_ROOT` and `$HF_REPO`.
- [ ] Evaluation descriptions include runner, seed, environment version, and official tester information.

## 6.3 Minimum Fields in Reproduction Report

A reproducible experiment should record at least:

```text
commit
environment_version
runner
seed
robot_platform
observation_keys
action_space
policy_checkpoint
episode_count
official_score
physical_grasp_verified
placement_verified
video_paths
failure_code
```

When `physical_grasp_verified` or `placement_verified` is missing, the report can only state "close to/touching/local control results" and cannot state "complete garbage collection successful".

## 6.4 Recommendations for Tutorial Readers

First, reproduction the official minimum environment, and then gradually enable search, end locking, grasping, and placement. Replace only one module at a time; submit the failed examples along with the fixes. The value of the tutorial lies precisely in these boundary conditions.
