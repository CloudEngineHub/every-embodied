# Hugging Face Publishing Recommendations

This directory is used to record the public release methods, and does not include model weights, original trajectories, videos, access tokens, or private operation logs.

## Recommended Splitting

If you want to publish it to Hugging Face later, it is recommended to split it into two repositories:

1. **Code repository**: Publish configurations, audit scripts, testing protocols, and review documents under `15-Challenge竞赛/TRONCamp-Mani`.
2. **Data or model repository**: Only after confirming the competition rules, data permissions, and third-party asset permissions, can the masked manifest, weights, or public samples be released.

Do not upload raw competition data, complete HDF5 files, checkpoints, evaluation service credentials, machine addresses, SSH keys, or internal logs directly to a public repository. Large files should use Hugging Face's official Git LFS/segmenting mechanism, and the SHA256, version, license, and recovery methods should be recorded in the repository README.

## Pre-release Check

Perform the following checks locally, and use environment variables or interactive login to provide the token. Do not write the token into a script:

```bash
cd 15-Challenge竞赛/TRONCamp-Mani
bash code/scan_public_release.sh .
python3 code/build_public_manifest.py . --output /tmp/troncamp-manifest.json
```

After release, save the generated public manifest and the corresponding commit SHA in the version description, so that readers can verify whether the files have changed. The current repository provides reusable project review and strict processed-data audit scripts, but it does not replace data permission review or official performance certification.
