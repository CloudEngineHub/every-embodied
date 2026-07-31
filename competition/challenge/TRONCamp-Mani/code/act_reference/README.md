# ACT reference snapshot

This directory contains the ACT policy, training, processing, deployment, and DETR-VAE source snapshot used by the TRONCamp Mani experiments.

- Upstream repository: `https://github.com/limx-troncamp/troncamp-mani`
- Upstream commit: `7630a1a68558c2e64d9a17c54d4e4a907bca7db4`
- ACT license: MIT; retain the upstream license when redistributing this snapshot.
- Deliberately excluded: raw/processed datasets, checkpoints, runtime environments, private logs, credentials, and machine-specific paths.

The reproducibility package in the neighboring `configs/`, `docs/`, and `results/` directories records the exact T4 configuration and dataset identity. The complete processed training input is stored separately in the private Hugging Face dataset package described by `hf/README.md`.
