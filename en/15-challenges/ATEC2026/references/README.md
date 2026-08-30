# External Scheme Index

This directory does not copy the complete code from third-party repositories. Instead, it records upstream links, fixed commits, licenses, and migration boundaries. This ensures the author attribution and version traceability, and prevents code from different robots and simulation versions from being mistakenly regarded as a direct submission method for Datawhale.

## Logic-TARS

- Upstream repository: [Logic-TARS/ATEC2026](https://github.com/Logic-TARS/ATEC2026)
- Fixed commit: `b78c4afd1b84302fe8f88bcfd287eac64c33692c`
- License: MIT, subject to the upstream repository `LICENSE`
- Key files: `README.zh-CN.md`, `submission/solution.py`, `scripts/train/`, `scripts/export/`
- Applicable direction: Task D B2W box obstacle avoidance
- Migratable content: 61D/16D interface, 16D→24D adapter, phase state machine, LiDAR box detection, heading correction, lock recovery
- Things that cannot be assumed directly: checkpoint availability, constant scoring threshold, identical interfaces between B2Piper and B2wPiper

## ZSN2024

- Upstream repository: [ZSN2024/ATEC2026_Simulation_Challenge](https://github.com/ZSN2024/ATEC2026_Simulation_Challenge)
- Fixed commit: `ee4e0eb97928754d9404a3acd5d644020ac7794c`
- Applicable direction: Task B B2wPiper Stage1 training, export, and adapter
- Note: The garbage collection part in the unified demo still belongs to basic heuristic control, not a proven high-score closed loop

## yma867

- Upstream repository: [yma867/ATEC2026_Simulation_Challenge_RIL](https://github.com/yma867/ATEC2026_Simulation_Challenge_RIL)
- Fixed commit: `e56a2a9e39c5231a91c0a8b1cce8ab1bc0e72403`
- Key directory: `taskb_perception/`
- Applicable directions: Task B YOLO, ByteTrack, RGB-D depth back projection and world coordinates
- Note: The manipulation entry retains a zero-action placeholder, and the chassis and robotic arm control need to be implemented separately
