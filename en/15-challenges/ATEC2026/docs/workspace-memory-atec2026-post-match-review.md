# ATEC2026 Post-Event Review Workspace Memory

Update time: 2026-07-31

## Work Boundaries

- GitHub public entry: `15-Challenge竞赛/ATEC2026/`.
- The original ATEC engineering, training cache, environment, and large checkpoint are not included in Every Embodied Git history.
- Large files, filtered data, videos, and full source code snapshots are uploaded to Datawhale Hugging Face, and versions and SHA256 values are recorded.
- The self-reported scores of external contestant repositories should not be written as official scores.

## Task Facts

- Task A: Off-road hiking.
- Task B: Garbage collection.
- Task D: Pushing boxes over obstacles.
- Task E: Desk arrangement.
- The focus of Task D is box contact, pit/platform passage, state machine, LiDAR correction, and score-aware recovery.
- The focus of Task B is RGB-D perception, target selection, chassis proximity, grasping/pushing, transportation, and a closed loop for trash bin placement.

## Current external solutions worth referring to

- Logic-TARS: Preferred reference for Task D. Fixed commit: `b78c4afd1b84302fe8f88bcfd287eac64c33692c`.
- ZSN2024: Reference for Training/Export/Adaptation of Task B B2wPiper Stage1. Fixed commit: `ee4e0eb97928754d9404a3acd5d644020ac7794c`.
- yma867: Reference for Task B YOLO + ByteTrack + RGB-D depth back projection. Fixed commit: `e56a2a9e39c5231a91c0a8b1cce8ab1bc0e72403`.

## Important Boundaries

- The global camera can be used for viewer, recording, and offline inspection; the input for the official policy must comply with the official observation contract.
- B2Piper and B2wPiper have different manipulation dimensions, and `solution.py` cannot be simply copied.
- An increase in training reward does not equal an increase in official score; score traces, videos, and failure stages must be preserved.
- The perception layer of `yma867` is not a complete garbage collection controller; its manipulation entry includes a zero-action placeholder.
- The `32/100` of Logic-TARS is reported by the README, and there is no official independent verification yet.

## Experimental Discipline

1. First, confirm the task ID, robot configuration, observation key, and action dimension.
2. First, perform Python syntax, model loading, single-environment playback, and adapter smoke test.
3. Change only one variable at a time to preserve the known best checkpoint.
4. Scoring, videos, and failure classification take precedence over training loss.
5. Check the license and fixed commits of external code before migrating the interface.
6. Scan for tokens, absolute paths, private IPs, caches, weights, and uncleaned logs before release.
