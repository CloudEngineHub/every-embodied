# 5. Archive reconciliation and local cleanup

Update time: 2026-07-31

This record is used to answer a specific question: which ATEC2026 materials have been publicly archived, which local copies can be deleted, and to what extent they can be reproduced after deletion.

## 5.1 Public Archive Mapping

| Local Data Category | Public Location | Status |
|---|---|---|
| GitHub tutorials, issue reviews, observation contracts, inspection scripts, and external solution indexes | `15-Challenge竞赛/ATEC2026/` of `datawhalechina/every-embodied` | Submitted to public branch |
| Official public simulation source code snapshot | `source/` for Datawhale HF Task B dataset | Uploaded; SHA-256 is `84ccfdc3903e4e03a5de8a7dedd90314b15ed09382c136bbeee6a858dae802d9` |
| Task B debugging/assessment preview videos and frames | `TaskB-B2Piper/videos/` and `frames/` of the same HF dataset | Uploaded; the media serves only as debug/evidence, not representing official results |
| Official player guide and update instructions | `official_docs/` of the same HF dataset | Uploaded; local paths, credentials, and private email addresses have been verified |
| Historical Task E XSA plugin and configuration | `historical_task_e/` of the same HF dataset | Uploaded; serves only as historical data, not representing the Task B controller |
| Task E filtered training data, code, robot assets, public logs, and ACT weights | Reproduction dataset and model repository for Datawhale HF Task E | Publicly available; filtered data includes shard SHA-256 and recovery scripts |
| Chinese Workspace Memory | GitHub `docs/WORKSPACE_MEMORY_ATEC2026_赛后复盘.md`, as well as masked copies in two HF archives | Uploaded; only public facts and reproduction boundaries are retained |

## 5.2 Masking Processing

- The native absolute path in the public Task E verification file has been removed, and only the verification value and the public file name are retained; the verification value remains unchanged.
- The original Workspace Memory contains machine paths and environment details that are not publicly disclosed; the public version has been changed to Chinese, path-independent post-match analysis.
- The Task B video and screenshots retain the annotation "Environment perspective used only for debugging," and the environment perspective is not included in the official observation contract.
- Before release, tokens, passwords, private keys, private IPs, native machine paths, and authenticated URLs were checked.
- For external contestants' plans, only public links, fixed commits, licenses, and migration boundaries are retained; self-reported results are not written as official results.

## 5.3 Deleted Local Copies

The following content has been publicly archived or is clearly an irreproducible intermediate product, so it is no longer retained as a local source:

- The official source code compressed package, official guide, Task B video/screenshots, and historical Task E XSA materials uploaded by HF have been reproduced. Delete the local copies for this project.
- The original Workspace Memory corresponding to the publicly available materials that have been de-identified have been removed. Delete the local version containing machine paths, and keep only the public Chinese version.
- Approximately 42 GB of Task E OpenPI failure/midpoint checkpoint files; the public reproduction instructions explicitly exclude these failed training states.
- Approximately 6.1 GB of Task E LeRobot conversion intermediate data; the public reproduction uses the filtered HDF5 fragments from HF, rather than these conversion copies.
- Temporary staging files and local media output copies generated during the HF upload process.

The public tutorials in the Every Embodied repository have not been deleted, and the workspace modifications unrelated to ATEC have not been changed.

## 5.4 Reproduction Boundary

After deleting the local copy, the public information can be reproduced:

1. Observation audits, release scans, file inventories, and comparison with external schemes in the tutorial.
2. Location and hash verification of official source code snapshots, official guides, and historical experimental data.
3. Download of filtered data from Task E, fragment merging, hash verification, and the reproduction entry for ACT.
4. Evidence review of media debugging in Task B, and the judgment that “environmental perspectives are not considered formal observations”.

Public materials cannot ensure that the complete Task B score can be obtained solely through `git clone` or HF downloads. A full simulation still requires a compatible Isaac Sim/Isaac Lab, GPU driver, Python dependencies, official runner, and evaluator; the currently available archives also do not claim to have a verified Task B complete score checkpoint.
