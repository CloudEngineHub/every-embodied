# Embodied AI Data and Evaluation Benchmarks

This chapter answers three questions: how robot data are organized, how policies are compared under fixed protocols, and how experiment results are preserved for review. Study LIBERO and SimplerEnv first, then continue to data collection or generative-manipulation evaluation as needed.

## Reading Order

1. [LIBERO Datasets and Engineering Format](./01-libero.md)
2. [SimplerEnv Benchmarking](./02-simplerenv.md)
3. [Quest 3 Teleoperation Data Collection](./03-vr-data-collection.md)
4. [Minimal EBench and GenManip Reproduction](./04-EBench.md)
5. [Related Open Benchmark Index](./99-recommended-related-open-source-projects.md)

## Evaluation Record

A comparable policy evaluation should fix the task suite, initial states or random seeds, episodes per task, maximum steps, success criterion, model checkpoint, action post-processing, and software versions. For long-horizon tasks, record stage outcomes such as approach, contact, grasp, transport, and placement in addition to the aggregate success rate.

## Completion Criteria

- Identify observations, states, actions, task instructions, and episode boundaries in a dataset.
- Explain the difference between offline action error and online closed-loop success rate.
- Run an evaluation with a fixed task denominator and preserve per-episode results and representative videos.
