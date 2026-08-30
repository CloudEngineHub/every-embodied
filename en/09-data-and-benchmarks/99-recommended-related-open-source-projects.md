# Open-Source Data and Evaluation Projects

This index collects projects that complement the datasets, tasks, and evaluation tools introduced in this chapter. Before combining a resource with an experiment, verify the robot embodiment, action space, task protocol, data license, and downloadable assets.

## RoboBenchMart

- Project: [emb-ai/RoboBenchMart](https://github.com/emb-ai/RoboBenchMart)
- Best used for: locating manipulation benchmarks, task definitions, datasets, and related resources.
- Check before use: published scores from different projects may use different robots, initial states, episode lengths, and success criteria.

## Open X-Embodiment

- Project: [Open X-Embodiment](https://robotics-transformer-x.github.io/)
- Best used for: understanding how observations, actions, and task descriptions can be unified across robot embodiments.
- Check before use: embodiment mappings, action dimensions, normalization statistics, and data licenses.

## LeRobot

- Project: [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- Best used for: organizing episodes, images, states, actions, and task text in a common dataset layout with training and visualization tools.
- Check before use: frame rate, timestamps, feature shapes, and episode boundaries after conversion.

## Benchmark Selection Checklist

1. Does the task suite measure the generalization capability relevant to the project?
2. Does it provide fixed initial states, an evaluation script, and an explicit success condition?
3. Are its observations and actions compatible with the target robot?
4. Are the required compute resources, simulation assets, and model weights available?
5. Does the license permit the intended educational, research, or commercial use?

For every reported result, record the task subset, number of episodes, maximum episode length, and aggregation method so that models are compared under the same protocol.
