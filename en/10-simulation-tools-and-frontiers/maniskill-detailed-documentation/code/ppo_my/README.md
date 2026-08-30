# Implementation of Stable PPO Algorithm

This directory contains a stable version of the PPO algorithm optimized for the ManiSkill environment. This implementation is based on the original PPO implementation (from [CleanRL](https://github.com/vwxyzjn/cleanrl/) and [LeanRL](https://github.com/pytorch-labs/LeanRL/)), but it includes many improvements to numerical stability.

## Key Features

The stable PPO has the following improvements compared to the original PPO implementation:

1. **Enhance numerical stability**
   - Automatically detect and handle NaN values during training
   - Gradient norm monitoring and constraint
   - Boundary constraints for action and value functions
   - Lower learning rate and more conservative network initialization
   - Fewer update iterations (4 instead of 8) to reduce overfitting

2. **Automatic error recovery**
   - Perform emergency checkpoint saving when a large number of NaN values are detected
   - Detection and handling of abnormal gradients
   - Automatically record stability metrics in the log

3. **Enhanced usability**
   - Chinese comments and more detailed parameter descriptions
   - Default parameters are optimized for stability
   - Clearer log output

## Usage Instructions

Here are some examples of how to use common tasks:

### PushCube Task (the simplest task)

```bash
python ppo_my.py --env_id="PushCube-v1" \
  --num_envs=1024 --update_epochs=4 --num_minibatches=32 \
  --total_timesteps=2_000_000 --learning_rate=1e-4 \
  --max_grad_norm=0.25 --eval_freq=10 --num-steps=20
```

### Evaluate the trained model

```bash
python ppo_my.py --env_id="PushCube-v1" \
   --evaluate --checkpoint=path/to/model.pt \
   --num_eval_envs=1 --num-eval-steps=1000
```

### Use Stability Parameters

To achieve optimal stability, the following parameters are recommended:

- `--learning_rate=1e-4`: Lower learning rate to improve stability
- `--max_grad_norm=0.25`: More stringent gradient clipping threshold
- `--update_epochs=4`: Reduce overfitting
- `--detect_anomaly=True`: Enable PyTorch gradient anomaly detection (slightly reduces performance)

More example commands can be found in the `examples.sh` file.

## Notes

- Due to the addition of stability checks, the training speed may be slightly slower than the original implementation.
- For particularly complex tasks, further parameter adjustments may be required.
- If continuous NaN issues occur, you can try reducing the learning rate or the network size.

## Supported Environments

The stable PPO version supports all ManiSkill environments, including:

- Push/Pick Cube tasks: PushCube, PickCube, StackCube
- robotic arm tasks: PegInsertionSide
- multi-robot tasks: TwoRobotPickCube, TwoRobotStackCube
- motion control tasks: MS-AntWalk, MS-CartpoleBalance
- and other ManiSkill environments

## Citation

If you use this stable PPO implementation, please refer to the original PPO paper:

```
@article{DBLP:journals/corr/SchulmanWDRK17,
  author       = {John Schulman and
                  Filip Wolski and
                  Prafulla Dhariwal and
                  Alec Radford and
                  Oleg Klimov},
  title        = {Proximal Policy Optimization Algorithms},
  journal      = {CoRR},
  volume       = {abs/1707.06347},
  year         = {2017},
  url          = {http://arxiv.org/abs/1707.06347},
  eprinttype    = {arXiv},
  eprint       = {1707.06347},
  timestamp    = {Mon, 13 Aug 2018 16:47:34 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/SchulmanWDRK17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```