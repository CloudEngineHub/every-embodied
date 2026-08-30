# Robot Manipulation and VLA Policies

This chapter presents the robot-policy learning route from behavior cloning to vision-language-action policies. Learn the model families and data interfaces first, then complete one end-to-end experiment. The method guides compare how current systems handle temporal memory, 3D space, action representation, and online adaptation.

## Foundation Reading

- [VLA Overview and Survey](./01-vla-overview-and-survey.md)
- [RT-1, RT-2, and RT-X](./large-model-control-vla-vlm/03-rt-series-paper-interpretation-and-code-analysis/01-rt-series-paper-and-code-analysis.md)
- [OpenVLA Environment, Evaluation, and Architecture](./large-model-control-vla-vlm/02-openvla-reproduction/02-openvla-reproduction.md)

## End-to-End Experiments

- [Train and Evaluate SmolVLA on LIBERO](./large-model-control-vla-vlm/01SmolVLA-LIBERO/01SmolVLA-libero.md)
- [Collect MuJoCo Data and Train ACT, Pi0, and SmolVLA](./large-model-control-vla-vlm/04-mujoco-reproduction-act-pi0-smolvla/README.md)
- [Train and Evaluate DiT4DiT on LIBERO](./large-model-control-vla-vlm/05DiT4DiT-LIBERO/01-dit4dit-libero-training-and-evaluation.md)
- [Policy Diagnosis and Physical-Success Evaluation](./large-model-control-vla-vlm/04-mujoco-reproduction-act-pi0-smolvla/09-policy-diagnosis-and-physical-success-evaluation.md)

## Method Guides

| Theme | Chapters |
| --- | --- |
| Temporal memory and visual evidence | [EventVLA](./large-model-control-vla-vlm/06-eventvla-visual-evidence-memory-guide/README.md), [VisualThink-VLA](./large-model-control-vla-vlm/15-visualthink-vla-visual-evidence-reasoning-guide/README.md) |
| Open models and engineering frameworks | [WALL-OSS](./large-model-control-vla-vlm/07-wall-oss-open-source-vla-model-introduction/README.md), [WALL-X](./large-model-control-vla-vlm/08-wall-x-open-source-engineering-framework-navigation/README.md), [EVA-Client](./large-model-control-vla-vlm/19-eva-client-real-robot-deployment-and-evaluation-engineering-navigation/README.md) |
| 3D and physical understanding | [3DVLA](./large-model-control-vla-vlm/09-3dvla-3d-spatial-and-instance-understanding-guide/README.md), [PhysBrain](./large-model-control-vla-vlm/10-physbrain-physical-common-sense-enhancement-vla-guide/README.md) |
| Reinforcement learning and online adaptation | [PRTS](./large-model-control-vla-vlm/11-prts-reinforcement-learning-native-vla-guidance/README.md), [LWD](./large-model-control-vla-vlm/14-lwd-real-robot-cluster-reinforcement-learning-introduction/README.md), [Agentic-VLA](./large-model-control-vla-vlm/16-agentic-vla-online-adaptation-guide/README.md), [Dexbotic-RLinf](./large-model-control-vla-vlm/17-dexbotic-rlinf-engineering-vla-post-training-guide/README.md) |
| Action representation and efficient inference | [Galaxea G0.5](./large-model-control-vla-vlm/12-galaxea-g0-5-autoregressive-vla-guidance/README.md), [Dexora](./large-model-control-vla-vlm/13-dexora-high-dof-dual-arm-dexterous-vla-guidance/README.md), [DM0.5 and OpenDM](./large-model-control-vla-vlm/18-dm0-5-high-performance-inference-and-opendm-guidance/README.md) |

## Completion Criteria

- Draw the data flow among images, language, robot state, action chunks, and environment feedback.
- Distinguish pretrained-model evaluation, fine-tuning, a short pipeline check, and a complete closed-loop evaluation.
- Record the data version, model configuration, task denominator, and per-episode result for one policy experiment.
