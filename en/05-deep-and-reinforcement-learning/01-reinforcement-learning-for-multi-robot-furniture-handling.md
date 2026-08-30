# CooHOI: Learning Collaborative Human-Object Interaction by Controlling Object Dynamics




The official implementation of the paper "CooHOI: Learning Cooperative Human-Object Interaction with Manipulated Object Dynamics".



![image-20251012190619444](../../05-具身场景的深度和强化学习/assets/image-20251012190619444.png)

This framework adopts a two-stage learning paradigm. In the first stage, as shown in the left figure, we use the AMP framework to train single-agent handling skills. In the second stage, we transfer these single-agent skills into a collaborative environment. It is worth noting that we utilize the dynamics of objects as feedback information and an implicit communication channel, as indicated by the bounding boxes in the figure.



## Installation



Download Isaac Gym (https://developer.nvidia.com/isaac-gym), or use command-line instructions:

Bash

```
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
```

Create a conda environment:

Bash

```
conda create -n coohoi python=3.8
conda activate coohoi
```

Install the Python wrapper for IsaacGym:

Bash

```
pip install -e isaacgym/python
```

Install other dependencies:

Bash

```
git clone https://github.com/Winston-Gu/CooHOI
pip install -r requirements.txt
```

If you encounter the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, you need to set the environment variables:

Bash

```
export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib
```









View the results for the single-agent object handling task:
If memory is insufficient, it is recommended to set num env to 1 or 4.

Bash

```
CUDA_VISIBLE_DEVICES=0 python coohoi/run.py --test \
--task HumanoidAMPCarryObject \
--num_envs 16 \
--cfg_env coohoi/data/cfg/humanoid_carrybox.yaml \
--cfg_train coohoi/data/cfg/train/amp_humanoid_task.yaml \
--motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
--checkpoint coohoi/data/models/SingleAgent.pth
```

View our results in the dual-agent object handling task:

Bash

```
CUDA_VISIBLE_DEVICES=0 python coohoi/run.py --test \
--task ShareHumanoidCarryObject \
--num_envs 16 \
--cfg_env coohoi/data/cfg/share_humanoid_carrybox.yaml \
--cfg_train coohoi/data/cfg/train/share_humanoid_task_coohoi.yaml \
--motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
--checkpoint coohoi/data/models/TwoAgent.pth
```



### Solo Humanoid Robot Skill Training



Training instructions:

Bash

```
CUDA_VISIBLE_DEVICES=0 python coohoi/run.py \
--task HumanoidAMPCarryObject \
--cfg_env coohoi/data/cfg/humanoid_carrybox.yaml \
--cfg_train coohoi/data/cfg/train/amp_humanoid_task.yaml \
--motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
--headless \
--wandb_name "<experiement_name>"
```

The model weight files (checkpoints) will be found in the `output/Humanoid_<date>_<time>/nn` directory for evaluation:

Bash

```
CUDA_VISIBLE_DEVICES=0 python coohoi/run.py --test \
--task HumanoidAMPCarryObject \
--num_envs 16 \
--cfg_env coohoi/data/cfg/humanoid_carrybox.yaml \
--cfg_train coohoi/data/cfg/train/amp_humanoid_task.yaml \
--motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
--checkpoint <checkpoint_path>
```

For example:

Bash

```
CUDA_VISIBLE_DEVICES=0 python coohoi/run.py --test \
--task HumanoidAMPCarryObject \
--num_envs 16 \
--cfg_env coohoi/data/cfg/humanoid_carrybox.yaml \
--cfg_train coohoi/data/cfg/train/amp_humanoid_task.yaml \
--motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
--checkpoint output/Humanoid_19-16-52-17/nn/Humanoid.pth
```



### Dual-Robot Collaboration Training



By default, collaborative training for dual humanoid robots starts with fine-tuning the individual humanoid robot policy. We load the weight file of the single-agent policy in `--checkpoint <ckpt_path>`, and can replace it with your own weight file.

Collaborative Training:

Bash

```
CUDA_VISIBLE_DEVICES=0 python coohoi/run.py \
--task ShareHumanoidCarryObject \
--cfg_env coohoi/data/cfg/share_humanoid_carrybox.yaml \
--cfg_train coohoi/data/cfg/train/share_humanoid_task_coohoi.yaml \
--motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
--headless \
--is_finetune \
--pretrain_checkpoint <ckpt_path> \
--wandb \
--wandb_name "<experiement_name>"
```

> Note: <ckpt_path> should be the relative path of the single agent policy weight file. This weight file is used to initialize the collaborative policy.

For example:

Bash

```
CUDA_VISIBLE_DEVICES=0 python coohoi/run.py \
--task ShareHumanoidCarryObject \
--cfg_env coohoi/data/cfg/share_humanoid_carrybox.yaml \
--cfg_train coohoi/data/cfg/train/share_humanoid_task_coohoi.yaml \
--motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
--headless \
--is_finetune \
--pretrain_checkpoint coohoi/data/models/SingleAgent.pth \
--wandb \
--wandb_name "CooHOI Training"
```

Evaluation:

Bash

```
CUDA_VISIBLE_DEVICES=0 python coohoi/run.py --test \
--task ShareHumanoidCarryObject \
--num_envs 16 \
--cfg_env coohoi/data/cfg/share_humanoid_carrybox.yaml \
--cfg_train coohoi/data/cfg/train/share_humanoid_task_coohoi.yaml \
--motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
--checkpoint <ckpt_path>
```



## Other reference content



- The initial code is based on [ASE project ](https://github.com/nv-tlabs/ASE).
- The motion dataset comes from [AMASS dataset ](https://amass.is.tue.mpg.de/).
- The object dataset comes from [SAMP dataset ](https://samp.is.tue.mpg.de/).
