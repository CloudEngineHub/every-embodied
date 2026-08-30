# ManiSkill Sim-to-Real Quick Start

This compact exercise installs the reference implementation, pins the compatible LeRobot revision, and verifies the ManiSkill runtime with a random-action example.

```
mamba create -n py311 "python==3.11" # 3.11 is recommended
git clone https://github.com/StoneT2000/lerobot-sim2real.git
pip install -e .
# 如果torch没有正确安装cuda，配置下面这句
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu121/   # install the version of torch that works for you
```





```
git clone https://github.com/huggingface/lerobot.git
cd lerobot
# note that the code was based on a slightly older lerobot commit. LeRobot recently changed the location of a few files we import so it broke some imports
# latest LeRobot can work but some LeRobot import paths need to be updated
git reset --hard a989c795587d122299275c65a38ffdd0a804b8dc
pip install -e .
```



Then, follow the vulkan configuration under other auxiliary tools.

```
python -m mani_skill.examples.demo_random_action
```

Please continue referring to the following:

https://github.com/StoneT2000/lerobot-sim2real.git
