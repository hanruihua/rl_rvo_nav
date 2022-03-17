# rl_rvo_nav

The source code of the paper "Reinforcement Learned Distributed Multi-Robot Navigation with Reciprocal Velocity Obstacle Shaped Rewards"


Circle 10                  |   Circle 16       | Circle 20 
:-------------------------:|:-------------------------:|:-------------------------:
![](rl_rvo_nav/gif/rl_rvo_cir_10.gif)  | ![](rl_rvo_nav/gif/rl_rvo_cir_16.gif) | ![](rl_rvo_nav/gif/rl_rvo_cir_20.gif)

Random 10                  | Random 16  | Circle 20 
:-------------------------:|:-------------------------:|:-------------------------:
![](rl_rvo_nav/gif/rl_rvo_random_10.gif) | ![](rl_rvo_nav/gif/rl_rvo_random_16.gif) | ![](rl_rvo_nav/gif/rl_rvo_random_20.gif)

## Prerequisites

- Python >= 3.8
- Pytorch >= 1.6.0
- [intelligent-robot-simulator](https://github.com/hanruihua/intelligent-robot-simulator) == v2.5

```
git clone -b v2.5 https://github.com/hanruihua/intelligent-robot-simulator.git
cd intelligent-robot-simulator
pip install -e .
```

## Test environment

- Ubuntu 20.04, 18.04
- Windows 10, 11

## Installation

```
git clone https://github.com/hanruihua/rl_rvo_nav.git
cd rl_rvo_nav
./setup.sh
```

## Policy Train

- First stage: circle scenario with 4 robots.

```
python train_process.py --use_gpu
```

- Second state: continue to train in circle scenario with 10 robots.

```
python train_process.py --con_train --robot_number 10 --train_epoch 2000 --use_gpu
```

## Policy Test




## Pretrained model







