# Reinforcement Learning with Twin Delayed Deep Deterministic Policy Gradient (TD3) and Deep Deterministic Policy Gradient (DDPG)
This repository contains the implementation of the Twin Delayed Deep Deterministic Policy Gradient (TD3) and Deep Deterministic Policy Gradient (DDPG) algorithms in PyTorch.
We are trying to train a agent (red car) to follow a target (green car) in a 2D environment. The agent has to follow the target while avoiding non-road areas which we call sand. The agent is rewarded for following the target and for staying on road and penalized for traversing on sand. The agent has to learn to navigate the environment to reach the target while avoiding sand area.

## Image
![img](/images/autonomous-car.png)