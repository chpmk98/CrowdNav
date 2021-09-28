# GroupNav
This repository contains the codes for our CS 289A final project. For more details, please refer to our paper
[Robot Navigation Around Groups of Pedestrians](http://people.eecs.berkeley.edu/~spohland/files/CS289.pdf) and our [video presentation](https://www.youtube.com/watch?v=-5TDMzaw9xY&ab_channel=AlvinTan)

## Abstract
In order to deploy service robots in non-industrial settings, it is imperative to design robots that navigate safely and effectively in the presence of humans. While a large body of work has designed robot navigation policies that account for typical human movement patterns, the majority of this work has treated humans as independent individuals. In reality, most people move in groups, so it is important to design robot navigation policies that consider the social rules surrounding group behavior. Because group-aware robot navigation is an important and underexplored area of research, we chose to critically revisit the published paper “Group-Aware Robot Navigation in Crowded Environments,” which used attention-based deep reinforcement learning to design a robot navigation policy. In recreating this paper, we generated data using the CrowdNav simulation, which we modified to simulate dynamic social groups. We also implemented a group-aware reward function and neural network architecture, integrated the PPO reinforcement learning algorithm and Adam optimizer into the CrowdNav learning pipeline, and enabled the use of imitation learning with PPO. After training a group-aware robot navigation policy, which we called GAP, we compared GAP with two baseline navigation policies: ORCA and SARL. We found that a robot controlled by GAP reaches its goal more often and collides with groups of pedestrians less often when compared to the baseline policies. However, when the robot is controlled by GAP, it also exhibits behaviors that are undesirable in the real-world. With more time and improved computational resources, we would have tuned the training parameters and trained the policies over more episodes to address this issue.

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting Started
This repository is organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy gap
```
2. Test policies with 500 test cases.
```
python test.py --policy orca --phase test
python test.py --policy gap --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy orca --phase test --visualize --test_case 0
python test.py --policy gap --model_dir data/output --phase test --visualize --test_case 0
```
4. Visualize a test case.
```
python test.py --policy gap --model_dir data/output --phase test --visualize --test_case 0
```
5. Plot training curve.
```
python utils/plot.py data/output/output.log
```
