# Navigation with deep reinforcement learning

## about
This repository is an implementation of deep reinforcement learning,
for the first project of Udacity Nanodegree program "Deep Reinforcement Leraning."
The environment is based on "[Unity Banana Collector](https://github.com/ostamand/banana-collector)",
where two types of bananas (yellow ones and blue ones) are randomly placed in a square world,
and an agent is required to collect as many yellow bananas as possible,
without picking blue bananas.

## environment
The state consists of 37 kinds of values,
which contains the agent's position and velocity.
The agent has 4 possible actions:
move forward or backward and change direction to left or right.

## model
The agent is trained with Deep Q-Network,
where the action-value function is given as a few layers of fully connected neural networks to output Q values for each action.
The objective function is mean squred error between an expected Q value and a target Q value.
The expected Q value is an output of the neural network
and the target Q value is obtained as (reward for the state and the action)+gamma*(maximum Q value for the next sate among possible actions),
where gamma is a discount rate.
In calculating the target Q value, Q values for the next state is taken from a "similar" neural network to one used in calculating the expected Q value.


## installation
[Unity ML-agents](https://github.com/openai/gym) and numpy need to be installed in adavance. 
'''
pip install gym
pip install gyn[box2d]
'''
