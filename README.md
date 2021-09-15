# Navigation with deep reinforcement learning

## about
This repository is an implementation of deep reinforcement learning,
for the first project of [Udacity Nanodegree program "Deep Reinforcement Leraning."](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
The environment is based on "[Unity Banana Collector](https://github.com/ostamand/banana-collector)",
where two types of bananas (yellow ones and blue ones) are randomly placed in a square world,
and an agent is required to collect as many yellow bananas as possible,
while avoiding blue bananas.

## environment
 - states : The state consists of 37 kinds of values,
including the agent's velocity and ray-based perception of objects around it.
 - actions : The agent has 4 possible actions:
move forward or backward and change direction to left or right.
 - rewards : +1 if the agent collects a yellow banana,
 while -1 for a blue banana.
 - This is an episodic-task problem,
 and the agent is required to get an average score better than +13
 over 100 consecutive episodes.


## installation
You need to work with python 3.6.13.
Otherwise, you may fail installation some package,
especially an old version of tensorflow.
In addition, it is recommended to create with a new virtual environment
and work in it for the following processes.

[Unity ML-agents](https://github.com/openai/gym) and numpy need to be installed in adavance. 

```
pip install gym
pip install gym[box2d]
```

You also need to download a zip file of the Unity environment as follows:
 - [for linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
 - [for mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
 - [for 64-bit windows](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
 - [for 32-bit windows](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)

It is convenient to unzip the downloaded zip file above
under the repository directory you will create next.


Then, after cloning this repository, go to the "deep-reinforcement-learning" submodule and install necessary libraries as below:

```
git clone https://github.com/maedamaeday/UdacityRL_p1.git --recursive
cd UdacityRL_p1/deep-reinforcement-leraning/python
pip install .
cd ../../
```

Note that you need "--recursive" option in cloning the repository
to incorporate deep-reinforcement-learning submodule properly.

Finally, create IPython kernel as below:

```
python -m ipkernel install --user --name env_name --display-name "env_name"
```

Please replace "env_name" with your virtual environment.

## model detail and its performance

The details for the model architecture, training method and its performance is
summarized in [Report.md](Report.md)