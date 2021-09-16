# report of the model and its performance

## model
The agent is trained with Deep Q-Network,
where the action-value function is given as deep neural networks.
In addition to using deep neural networks,
several techniques are applied to stabilize learning.
### neural network architecture
The network structure is a fully-connected neural networks
with two hidden layers, where the number of each unit is 64 and 16, respectively.
For simplicity, the agent's state (37 variables) is used
as an input of the neural network.
The outputs are four Q values, corresponding to each of four actions.
The objective function is temporal difference (TD) error,
or mean squred error between an expected Q value and a target Q value.
The expected Q value is an output of the neural network
and the target Q value is obtained as (reward for the state and the action)+gamma*(maximum Q value for the next sate among possible actions),
where gamma is a discount rate and is 0.99 in this repository. 
### epsilon-greedy method
While the agent is being trained,
it select an optimal action, where the largest value is expected,
with probability of 1-epsilon,
while choose an action randomly with epsilon probability.
Large epsilon value, or closer to 1, is used at the beginning,
and it gradually decreased.
Here, epsilon=1 at the beginning
and decreased by a constant rate (0.995) for each episode until it reaches 0.01.
### experience replay
The agent does not learn every time it moves,
but it used past state-action-reward information which is randomly sampled.
For each time step, a tuple of state, action, reward, and next state is stored
in a replay buffer.
It can hold information up to 1e5 steps
and old information is discard to add new information when the buffer is full.
The agent learns, or updates it weight every 4 steps,
where it samples information of 64(=batch size) steps from the replay buffer
and it is used for training.
### fixed target
In calculating the target Q value above, Q values for the next state is taken from a "similar" neural network to one used in calculating the expected Q value.
Here, the "similar" network, refered to as the "target" network,
is the same as the main network,
but updated less often.


## performance
Here is an example of learning history of the agent.
The training continued until an average score of 100 consecutive episode
reches 15.
This goal was achieved with about 600 episodes.

![learning history](learning_history.png)

The gif file "demo.gif" shows performance of the trained agent.
In this example, the achieved score is 17.

![demo of the trained agent](demo.gif)

## possible future improvement
Various techniques can be applied to improve training and performance of the agent.
One option is priotized experience replay,
an improved version of experience replay applied in this model.
In this method, rare experience with large TD error is sampled with higher probability
and the agent can learn wider cases effectively.
In watching actions of the current trained agent,
a rare case actually happened;
two yellow bananas are observed
at the left and right side with the same distance from the agent
and the agent got stuck,
where it repeats to turn right and left infinitely.
If such a case is effectively learned in the training,
it would be possible for the agent to avoid to got stuck.



