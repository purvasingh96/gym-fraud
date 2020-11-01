# Deep Reinforcement Learning for Credit Card Fraud Detection

* [How to create new gym environment in openai](https://github.com/openai/gym/blob/master/docs/creating-environments.md)

# Installation 
```
cd gym-fraud
pip install -e .
```

# Usage 

**Step - 1 :** Create a directory named *dataset* in your folder containing the main program.<br>
**Step - 2 :** Download [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it inside *dataset folder*<br>
**Step - 3 :** In your code create an instance of gym_fraud environment using the following commands <br>

 ```python
import gym
import gym_fraud
env = gym.make('fraud-v0')
```

# Overview
Due to the rapid advancement in electronic commerce technology, the use of credit cards has
dramatically increased. The increasing popularity of credit
card as a payment mode for both online and regular
purchases has led to a rise in fraudulent cases of credit card
transactions. For many years, numerous supervised machine
learning models for anomaly detection have achieved
state-of-the-art performance. In this paper, we present a
novel deep Q-network architecture and a custom OpenAI
Gym environment for our deep reinforcement learning
agent that utilizes Experience Replay and uses value
function approximation. The deep Q-agent employs
epsilon-greedy policy to perform classification action based
on batches of input. The OpenAI environment then
evaluates the agent’s action and rewards the agent
accordingly. The agent’s memory stores this entire
experience. At the end of batch completion, the deep Q-agent
samples a batch of memory from its experience buffer and
updates the Q-value using the Q-network computes the loss
and performs back-propagation to update the weights.
Results show that our model successfully classified
fraudulent and non-fraudulent transactions and has
achieved state-of-the-art performance.

# Algorithm

# Results 

# Research Paper
