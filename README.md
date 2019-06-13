# gym-fraud

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
