import logging
from gym.envs.registration import register

register(
    id='fraud-v0',
    entry_point='gym_fraud.envs:FraudEnv',
)
