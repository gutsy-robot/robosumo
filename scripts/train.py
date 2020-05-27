import gym, mujoco_py
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C
import numpy as np
import robosumo.envs
from gym import spaces
from robosumo.policy_zoo import LSTMPolicy, MLPPolicy
from robosumo.policy_zoo.utils import load_params, set_from_flat
from wrapper import RoboSumoWrapper
from stable_baselines import PPO1

# env = make_vec_env('RoboSumo-Ant-vs-Ant-v0', n_envs=4)
env = gym.make('RoboSumo-Ant-vs-Ant-v0')
print("original action space: ", env.action_space)
print("original observation space: ", env.observation_space)

env_player1 = RoboSumoWrapper(env, player_id=1)
policy1 = PPO1(MlpPolicy, env_player1, verbose=1)

env_player0 = RoboSumoWrapper(env)
policy0 = PPO1(MlpPolicy, env_player0, verbose=1)

env_player0.opponent_policy = policy1

print("action space of policy0 is: ", policy0.action_space)
print("observation  space of policy0 is: ", policy0.observation_space)

policy0.learn(total_timesteps=5)
policy0.save("policy0")

del policy0  # remove to demonstrate saving and loading

model = PPO1.load("policy0")

obs = env_player0.reset()
while True:
    print("shape of obs is: ", obs.shape)
    action, _states = model.predict(obs)

    print("action is: ", action)
    obs, rewards, dones, info = env_player0.step(action)
    # obs = env.state
    env.render(mode="human")
