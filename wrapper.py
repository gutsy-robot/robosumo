import gym, mujoco_py
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C
import numpy as np
import robosumo.envs
from gym import spaces
from robosumo.policy_zoo import LSTMPolicy, MLPPolicy
from robosumo.policy_zoo.utils import load_params, set_from_flat


class RoboSumoWrapper(gym.Wrapper):
    def __init__(self, env, player_id=0, opponent_policy=None):
        super().__init__(env)
        # super().__init__()
        self.env = env
        self.player_id = player_id
        obs_space = env.observation_space
        self.observation_space = spaces.Box(shape=(2 * obs_space[0].shape[0],), low=obs_space[0].low[0],
                                            high=obs_space[0].high[0])
        self.action_space = spaces.Box(shape=env.action_space[0].shape, low=env.action_space[0].low[0],
                                       high=env.action_space[0].high[0])
        self.opponent_policy = opponent_policy
        self.state = self.reset()

    def step(self, action):
        opponent_action, _ = self.opponent_policy.predict(self.state)
        if self.player_id == 0:
            action = (action, opponent_action)
        elif self.player_id == 1:
            action = (opponent_action, action)

        s, reward, gameOver, info = self.env.step(action)
        self.state = np.concatenate((s[0], s[1]))
        return self.state, reward[self.player_id], gameOver[self.player_id], info[self.player_id]

    def reset(self):
        state = self.env.reset()
        self.state = np.concatenate((state[0], state[1]))
        return self.state
