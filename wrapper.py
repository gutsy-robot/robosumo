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
    def __init__(self, env, player_id=0, opponent_policy=None, combined_train=False):
        print("calling super")
        super().__init__(env)
        # super().__init__()
        self.env = env
        self.reward_range = (-np.inf, np.inf)
        self.player_id = player_id
        self.combined_train = combined_train
        obs_space = env.observation_space
        self.observation_space = spaces.Box(shape=(2 * obs_space[0].shape[0],), low=obs_space[0].low[0],
                                            high=obs_space[0].high[0])
        if not self.combined_train:
            self.action_space = spaces.Box(shape=env.action_space[0].shape, low=env.action_space[0].low[0],
                                           high=env.action_space[0].high[0])
        else:
            self.action_space = spaces.Box(shape=(2 * env.action_space[0].shape[0], ), low=env.action_space[0].low[0],
                                           high=env.action_space[0].high[0])

        self.opponent_policy = opponent_policy
        self.state = self.reset()
        self.win_rate = []
        self.loss_rate = []
        self.draw_rate = []
        self.num_wins = 0
        self.num_loss = 0
        self.num_draws = 0
        self.num_episodes = 0

    def step(self, action):
        if not self.combined_train:
            opponent_action, _ = self.opponent_policy.predict(self.state)
            if self.player_id == 0:
                action = (action, opponent_action)
            elif self.player_id == 1:
                action = (opponent_action, action)

            s, reward, gameOver, info = self.env.step(action)
            self.state = np.concatenate((s[0], s[1]))

            if gameOver[0]:
                draw = True
                if 'winner' in info[self.player_id].keys():
                    self.num_wins += 1
                    draw = False
                if 'winner' in info[np.abs(self.player_id - 1)].keys():
                    self.num_loss += 1
                    draw = False

                if draw:
                    self.num_draws += 1

                self.num_episodes += 1
                self.draw_rate.append(self.num_draws / self.num_episodes)
                self.win_rate.append(self.num_wins / self.num_episodes)
                self.loss_rate.append(self.num_loss / self.num_episodes)

            return self.state, reward[self.player_id], gameOver[self.player_id], info[self.player_id]

        else:
            action = (action[:int(len(action) / 2)], action[int(len(action) / 2):])
            s, reward, gameOver, info = self.env.step(action)
            self.state = np.concatenate((s[0], s[1]))

            if gameOver[0]:
                draw = True
                if 'winner' in info[self.player_id].keys():
                    self.num_wins += 1
                    draw = False
                if 'winner' in info[np.abs(self.player_id - 1)].keys():
                    self.num_loss += 1
                    draw = False

                if draw:
                    self.num_draws += 1

                self.num_episodes += 1
                self.draw_rate.append(self.num_draws / self.num_episodes)
                self.win_rate.append(self.num_wins / self.num_episodes)
                self.loss_rate.append(self.num_loss / self.num_episodes)

            return self.state, reward[0] - reward[1], gameOver[self.player_id], info[self.player_id]

    def reset(self):
        state = self.env.reset()
        self.state = np.concatenate((state[0], state[1]))
        return self.state
