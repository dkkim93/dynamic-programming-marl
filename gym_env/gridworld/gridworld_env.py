import gym
import numpy as np
from gym import spaces


class GridworldEnv(gym.Env):
    def __init__(self, row, n_agent):
        super(GridworldEnv, self).__init__()

        self.row = row
        self.agent_locs = [0. for _ in range(n_agent)]

        self.observation_space = spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # Right and left

    def step(self, actions):
        # Get reward
        reward = self._reward_func()

        # Take step and get next observation
        for i_agent, action in enumerate(actions):
            self._take_action(i_agent, action)
        next_observation = self.agent_locs

        done = True if reward == 1. else False

        return (next_observation, reward, done, {})

    def reset(self, rows):
        for i_agent, row in enumerate(rows):
            self.agent_locs[i_agent] = row
        return self.agent_locs

    def render(self):
        raise NotImplementedError()

    def _take_action(self, i_agent, action):
        action = np.argmax(action)

        if action == 0:  # Left
            new_loc = self.agent_locs[i_agent] - 1
            if new_loc < 0:
                new_loc = 0  # Out of bound
            self.agent_locs[i_agent] = new_loc
    
        elif action == 1:  # Right
            new_loc = self.agent_locs[i_agent] + 1
            if new_loc >= self.row:
                new_loc = self.row - 1  # Out of bound
            self.agent_locs[i_agent] = new_loc
    
        else:
            raise ValueError("Wrong action")

    def _reward_func(self):
        """Reward function:
        - +1 iff two agents are located at both ends
        - +0 o/w
        Because the starting end is zero and the other end is always self.row - 1,
        to check the condition, we can simply sum their location
        """
        reward = 1. if np.sum(self.agent_locs) == self.row - 1 else 0.
        return reward
