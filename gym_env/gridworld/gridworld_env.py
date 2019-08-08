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
        reward = self.reward_func(self.agent_locs)

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
            next_loc = self.agent_locs[i_agent] - 1
            if next_loc < 0:
                next_loc = 0  # Out of bound
            self.agent_locs[i_agent] = next_loc
    
        elif action == 1:  # Right
            next_loc = self.agent_locs[i_agent] + 1
            if next_loc >= self.row:
                next_loc = self.row - 1  # Out of bound
            self.agent_locs[i_agent] = next_loc
    
        else:
            raise ValueError("Wrong action")

    def transition_func(self, rows, actions):
        # TODO combine with _take_action
        next_rows = []

        for row, action in zip(rows, actions):
            if action == 0:  # Left
                next_row = row - 1
                if next_row < 0:
                    next_row = 0  # Out of bound
            elif action == 1:  # Right
                next_row = row + 1
                if next_row >= self.row:
                    next_row = self.row - 1  # Out of bound
            else:
                raise ValueError("Wrong action")

            next_rows.append(next_row)

        return next_rows

    def reward_func(self, agent_locs):
        """Reward function:
        - +1 iff two agents are located at both ends
        - +0 o/w
        """
        if agent_locs[0] == 0 and agent_locs[1] == self.row - 1:
            return 1.
        elif agent_locs[1] == 0 and agent_locs[0] == self.row - 1:
            return 1.
        else:
            return 0.
