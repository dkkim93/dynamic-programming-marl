import numpy as np
from policy.policy_base import PolicyBase


class Agent(PolicyBase):
    """Between two agent, they have different policies
    - Agent0: random agent that selects either action with 0.5 probability
    - Agent1: probability of selecting action0 decreases linearly
    - decay_rate specifies how fast action0 would decrease (i.e., slope)
    """
    def __init__(self, env, tb_writer, log, args, name, i_agent):
        super(Agent, self).__init__(
            env=env, log=log, tb_writer=tb_writer, args=args, 
            name=name, i_agent=i_agent)

        self.actor_output_dim = self.env.action_space.n

        if i_agent == 1:
            self.decay_rate = -0.5 / float(self.args.agent1_max_timesteps)

    def select_stochastic_action(self, total_timesteps):
        if self.i_agent == 0:
            if np.random.rand() > 0.5:
                action = np.array([1., 0.])
            else:
                action = np.array([0., 1.])

        elif self.i_agent == 1:
            threshold = self.set_threshold(total_timesteps)

            if np.random.rand() > threshold:
                action = np.array([1., 0.])
            else:
                action = np.array([0., 1.])

        else:
            raise ValueError("Only two agents are supported")

        assert not np.isnan(action).any()

        return action
