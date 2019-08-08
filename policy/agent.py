import numpy as np
from policy.policy_base import PolicyBase


class Agent(PolicyBase):
    def __init__(self, env, tb_writer, log, args, name, i_agent):
        super(Agent, self).__init__(
            env=env, log=log, tb_writer=tb_writer, args=args, name=name, i_agent=i_agent)

        self.set_dim()

    def set_dim(self):
        self.actor_output_dim = self.env.action_space.n

        self.log[self.args.log_name].info("[{}] Actor output dim: {}".format(
            self.name, self.actor_output_dim))

    def select_stochastic_action(self, total_timesteps):
        """Between two agent, they have different policies
        - Agent0: random agent
        - Agent1: agent selects action a function of total_timesteps
        """
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
