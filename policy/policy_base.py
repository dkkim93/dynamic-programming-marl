class PolicyBase(object):
    def __init__(self, env, log, tb_writer, args, name, i_agent):
        super(PolicyBase, self).__init__()

        self.env = env
        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.name = name + str(i_agent)
        self.i_agent = i_agent

    def set_dim(self):
        raise NotImplementedError()

    def select_stochastic_action(self):
        raise NotImplementedError()

    def set_threshold(self, total_timesteps):
        if total_timesteps > self.args.agent1_max_timesteps:
            return 0.
        else:
            slope = -0.5 / float(self.args.agent1_max_timesteps)  # TODO don't compute again
            return slope * total_timesteps + 0.5
