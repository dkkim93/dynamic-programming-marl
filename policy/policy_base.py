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
        if total_timesteps > self.args.decay_max_timesteps:
            return 0.
        else:
            return self.decay_rate * total_timesteps + 0.5
