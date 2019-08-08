import argparse
import os
import numpy as np
from misc.utils import set_log, make_env
from tensorboardX import SummaryWriter
from trainer.train import train


def set_policy(env, tb_writer, log, args, name, i_agent):
    from policy.agent import Agent
    policy = Agent(env=env, tb_writer=tb_writer, log=log, args=args, name=name, i_agent=i_agent)

    return policy


def main(args):
    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # Set logs
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))
    log = set_log(args)

    # Create env
    env = make_env(args)

    # Set seeds
    env.seed(args.seed)
    np.random.seed(args.seed)

    # Set agents
    agents = [
        set_policy(env, tb_writer, log, args, name="agent", i_agent=i_agent)
        for i_agent in range(2)]

    # Start train
    train(agents=agents, env=env, log=log, tb_writer=tb_writer, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Algorithm
    parser.add_argument(
        "--agent1-max-timesteps", default=1000, type=int,
        help="Maximum timestep for agent1's time-varying policy")
    parser.add_argument(
        "--discount", default=0.99, type=float, 
        help="Discount factor")

    # Env
    parser.add_argument(
        "--env-name", type=str, required=True,
        help="OpenAI gym environment name")
    parser.add_argument(
        "--row", type=int, required=True,
        help="# of rows in the environment")
    parser.add_argument(
        "--ep-max-timesteps", type=int, required=True,
        help="Episode is terminated when max timestep is reached")

    # Misc
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--seed", default=0, type=int, 
        help="Sets Gym, PyTorch and Numpy seeds")

    args = parser.parse_args()

    # Set log name
    args.log_name = "env::%s_row::%s_prefix::%s_log" % (args.env_name, args.row, args.prefix)

    main(args=args)
