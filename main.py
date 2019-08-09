import argparse
import os
import numpy as np
from misc.utils import set_log, make_env
from tensorboardX import SummaryWriter
from policy.agent import Agent
from misc.vis import vis


def main(args):
    # Check arguments
    assert args.n_agent == 2, "Only two agents are supported"

    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./data"):
        os.makedirs("./data")

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
        Agent(env=env, tb_writer=tb_writer, log=log, args=args, name="agent", i_agent=i_agent)
        for i_agent in range(args.n_agent)]

    # Get true return by Monte Carlo estimate
    if args.estimate_option == "montecarlo":
        from trainer.montecarlo import train
        table = train(agents=agents, env=env, log=log, tb_writer=tb_writer, args=args)
    elif args.estimate_option == "naive":
        from trainer.naive import train
        table = train(agents=agents, env=env, log=log, tb_writer=tb_writer, args=args)
    elif args.estimate_option == "ours":
        from trainer.ours_mp import train
        # from trainer.ours import train
        table = train(agents=agents, env=env, log=log, tb_writer=tb_writer, args=args)
    else:
        raise ValueError("Invalid option")

    # Save and vis result
    save_name = args.estimate_option + "_" + str(args.decay_max_timesteps)
    if args.estimate_option == "ours":
        save_name += "_" + str(args.future_max_timesteps)
    np.save("./data/" + save_name + ".npy", table) 

    # vis(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Algorithm
    parser.add_argument(
        "--n-agent", default=2, type=int,
        help="Number of agents")
    parser.add_argument(
        "--decay-max-timesteps", default=250, type=int,
        help="Maximum timestep for agent1's time-varying policy")
    parser.add_argument(
        "--future-max-timesteps", default=5, type=int, 
        help="Discount factor")
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
        "--estimate-option", default="montecarlo", 
        choices=["montecarlo", "naive", "ours"], 
        help="Options for estimating return")
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--seed", default=0, type=int, 
        help="Sets Gym, PyTorch and Numpy seeds")

    args = parser.parse_args()

    # Set log name
    args.log_name = \
        "env::%s_row::%s_estimate_option::%s_future_max_timesteps::%s" \
        "ep_max_timesteps::%s_decay_max_timesteps::%s_prefix::%s_log" % (
            args.env_name, args.row, args.estimate_option, 
            args.future_max_timesteps, args.ep_max_timesteps, 
            args.decay_max_timesteps, args.prefix)

    main(args=args)
