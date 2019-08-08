import numpy as np

RUNS = 100
timesteps = 0


def process_one_traj(agents, row0, row1, env, args):
    # Reset variables and environment
    global timesteps
    timesteps = 0
    discounted_return = 0.
    env.reset(rows=[row0, row1])

    while True:
        # Select actions
        actions = []
        for agent in agents:
            actions.append(agent.select_stochastic_action(timesteps))
            
        # Take actions in env
        _, reward, done, _ = env.step(actions)

        # For next timestep
        discounted_return += (args.discount ** timesteps) * reward
        timesteps += 1

        if done:
            return discounted_return


def train(agents, env, log, tb_writer, args):
    # Initialize the return table
    table = np.zeros((args.row, args.row), dtype=np.float32)

    for iteration in range(RUNS):
        for row0 in range(args.row):
            for row1 in range(args.row):
                discounted_return = process_one_traj(agents, row0, row1, env, args)
                table[row0, row1] += discounted_return

    # Normalize
    table /= float(RUNS)

    return table
