import numpy as np
import matplotlib.pyplot as plt

total_timesteps = 0


def montecarlo(agents, row0, row1, env, log, tb_writer, args):
    global total_timesteps

    env.reset(rows=[row0, row1])
    discounted_return = 0.
    
    while True:
        # Select actions
        actions = []
        for agent in agents:
            action = agent.select_stochastic_action(total_timesteps)
            actions.append(action)
            
        # Take actions in env
        _, reward, done, _ = env.step(actions)
        discounted_return += (args.discount ** total_timesteps) * reward

        # For next timestep
        total_timesteps += 1

        if done:
            return discounted_return


def train(agents, env, log, tb_writer, args):
    global total_timesteps

    answer = np.zeros((args.row, args.row), dtype=np.float32)

    iterations = 100
    for iteration in range(iterations):
        for row0 in range(args.row):
            for row1 in range(args.row):
                total_timesteps = 0
                discounted_return = montecarlo(agents, row0, row1, env, log, tb_writer, args)

                answer[row0, row1] += discounted_return

    answer /= float(iterations)

    fig, ax = plt.subplots()

    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(answer, cmap='seismic')

    # for key, item in answer.items():
    #     key_split = key.split("_")
    #     row = int(key_split[0])
    #     col = int(key_split[1])

    #     ax.text(row, col, '{:0.1f}'.format(item), ha='center', va='center')
    for (i, j), z in np.ndenumerate(answer):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    plt.show()
