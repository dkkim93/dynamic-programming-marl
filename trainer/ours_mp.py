import itertools
import multiprocessing as mp
import numpy as np


def get_value(agents, env, states, table, args, timesteps, new_values):
    # Iterate through all possible actions
    new_value = 0.

    for actions in all_actions:
        # Get action probability for a specific action
        action_prob = 1.
        for i_agent, agent in enumerate(agents):
            action_prob *= agent.get_action_prob(actions[i_agent], timesteps)

        # Get next states
        next_states = env.transition_func(states, actions)

        # Get reward
        reward = env.reward_func(states)

        # Get next value
        if reward == 1.:
            next_value = 0.
        else: 
            if timesteps >= args.future_max_timesteps:
                next_value = args.discount * table[next_states[0], next_states[1]]
            else:
                next_value = get_value(
                    agents=agents, env=env, states=next_states, table=table, args=args, 
                    timesteps=timesteps + 1, new_values=None)
                next_value *= args.discount

        new_value += action_prob * (reward + next_value)

    if new_values is not None:
        key = str(states[0]) + "_" + str(states[1])
        new_values[key] = new_value

    return new_value


def train(agents, env, log, tb_writer, args):
    # Initialize the return table
    table = np.zeros((args.row, args.row), dtype=np.float32)
    prev_table = np.zeros((args.row, args.row), dtype=np.float32)

    # Get all possible states: possible locations of all agents
    state0 = [row for row in range(args.row)]
    all_states = list(itertools.product(state0, state0))

    # Get all possible actions
    global all_actions
    action0 = [action for action in range(env.action_space.n)]
    all_actions = list(itertools.product(action0, action0))

    # Start to do policy evaluation
    iteration = 0
    while True:
        # Reset delta which is used to terminate while loop
        delta = 0.

        # For each state (i.e., possible locations of all agents), 
        processes = []
        manager = mp.Manager()
        new_values = manager.dict()
        for states in all_states:
            # Overwrite with new value
            process = mp.Process(
                target=get_value, 
                args=(agents, env, states, table, args, 0, new_values))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        for key, item in new_values.items():
            key = key.split("_")
            table[int(key[0]), int(key[1])] = item

        # Check delta
        delta = sum((table - prev_table).flatten())
        log[args.log_name].info("Delta {} at iteration {}".format(delta, iteration))
        tb_writer.add_scalar("delta", delta, iteration)
        if delta == 0.:
            return table

        # For next iteration
        prev_table = np.copy(table)
        iteration += 1
