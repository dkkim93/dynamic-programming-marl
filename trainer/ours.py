import itertools
import numpy as np


def get_value(agents, env, states, table, args, timesteps):
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
                    agents, env, states=next_states, table=table, args=args, timesteps=timesteps + 1)
                next_value *= args.discount

        new_value += action_prob * (reward + next_value)

    return new_value


def train(agents, env, log, tb_writer, args):
    # Initialize the return table
    table = np.zeros((args.row, args.row), dtype=np.float32)

    # Get all possible states: possible locations of all agents
    state0 = [row for row in range(args.row)]
    all_states = list(itertools.product(state0, state0))

    # Get all possible actions
    global all_actions
    action0 = [action for action in range(env.action_space.n)]
    all_actions = list(itertools.product(action0, action0))

    # Start to do policy evaluation
    prev_delta = 0.
    while True:
        # Reset delta which is used to terminate while loop
        delta = 0.

        # For each state (i.e., possible locations of all agents), 
        for states in all_states:
            value = table[states[0], states[1]]

            # Overwrite with new value
            new_value = get_value(
                agents=agents, env=env, states=states, table=table, args=args, timesteps=0)
            table[states[0], states[1]] = new_value

            delta = max(delta, abs(value - new_value))

        print("delta:", delta)
        if abs(prev_delta - delta) == 0.:
            return table

        prev_delta = delta
