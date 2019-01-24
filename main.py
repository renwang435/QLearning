import matplotlib.pyplot as plt
import numpy as np

from GridWorld import GridWorld
from QAgent import QAgent

if __name__ == '__main__':
    num_states = 46
    num_actions = 4
    eps = 0.65
    step_size = 0.1
    discount_rate = 0.95
    rand_seed = 42

    agent = QAgent(num_states=num_states,
                   num_actions=num_actions,
                   eps=eps,
                   step_size=step_size,
                   discount_rate=discount_rate,
                   seed=rand_seed)

    # Note we define 0 to be the bottom left of the world, and 46 to the top right
    # States increase first as we move from left to right, and then when we move from bottom to top
    height = 6
    length = 9
    world_size = num_states

    # Define the rewards for every state in the world
    reward_grid = np.zeros((world_size,))
    reward_grid[-1] = 1

    # Define the obstacles
    # obstacleA is what is used for the first 1000 steps, obstacleB is used for the rest
    # We define obstacleA as 1 and obstacleB as 0
    obstacleA = 1
    obstacleB = 0

    # Define the initial state and the final state
    begin = 3
    end = world_size - 1

    initial_world = GridWorld(height=height,
                              length=length,
                              world_size=world_size,
                              reward_grid=reward_grid,
                              obstacle_type=obstacleA,
                              begin=begin,
                              end=end)

    changed_world = GridWorld(height=height,
                              length=length,
                              world_size=world_size,
                              reward_grid=reward_grid,
                              obstacle_type=obstacleB,
                              begin=begin,
                              end=end)

    num_episodes = 5000
    num_iter_per_episode = 5000
    all_rewards = np.zeros((num_episodes,))

    world = initial_world
    for i in range(num_episodes):
        if i > 1000:
            world = changed_world

        world.reset()
        reached_end = False
        next_action = agent.reset(begin_state=begin)
        for j in range(num_iter_per_episode):
            next_state, reward, reached_end = world.move(next_action)

            next_action = agent.interaction(next_state, reward)
            if reached_end:
                break

        final_policy, final_reward = agent.get_policy_and_rewards()
        all_rewards[i] = np.average(final_reward)

    plt.plot(np.arange(num_episodes), all_rewards)
    # plt.show()
    plt.savefig('qlearning_eps_0.65.png')