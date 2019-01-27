import random
import threading
import numpy as np

from GridWorld import GridWorld
from QAgent import QAgent
from AsyncQAgent import AsyncQAgent

import matplotlib.pyplot as plt

TMAX = 100000
T = 0
I_Async = 5

num_states = 46
num_actions = 4
eps = np.arange(70, 90).astype(np.float) / 100
step_size = 0.1
discount_rate = 0.95
rand_seed = 42

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

all_rewards = []
master_agent = QAgent(num_states=num_states,
               num_actions=num_actions,
               eps=eps,
               step_size=step_size,
               discount_rate=discount_rate,
               seed=rand_seed)

# all_threads = np.arange(5)
all_threads = [1]

def actorLearner(num):
    # We use global shared O parameter vector
    # We use global shared Otarget parameter vector
    # We use global shared counter T, and TMAX constant
    global TMAX, T, master_agent, all_rewards

    # epsilon_index = random.randrange(len(eps))
    # epsilon = eps[epsilon_index]
    epsilon = eps[9]

    # Initialize network params
    thread_agent = AsyncQAgent(master_agent, epsilon)
    thread_world = GridWorld(height=height,
                              length=length,
                              world_size=world_size,
                              reward_grid=reward_grid,
                              obstacle_type=1,
                              begin=begin,
                              end=end)

    print("THREAD %d STARTING...EXPLORATION POLICY => INITIAL_EPSILON: %.3f" % (num, epsilon))

    # Initialize thread step counter
    t = 0
    thread_world.reset()
    next_action = thread_agent.reset(begin_state=begin)
    while T < TMAX:
        # Run the selected action and observe next state and reward
        next_state, reward, reached_end = thread_world.move(next_action)

        # Accumulate gradients
        next_action, curr_state, curr_action = thread_agent.interaction(next_state, reward)

        # Update the old values
        T += 1
        t += 1

        # Update our local QAgent policy
        thread_agent.copy_policy(master_agent)

        # Update the master agent policy if needed
        if t % I_Async == 0 or reached_end:
            # Perform asynchronous update of master agent policy
            master_agent._Q[curr_state, curr_action] += thread_agent._step_size * thread_agent._grads[curr_state, curr_action]
            final_policy, final_reward = master_agent.get_policy_and_rewards()
            all_rewards.append(np.average(final_reward))
            # Clear gradients
            thread_agent.zero_grad()
            if reached_end:
                # final_policy, final_reward = master_agent.get_policy_and_rewards()
                # all_rewards.append(np.average(final_reward))
                thread_world.reset()
                next_action = thread_agent.reset(begin_state=begin)

if __name__ == "__main__":
    # Start n concurrent actor threads
    for num_threads in all_threads:
        all_rewards.clear()
        lock = threading.Lock()
        threads = list()
        for i in range(num_threads):
            t = threading.Thread(target=actorLearner, args=(i,))
            threads.append(t)

        # Start all threads
        for i, x in enumerate(threads):
            x.start()

        # Wait for all of them to finish
        for x in threads:
            x.join()

        plt.plot(np.arange(len(all_rewards)), all_rewards, label='num_threads: %d' % num_threads)
        # plt.savefig('qlearning_eps_0.65.png')
    plt.legend()
    plt.show()