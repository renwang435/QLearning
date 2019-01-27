import numpy as np
from QAgent import QAgent

class AsyncQAgent(QAgent):

    # TODO: Implementation which does not access hidden variables in master agent
    def __init__(self, agent, eps):
        self._num_states = agent._num_states
        self._num_actions = agent._num_actions
        self._eps = eps
        self._step_size = agent._step_size
        self._discount_rate = agent._discount_rate
        self._seed = agent._seed

        self.copy_policy(agent)
        self.zero_grad()

    def copy_policy(self, agent):
        self._Q = agent._Q

    def zero_grad(self):
        self._grads = np.zeros_like(self._Q)

    def _accumulate_grads(self, curr_state, next_action, next_state, reward):
        max_reward_possible = self._Q[next_state, self._get_best_action(next_state)]
        self._grads[curr_state, next_action] += (reward +
                                                 self._discount_rate * max_reward_possible -
                                                 self._Q[curr_state, next_action])

    def interaction(self, next_state, reward):
        self._accumulate_grads(self._curr_state, self._next_action, next_state, reward)

        # Some parameters to keep track of for updating the master agent
        state, action = self._curr_state, self._next_action

        # Determine the next action to take
        self._get_next_action(next_state)

        # Update our current state
        self._curr_state = next_state

        return self._next_action, state, action