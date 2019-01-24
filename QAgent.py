import random

import numpy as np


class QAgent:

    def __init__(self,
                 num_states,
                 num_actions,
                 eps,
                 step_size,
                 discount_rate,
                 seed):

        self._num_states = num_states
        self._num_actions = num_actions
        self._eps = eps
        self._step_size = step_size
        self._discount_rate = discount_rate
        self._seed = seed

        self._Q = np.zeros((self._num_states, self._num_actions))

    def reset(self, begin_state):
        '''Set the initial state and return the learner's first action'''
        self._get_next_action(begin_state)
        self._curr_state = begin_state

        return self._next_action

    def _get_next_action(self, next_state):
        if random.random() <= self._eps:
            self._next_action = random.randint(0, self._num_actions - 1)
        else:
            self._next_action = self._get_best_action(next_state)

    def _get_best_action(self, next_state):
        return np.argmax(self._Q[next_state, :])

    def _update_Q_val(self, curr_state, next_action, next_state, reward):
        max_reward_possible = self._Q[next_state, self._get_best_action(next_state)]
        self._Q[curr_state, next_action] += self._step_size * (reward +
                                                               self._discount_rate * max_reward_possible -
                                                               self._Q[curr_state, next_action])

    def get_policy_and_rewards(self):
        policy = np.argmax(self._Q, axis=1)
        rewards = np.max(self._Q, axis=1)

        return policy, rewards

    def interaction(self, next_state, reward):
        self._update_Q_val(self._curr_state, self._next_action, next_state, reward)

        # Determine the next action to take
        self._get_next_action(next_state)

        # Update our current state
        self._curr_state = next_state

        return self._next_action

