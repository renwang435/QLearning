import unittest

import numpy as np

from AsyncQAgent import AsyncQAgent
from GridWorld import GridWorld
from QAgent import QAgent


class TestAsyncQAgent(unittest.TestCase):

    def setUp(self):
        self._agent = QAgent(num_states=46,
                             num_actions=4,
                             eps=0.75,
                             step_size=0.1,
                             discount_rate=0.95,
                             seed=42)
        self._world = GridWorld(height=6,
                                length=9,
                                world_size=46,
                                reward_grid=np.zeros((46,)),
                                obstacle_type=1,
                                begin=3,
                                end=45)
        self._world.reset()

    # Test that we can correctly copy the Q function across different agents
    def test_copy_policy(self):
        print('Testing that we can copy the Q function across different agents...')
        num_states = self._agent._num_states
        num_actions = self._agent._num_states
        self._agent._Q = np.random.rand(num_states, num_actions)
        another_agent = AsyncQAgent(agent=self._agent,
                                    eps=0.8)
        self.assertTrue(np.array_equal(self._agent._Q, another_agent._Q))

        print('OK')

    # Test that we can successfully zero out accumulation of gradients
    def test_zero_grad(self):
        print('Testing that we can copy zero out accumulated values in thread agents...')
        another_agent = AsyncQAgent(agent=self._agent,
                                    eps=0.8)
        num_states = self._agent._num_states
        num_actions = self._agent._num_states

        another_agent._grads = np.random.rand(num_states, num_actions)
        self.assertTrue(np.count_nonzero(another_agent._grads) != 0)
        another_agent.zero_grad()
        self.assertTrue(np.count_nonzero(another_agent._grads) == 0)

        print('OK')

    # This is the key test; idea is to evaluate rewards over 1 thread, 2 threads, and then finally 5 threads
    # --> show that average episodic reward increases with an increasing number of agent/threads
    def test_learning(self):
        pass


    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()