import unittest

import numpy as np

from GridWorld import GridWorld
from QAgent import QAgent


class TestQAgent(unittest.TestCase):

    def setUp(self):
        self._agent = QAgent(num_states=46,
                             num_actions=4,
                             eps=0.75,
                             step_size=0.1,
                             discount_rate=0.95)
        self._world = GridWorld(world_size=46,
                                reward_grid=np.array((46,)),
                                obstacles=np.zeros((8,)),
                                begin=3,
                                end=45)

    # Test that initial reward is 0
    def test_initial_reward(self):
        pass

    # Test that the initial action returned is valid
    def test_initial_action(self):
        pass

    # Test that the next state from the initial state is a valid one
    def test_next_state(self):
        pass

    # Test that maximum reward possible is 1, and minimum reward is 0
    def test_rewards_possible(self):
        pass

    # This is the key test; idea is to average over first 10 episodes, and then first 50 and finally
    # first 100 --> show that average episodic reward is increasing
    def test_learning(self):
        pass

    # For sake of completeness
    def tearDown(self):
        pass



if __name__ == '__main__':
    pass
