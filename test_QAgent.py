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

    # Test reset methods perform as expected
    def test_reset(self):
        print("Testing that agent and world reset properly...")
        _ = self._agent.reset(begin_state=10)
        self.assertTrue(self._agent._curr_state == 10)

        self.assertTrue(self._world._agent_pos == 3)
        _, _, _, = self._world.move(action=0)
        self.assertTrue(self._world._agent_pos == 4)
        self._world.reset()
        self.assertTrue(self._world._agent_pos == 3)

        print('OK')

    # Test that initial reward is 0
    def test_initial_reward(self):
        print("Testing that initial reward returned is always 0...")
        self._world.reset()
        next_action = self._agent.reset(begin_state=self._world._begin)
        _, reward, _ = self._world.move(next_action)

        self.assertTrue(reward == 0)

        print("OK")

    # Test that the initial action returned is valid
    def test_initial_action(self):
        print("Testing that action returned is valid...")
        next_action = self._agent.reset(begin_state=self._world._begin)
        self.assertTrue(next_action in np.arange(0, 4))

        print("OK")

    # Test that the next state from the initial state is a valid one
    def test_next_state(self):
        self.assertTrue(self._world._agent_pos == 3)
        _, _, _ = self._world.move(action=3)
        self.assertTrue(self._world._agent_pos == 3)
        _, _, _ = self._world.move(action=1)
        self.assertTrue(self._world._agent_pos == 12)

    # Test that maximum reward possible is 1, and minimum reward is 0
    def test_rewards_possible(self):
        print("Testing that maximum reward over all episodes is 1, and minimum reward is 0...")
        for i in range(1000):
            self._world.reset()
            next_action = self._agent.reset(begin_state=self._world._begin)
            for j in range(500):
                next_state, reward, reached_end = self._world.move(next_action)

                next_action = self._agent.interaction(next_state, reward)
                if reached_end:
                    break

            _, final_reward = self._agent.get_policy_and_rewards()
            self.assertGreaterEqual(np.min(final_reward), 0)
            self.assertLessEqual(np.max(final_reward), 1)

        print("OK")

    # This is the key test; idea is to average over first 10 episodes, and then first 50 and finally
    # first 100 --> show that average episodic reward is increasing
    def test_learning(self):
        print("Testing that average episodic reward increases over training time...")
        curr_avg_reward = -1
        for i in range(100):
            self._world.reset()
            next_action = self._agent.reset(begin_state=self._world._begin)
            for j in range(500):
                next_state, reward, reached_end = self._world.move(next_action)

                next_action = self._agent.interaction(next_state, reward)
                if reached_end:
                    break

            final_policy, final_reward = self._agent.get_policy_and_rewards()
            all_rewards = np.average(final_reward)
            if i in [10, 50, 100]:
                self.assertGreaterEqual(all_rewards, curr_avg_reward)
                curr_avg_reward = all_rewards

        print("OK")

    # For sake of completeness
    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
