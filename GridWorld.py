import numpy as np

class GridWorld():

    def __init__(self,
                 height,
                 length,
                 world_size,
                 reward_grid,
                 obstacle_type,
                 begin,
                 end):

        self._height = height
        self._length = length
        self._world_size = world_size
        self._reward_grid = reward_grid
        self._obstacle_type = obstacle_type
        self._begin = begin
        self._end = end

    def reset(self):
        self._agent_pos = self._begin

    # 0 --> move right
    # 1 --> move up
    # 2 --> move left
    # 3 --> move down
    def _action_to_index(self, action):
        if action == 0:
            return 1
        elif action == 1:
            return self._length
        elif action == 2:
            return -1
        elif action == 3:
            return -1 * self._length
        else:
            raise NotImplementedError

    def _check_valid_move(self, action):
        # Hypothesis an update
        next_possible_move = self._agent_pos + self._action_to_index(action)

        # Now check that move is within top and bottom world bounds
        if (next_possible_move < 0 or next_possible_move >= self._world_size):
            return False

        # Checking left and right boundaries is trickier, need to take into account current state
        if (self._agent_pos % self._length == 0 and action == 2):
            return False
        elif ((self._agent_pos + 1) % self._length == 0 and action == 0):
            return False

        # Check that move does not result in obstacle collision
        if self._obstacle_type:
            # We are considering obstacleA
            if (self._agent_pos in np.arange(9, 17) and action == 1):
                return False
            elif (self._agent_pos in np.arange(27, 35) and action == 3):
                return False
            elif (self._agent_pos == 26 and action == 2):
                return False
        else:
            # We are considering obstacleB
            if (self._agent_pos in np.arange(10, 18) and action == 1):
                return False
            elif (self._agent_pos in np.arange(28, 36) and action == 3):
                return False
            elif (self._agent_pos == 18 and action == 0):
                return False

        return True

    def move(self, action):
        if self._check_valid_move(action=action):
            # Make the move
            self._agent_pos += self._action_to_index(action)

            # Retrieve reward
            reward = self._reward_grid[self._agent_pos]

            # Check if we have reached the end
            if self._agent_pos == self._end:
                reached_end = True
            else:
                reached_end = False

            return self._agent_pos, reward, reached_end

        else:
            # Invalid move, agent stays where it is and receives 0 reward
            return self._agent_pos, 0, False


