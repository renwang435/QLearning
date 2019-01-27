# Basic Reinforcement Learning - Q Learning

# Setup Instructions
1. Environment: Python 2.7.15+ virtual environment
2. Packages: After cloning the repository, navigate to the project directory and run pip install -r requirements.txt

# Part 1
Relevant files: GridWorld.py, QAgent.py, test_QAgent.py, main.py

The purpose of this section is to demonstrate that an agent within a grid with a simple obstacle can learn an optimal strategy to achieve a reward located in an alternate position within the grid. To show that the agent succesfully navigates around this obstacle, we shift the obstacle and demonstrate that the agent can then relearn another strategy to navigate around this new obstacle.

To run the code, activate a virtual environment as specified in 1. and run:
```
python main.py
```
The code will then simulate 5000 episodes of agent learning, with the obstacle shifting position after 1000 episodes. The reward vs. episodes curve is then saved in "qlearning_0.65.png" and is shown below:

![alt text](https://github.com/renwang435/QLearning/blob/master/qlearning_eps_0.65.png)

Above we see a plot of reward (y-axis) and episodes trained for (x_axis). The agent succesfully learns to navigate through the grid around the obstacle in order to obtain a reward; once the obstacle shift right by 1 position after 1000 episodes, we see a sharp decrease in the reward obtained as the agent relearns an alternate strategy to circumvent the obstacle. Subsequently, the agent's reward continues to increase after relearning an optimal heuristic.

# Part 2
Relevant files: GridWorld.py, QAgent.py, AsyncQAgent.py, test_AsyncQAgent.py, async_main.py

The purpose of this section is to demonstrate that agents can be succesfully trained in separate environments, each contributing to a universal policy.

To run the code, activate a virtual environment as specified in 1. and run
```
python main --n_threads=3
```
where the argument following n_threads specifies the number of threads (and thus agents) we want to perform training with. The default value is a single agent, which is analogous to the first part.

![alt text](https://github.com/renwang435/QLearning/blob/master/async_qlearning_1.png)
![alt text](https://github.com/renwang435/QLearning/blob/master/async_qlearning_2.png)
![alt text](https://github.com/renwang435/QLearning/blob/master/async_qlearning_5.png)
![alt text](https://github.com/renwang435/QLearning/blob/master/async_qlearning_10.png)

Above, we see a series of plots of average episodic reward (y-axis) and the number of total times steps for all agents that we have iterated through. As we can see, rewards increase proportionally with the number of threads run concurrently (aka agents contributing to the final policy).

# Part 3
