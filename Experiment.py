import numpy as np
from Tilecoding import TileCoder
from Helper import FuncApproxPlot, MSECurvePlot
import gymnasium as gym


class FuncApproxAgent:
    def __init__(self, tiles_per_dim, lims, tilings, alpha=0.1):
        self.tiles_per_dim = tiles_per_dim
        self.tilings = tilings
        self.T = TileCoder(tiles_per_dim, lims, tilings)
        self.w = np.zeros(self.T.n_tiles)
        self.alpha = alpha / tilings

    def get(self, params):
        tiles = self.T[params]
        pred = self.w[tiles].sum()
        return pred

    def update(self, params, target):
        tiles = self.T[params]
        pred = self.w[tiles].sum()
        self.w[tiles] += self.alpha * (target - pred)


class TiledQLearningAgent:
    def __init__(self, n_actions, tiles_per_dim, lims, tilings, epsilon, alpha=0.1, gamma=1):
        self.n_actions = n_actions
        self.tiles_per_dim = tiles_per_dim
        self.tilings = tilings
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.T = TileCoder(tiles_per_dim, lims, tilings)
        self.w_vectors = np.zeros((self.T.n_tiles, n_actions))  # Initialize Q-table with zeros
        pass

    def select_action(self, state):  # ϵ-greedy policy
        tiles = self.T[state]
        best_action = np.argmax(self.w_vectors[tiles].sum(axis=0))
        if np.random.random() > self.epsilon:
            action = best_action
        else:
            action = np.random.choice([x for x in range(self.n_actions) if x != best_action])
        return action

    def update(self, state, action, reward, next_state):  # Q-Learning update equation
        tiles_state = self.T[state]
        tiles_next_state = self.T[next_state]
        best_next_action = np.argmax(self.w_vectors[tiles_next_state].sum(axis=0))
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_delta = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_delta
        pass

############################
# sin(x*2pi) approxamation #
############################


def experiment_1():
    def target_function(x):
        return np.sin(x*2*np.pi)

    lims = [(0, 1)]

    # Single tiling of 20 tiles (20 total)
    tiles_per_dim = [20]
    tilings = 1

    agent1 = FuncApproxAgent(tiles_per_dim, lims, tilings)

    # Two tilings of 10 tiles (20 total)
    tiles_per_dim = [10]
    tilings = 2

    agent2 = FuncApproxAgent(tiles_per_dim, lims, tilings)

    agents = (agent1, agent2)
    agents_eval = [{} for _ in agents]

    batch_size = 200
    batch_number = 3
    x_eval = np.arange(lims[0][0], lims[0][1], 0.01)
    target_eval = target_function(x_eval)
    for batch in range(batch_number):
        for _ in range(batch_size):
            x = lims[0][0] + np.random.rand() * lims[0][1] - lims[0][0]
            target = target_function(x)
            for agent in agents:
                agent.update(x, target)
        plot = FuncApproxPlot(title=f"Learned Approximation after {(batch+1) * batch_size} samples")
        plot.add_curve(x_eval, target_eval, label="target")
        for agent, eval in zip(agents, agents_eval):
            eval[batch] = np.zeros(len(x_eval))
            for i in range(len(x_eval)):
                pred = agent.get(x_eval[i])
                eval[batch][i] = pred
            plot.add_curve(
                x_eval, eval[batch],
                label=f"tiles: {agent.tiles_per_dim[0]}, tilings: {agent.tilings}, n_tiles: {agent.T.n_tiles}")
        plot.save(f"batch_{batch}.png")

#######################################
# sin(x*2pi)+cos(y*2pi) approxamation #
#######################################


def experiment_2():
    def target_function(x, y):
        return np.sin(x*2*np.pi)+np.cos(y*2*np.pi)

    lims = [(0, 1), (0, 1)]

    # Single tiling of 20 tiles per dim (21*21=441 tiles total)
    tiles_per_dim = [20, 20]
    tilings = 1

    agent1 = FuncApproxAgent(tiles_per_dim, lims, tilings)

    # Two tilings of 10 tiles (11*11*2=242 total)
    tiles_per_dim = [10, 10]
    tilings = 2

    agent2 = FuncApproxAgent(tiles_per_dim, lims, tilings)

    # Four tilings of 5 tiles (6*6*4=144 total)
    tiles_per_dim = [5, 5]
    tilings = 4

    agent3 = FuncApproxAgent(tiles_per_dim, lims, tilings)

    # Two tilings of 14 tiles (15*15*2=450 total)
    tiles_per_dim = [14, 14]
    tilings = 2

    agent4 = FuncApproxAgent(tiles_per_dim, lims, tilings)

    agents = (agent1, agent2, agent3, agent4)
    agents_eval_rmse = [[] for _ in agents]

    batch_size = 100
    batch_number = 140
    batch_timesteps = [i*batch_size for i in range(1, batch_number+1)]
    x_eval = np.arange(lims[0][0], lims[0][1], 0.01)
    y_eval = np.arange(lims[1][0], lims[1][1], 0.01)
    for batch in range(batch_number):
        # Training for the function approxamation using random samples
        for _ in range(batch_size):
            x = lims[0][0] + np.random.rand() * lims[0][1] - lims[0][0]
            y = lims[1][0] + np.random.rand() * lims[1][1] - lims[1][0]
            target = target_function(x, y)
            for agent in agents:
                agent.update((x, y), target)
        # Evaluating the function approxamation after every batch
        batch_mse = {agent: 0 for agent in agents}
        for x in x_eval:
            for y in y_eval:
                target = target_function(x, y)
                for agent in agents:
                    pred = agent.get((x, y))
                    batch_mse[agent] += (target - pred) ** 2
        for agent, eval_rmse in zip(agents, agents_eval_rmse):
            eval_rmse.append(np.sqrt(batch_mse[agent] / (len(x_eval) * len(y_eval))))
    # Plot mse learning curves
    plot = MSECurvePlot("RMSE Learning Curve for 2D Function Approxamation")
    for agent, eval_rmse in zip(agents, agents_eval_rmse):
        plot.add_curve(batch_timesteps, eval_rmse,
                       label=f"tiles: {agent.tiles_per_dim[0]}, tilings: {agent.tilings}, n_tiles: {agent.T.n_tiles}")
    plot.save("2d_mse.png")


def experiment_3():
    env = gym.make("Acrobot-v1")
    print(np.array([env.observation_space.low, env.observation_space.high]).T)
    print(env.action_space.n)


if __name__ == '__main__':
    # experiment_1()
    # experiment_2()
    experiment_3()
