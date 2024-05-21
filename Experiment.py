import numpy as np
from Tilecoding import TileCoder
from Helper import FuncApproxPlot, MSECurvePlot


class Agent:
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

    agent1 = Agent(tiles_per_dim, lims, tilings)
    agent1_eval = {}

    # Two tilings of 10 tiles (20 total)
    tiles_per_dim = [10]
    tilings = 2

    agent2 = Agent(tiles_per_dim, lims, tilings)
    agent2_eval = {}

    batch_size = 200
    batch_number = 3
    x_eval = np.arange(lims[0][0], lims[0][1], 0.01)
    target_eval = target_function(x_eval)
    for batch in range(batch_number):
        for _ in range(batch_size):
            x = lims[0][0] + np.random.rand() * lims[0][1] - lims[0][0]
            target = target_function(x)
            for agent in (agent1, agent2):
                agent.update(x, target)
        plot = FuncApproxPlot(title=f"Learned Approximation after {(batch+1) * batch_size} samples")
        plot.add_curve(x_eval, target_eval, label="target")
        for agent, eval in ((agent1, agent1_eval), (agent2, agent2_eval)):
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

    agent1 = Agent(tiles_per_dim, lims, tilings)

    # Two tilings of 10 tiles (11*11*2=242 total)
    tiles_per_dim = [10, 10]
    tilings = 2

    agent2 = Agent(tiles_per_dim, lims, tilings)

    # Two tilings of 14 tiles (15*15*2=450 total)
    tiles_per_dim = [14, 14]
    tilings = 2

    agent3 = Agent(tiles_per_dim, lims, tilings)

    # Three tilings of 8 tiles (9*9*3=243 total)
    tiles_per_dim = [8, 8]
    tilings = 3

    agent4 = Agent(tiles_per_dim, lims, tilings)

    # Three tilings of 11 tiles (12*12*3=432 total)
    tiles_per_dim = [11, 11]
    tilings = 3

    agent5 = Agent(tiles_per_dim, lims, tilings)

    agents = (agent1, agent2, agent3, agent4, agent5)
    agents_eval_rmse = [[] for _ in agents]

    batch_size = 100
    batch_number = 140
    batch_timesteps = [i*batch_size for i in range(1, batch_number+1)]
    x_eval = y_eval = np.arange(lims[0][0], lims[0][1], 0.01)
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
            eval_rmse.append(np.sqrt(batch_mse[agent] / batch_size))
    # Plot mse learning curves
    plot = MSECurvePlot("RMSE Learning Curve for 2D Function Approxamation")
    for agent, eval_rmse in zip(agents, agents_eval_rmse):
        plot.add_curve(batch_timesteps, eval_rmse,
                       label=f"tiles: {agent.tiles_per_dim[0]}, tilings: {agent.tilings}, n_tiles: {agent.T.n_tiles}")
    plot.save("2d_mse.png")


if __name__ == '__main__':
    # experiment_1()
    experiment_2()
