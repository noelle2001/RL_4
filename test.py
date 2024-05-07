import numpy as np
from references.tilecoding.tilecoding import TileCoder
from Helper import FuncApproxPlot

############################
# Sin(x*2pi) approxamation #
############################


def target_function(x):
    return np.sin(x*2*np.pi)


class Agent:
    def __init__(self, tiles_per_dim, lims, tilings, alpha=0.1):
        self.tiles_per_dim = tiles_per_dim
        self.tilings = tilings
        self.T = TileCoder(tiles_per_dim, lims, tilings)
        self.w = np.zeros(self.T.n_tiles)
        self.alpha = alpha / tilings

    def get(self, x):
        tiles = self.T[x]
        pred = self.w[tiles].sum()
        return pred

    def update(self, x, target):
        tiles = self.T[x]
        pred = self.w[tiles].sum()
        self.w[tiles] += self.alpha * (target - pred)


lims = [(0, 1)]


# Single tiling of 20 tiles
tiles_per_dim = [20]
tilings = 1

agent1 = Agent(tiles_per_dim, lims, tilings)
agent1_eval = {}

# Two tiles of 10 tiles (20 total)
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
        plot.add_curve(x_eval, eval[batch], label=f"tiles: {agent.tiles_per_dim[0]}, tilings: {agent.tilings}")
    plot.save(f"batch_{batch}.png")
