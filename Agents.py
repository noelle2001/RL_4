import numpy as np
from Tilecoding import TileCoder


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


class TiledSARSAAgent:
    def __init__(self, n_actions, tiles_per_dim, lims, tilings, alpha, gamma):
        self.n_actions = n_actions
        self.tiles_per_dim = tiles_per_dim
        self.tilings = tilings
        self.alpha = alpha / tilings
        self.gamma = gamma
        self.T = TileCoder(tiles_per_dim, lims, tilings)
        self.w_vectors = np.zeros((self.T.n_tiles, n_actions))  # Initialize Q-table with zeros
        pass

    def select_action(self, state, epsilon):  # ϵ-greedy policy
        tiles_state = self.T[state]
        best_action = np.argmax(self.w_vectors[tiles_state].sum(axis=0))
        if np.random.random() > epsilon:
            action = best_action
        else:
            action = np.random.choice([x for x in range(self.n_actions) if x != best_action])
        return action

    def update(self, state, action, reward, next_state, next_action):  # SARSA update equation
        tiles_state = self.T[state]
        tiles_next_state = self.T[next_state]
        td_target = reward + self.gamma * self.w_vectors[tiles_next_state, next_action].sum()
        td_delta = td_target - self.w_vectors[tiles_state, action].sum()
        self.w_vectors[tiles_state, action] += self.alpha * td_delta
        pass
