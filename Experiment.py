import numpy as np
from Helper import FuncApproxPlot, MSECurvePlot, LearningCurvePlot, smooth
import gymnasium as gym
from Agents import FuncApproxAgent, TiledSARSAAgent

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
            x = lims[0][0] + np.random.rand() * (lims[0][1] - lims[0][0])
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

    lims = [(1, 2), (1, 2)]

    # Single tiling of 20 tiles per dim (21*21=441 tiles total)
    tiles_per_dim = [20, 20]
    tilings = 1

    agent1 = FuncApproxAgent(tiles_per_dim, lims, tilings)

    # Two tilings of 10 tiles (11*11*2=242 total)
    tiles_per_dim = [10, 10]
    tilings = 2

    agent2 = FuncApproxAgent(tiles_per_dim, lims, tilings)

    # 10 tilings of 2 tiles (3*3*10=90 total)
    tiles_per_dim = [2, 2]
    tilings = 10

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
            x = lims[0][0] + np.random.rand() * (lims[0][1] - lims[0][0])
            y = lims[1][0] + np.random.rand() * (lims[1][1] - lims[1][0])
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
    plot = MSECurvePlot("RMSE Learning Curve for 2D Function Approximation")
    for agent, eval_rmse in zip(agents, agents_eval_rmse):
        plot.add_curve(batch_timesteps, eval_rmse,
                       label=f"tiles: {agent.tiles_per_dim[0]}, tilings: {agent.tilings}, n_tiles: {agent.T.n_tiles}")
    plot.save("2d_mse.png")

###############################
# Q-Learning with Tile Coding #
###############################


def run_repetitions(n_repetitions, n_episodes, tiles, tilings, epsilon=0.1, alpha=0.1, gamma=1):
    print("Running repetitions with the following settings:")
    print(locals())

    env = gym.make('MountainCar-v0')
    n_actions = env.action_space.n
    lims = np.array([env.observation_space.low, env.observation_space.high]).T
    tiles_per_dim = [tiles] * env.observation_space.shape[0]

    episode_returns = np.zeros((n_repetitions, n_episodes))

    for rep in range(n_repetitions):
        agent = TiledSARSAAgent(n_actions, tiles_per_dim, lims, tilings, alpha, gamma)
        for ep in range(n_episodes):
            s, info = env.reset()
            a = agent.select_action(s, epsilon)
            if ep % 100 == 0:
                print(f"Running repitition {rep+1:2}, Finished {ep:4} episodes", end="\r")
            done = False
            while not done:
                s_next, r, done, trunc, info = env.step(a)  # Simulate environment
                a_next = agent.select_action(s_next, epsilon)
                agent.update(s, a, r, s_next, a_next)
                episode_returns[rep, ep] += r
                s = s_next
                a = a_next

    # Compute average evaluations over all repetitions
    mean_episode_returns = np.mean(episode_returns, axis=0)
    return mean_episode_returns


def experiment_3():
    n_repetitions = 10
    n_episodes = 700
    gamma = 1
    alpha = 0.1
    epsilon = 0.1

    tiling_settings = [(5, 1), (10, 1), (20, 1), (10, 2), (2, 10)]
    episode_returns = []

    smoothing_window = 31

    for tiles, tilings in tiling_settings:
        episode_returns.append(run_repetitions(n_repetitions, n_episodes, tiles, tilings, epsilon, alpha, gamma))

    plot = LearningCurvePlot("Mountain Car SARSA with 1 tilings")
    for (tiles, tilings), episode_return in zip(tiling_settings[:3], episode_returns[:3]):
        plot.add_curve(range(1, n_episodes+1), smooth(episode_return, smoothing_window),
                       label=f"tiles: {tiles}, tilings: {tilings}, n_tiles: {(tiles+1)**2*tilings}")
    plot.save(name=f"mountain_car_sarsa_rep{n_repetitions}_1tiling.png")

    plot = LearningCurvePlot("Mountain Car SARSA with 20 total bins")
    for (tiles, tilings), episode_return in zip(tiling_settings[-3:], episode_returns[-3:]):
        plot.add_curve(range(1, n_episodes+1), smooth(episode_return, smoothing_window),
                       label=f"tiles: {tiles}, tilings: {tilings}, n_tiles: {(tiles+1)**2*tilings}")
    plot.save(name=f"mountain_car_sarsa_rep{n_repetitions}_20bins.png")


if __name__ == '__main__':
    experiment_1()
    experiment_2()
    experiment_3()
