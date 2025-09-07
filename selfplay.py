import numpy as np

from env_rl import OthelloEnv
from mcts import MCTS, softmax_t


def play_game(model, sims=200, temp_moves=10):
    env = OthelloEnv()
    obs, _ = env.reset()
    mcts = MCTS(model, sims=sims)
    history = []
    t = 0

    while not env.terminal():
        visits = mcts.run(env, temp_moves=temp_moves)
        T = 1.0 if t < temp_moves else 1e-3
        pi = softmax_t(np.log(visits + 1e-8), T)
        history.append((env.obs(), pi, env._player))
        a = np.random.choice(65, p=pi)
        env.step(a)
        t += 1

    z = env.outcome()
    data = []

    for x, pi, side in history:
        data.append((x, pi, z if side == 1 else -z))

    return data
