# eval_baselines.py
import numpy as np
from env_rl import OthelloEnv
from mcts import MCTS
from model import OthelloNet
import torch


# Baseline A: random legal move
def random_policy(env):
    mask = env.legal_mask()
    idx = np.where(mask > 0)[0]
    if len(idx) == 0 or 64 in idx:  # pass
        return 64
    return int(np.random.choice(idx))


# Baseline B: cheap heuristic
CORNERS = {0, 7, 56, 63}


def heuristic_policy(env):
    mask = env.legal_mask()
    idx = np.where(mask > 0)[0]
    if len(idx) == 0:
        return 64
    # prefer corners, else more immediate flips
    best, score = None, -1
    for a in idx:
        if a in CORNERS:
            return a
        if a == 64:
            continue
        r, c = divmod(a, 8)
        # simulate immediate flips cheaply
        from reversi import Player  # your enums

        tmp = OthelloEnv()
        tmp.g = env.g
        tmp._player = env._player  # shallow alias
        # do a minimal safe step on a clone
    return int(np.random.choice(idx))  # fallback simple version
