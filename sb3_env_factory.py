import random
from functools import partial
from typing import Callable, Optional

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv

from bots import CornerAwareMobilityBot
from env_reversi import ReversiEnv
from reversi import Player, ReversiGame


def make_heuristic_opponent(bot_name: str):
    if bot_name == "heuristic":
        return CornerAwareMobilityBot()
    raise ValueError(f"Unknown bot '{bot_name}'")


def _copy_board(board):
    return [[cell for cell in row] for row in board]


class HeuristicVsSelfPlayEnv(ReversiEnv):
    """
    Env that alternates between self-play (agent vs itself) and a fixed heuristic
    opponent, depending on a probability. When playing vs heuristic, the current
    player is the learning agent; the opponent move is applied automatically.
    """

    def __init__(self, opponent_prob: float, opponent_name: str, seed: Optional[int] = None):
        super().__init__()
        self.opponent_prob = opponent_prob
        self.opponent = make_heuristic_opponent(opponent_name)
        self.rng = random.Random(seed)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.playing_vs_bot = self.rng.random() < self.opponent_prob
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated or not self.playing_vs_bot:
            return obs, reward, terminated, truncated, info

        # If opponent_turn, have the bot make a move (or pass) before returning.
        if not terminated:
            bot_move = self.opponent.select_move(self.game, self.current_player)
            if bot_move is None:
                # Pass
                self.pass_count += 1
                if self.pass_count >= 2 or self.game.is_game_over():
                    terminated = True
                else:
                    info = {"action_mask": self._legal_mask()}
                    return self._encode_board(), reward, terminated, truncated, info
            else:
                self.pass_count = 0
                self.game.make_move(bot_move[0], bot_move[1], self.current_player)

            if self.game.is_game_over():
                terminated = True
                winner = self.game.get_winner()
                if winner is None:
                    reward = 0.0
                else:
                    reward = 1.0 if winner == self.current_player else -1.0
                info = {"action_mask": self._legal_mask()}
                return self._encode_board(), reward, terminated, truncated, info

            # swap back to learning agent
            self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
            obs = self._encode_board()
            info = {"action_mask": self._legal_mask()}
        return obs, reward, terminated, truncated, info


def make_env(seed: Optional[int] = None, bot_mix_prob: float = 0.0, bot_name: str = "heuristic") -> Callable[[], ReversiEnv]:
    """Return a thunk that creates a fresh ReversiEnv (or bot-mix variant), optionally seeded."""

    def _init():
        if bot_mix_prob > 0.0:
            env = HeuristicVsSelfPlayEnv(opponent_prob=bot_mix_prob, opponent_name=bot_name, seed=seed)
        else:
            env = ReversiEnv()
        env.reset(seed=seed)
        return env

    return _init


def make_vec_env(
    n_envs: int = 1,
    seed: Optional[int] = None,
    bot_mix_prob: float = 0.0,
    bot_name: str = "heuristic",
    use_subproc: bool = True,
):
    """
    Create a vectorized Reversi environment wrapped with VecMonitor.
    Requires stable-baselines3 to be installed.
    """
    env_fns = [
        make_env(
            seed=seed + i if seed is not None else None,
            bot_mix_prob=bot_mix_prob,
            bot_name=bot_name,
        )
        for i in range(n_envs)
    ]
    if use_subproc and n_envs > 1:
        vec = SubprocVecEnv(env_fns)
    else:
        vec = DummyVecEnv(env_fns)
    return VecMonitor(vec)
