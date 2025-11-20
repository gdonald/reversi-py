from typing import Callable, Optional

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env_reversi import ReversiEnv


def make_env(seed: Optional[int] = None) -> Callable[[], ReversiEnv]:
    """Return a thunk that creates a fresh ReversiEnv, optionally seeded."""

    def _init():
        env = ReversiEnv()
        env.reset(seed=seed)
        return env

    return _init


def make_vec_env(n_envs: int = 1, seed: Optional[int] = None):
    """
    Create a vectorized Reversi environment wrapped with VecMonitor.
    Requires stable-baselines3 to be installed.
    """
    env_fns = [make_env(seed=seed + i if seed is not None else None) for i in range(n_envs)]
    vec = DummyVecEnv(env_fns)
    return VecMonitor(vec)
