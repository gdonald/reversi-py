import argparse
import os
import json
import csv
import re
from typing import Optional, Tuple

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from bots import RandomBot, CornerAwareMobilityBot
from sb3_env_factory import make_vec_env
from env_reversi import ReversiEnv
from reversi import Player, ReversiGame

# Default training hyperparameters (easy to tweak in one place)
DEFAULT_TOTAL_TIMESTEPS = 10_000_000
DEFAULT_N_ENVS = 24
DEFAULT_LOGDIR = "logs/sb3"
DEFAULT_CHECKPOINT_DIR = "checkpoints/sb3"
DEFAULT_SAVE_FREQ = 100_000
DEFAULT_EVAL_FREQ = 40_000
DEFAULT_EVAL_GAMES = 50
DEFAULT_BOT_MIX_PROB = 0.8
DEFAULT_BOT_MIX_OPP = "heuristic"
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_LR = 5e-5
DEFAULT_N_STEPS = 4096
DEFAULT_BATCH_SIZE = 8192
DEFAULT_ENT_COEF = 0.003
DEFAULT_CLIP_RANGE = 0.15
DEFAULT_N_EPOCHS = 4
DEFAULT_TARGET_KL = 0.01
TB_LOG_NAME = "ppo_reversi"


class EvalVsBotCallback(BaseCallback):
    """
    Periodically evaluate the policy against a fixed bot and snapshot the best model.
    """

    def __init__(
        self,
        eval_freq: int,
        games: int,
        opponent: str,
        best_path: str,
        log_path: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.games = games
        self.opponent_name = opponent
        self.best_path = best_path
        self.best_winrate: float = -1.0
        self.log_path = log_path

        if opponent == "random":
            self.bot = RandomBot()
        elif opponent == "heuristic":
            self.bot = CornerAwareMobilityBot()
        else:
            raise ValueError(f"Unknown opponent '{opponent}'")

    def _predict_move(self, env: ReversiEnv) -> Optional[tuple]:
        obs = env._encode_board()
        mask = env._legal_mask()
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)
        if action == env.board_size * env.board_size:
            return None
        return divmod(action, env.board_size)

    def _play_one(self, agent_as: Player) -> Player:
        game = ReversiGame(ai=None)
        env = ReversiEnv()
        env.game = game
        current = Player.BLACK
        while True:
            env.current_player = current
            moves = game.get_valid_moves(current)
            if moves:
                if current == agent_as:
                    move = self._predict_move(env)
                else:
                    move = self.bot.select_move(game, current)
                if move is not None:
                    game.make_move(move[0], move[1], current)
            else:
                other = Player.WHITE if current == Player.BLACK else Player.BLACK
                if not game.get_valid_moves(other):
                    break
            current = Player.WHITE if current == Player.BLACK else Player.BLACK
        return game.get_winner()

    def _evaluate(self) -> float:
        wins = 0
        total = max(1, self.games)
        for i in range(self.games):
            agent_as = Player.BLACK if i % 2 == 0 else Player.WHITE
            winner = self._play_one(agent_as)
            if winner == agent_as:
                wins += 1
        return wins / total

    def _on_step(self) -> bool:
        if self.n_calls == 1 or self.num_timesteps % self.eval_freq != 0:
            return True
        winrate = self._evaluate()
        if self.verbose:
            print(f"[Eval] games={self.games} vs {self.opponent_name} winrate={winrate:.2%}")
        # append to CSV log
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timesteps", "winrate", "opponent", "games"])
            writer.writerow([self.num_timesteps, winrate, self.opponent_name, self.games])
        if winrate > self.best_winrate:
            self.best_winrate = winrate
            os.makedirs(os.path.dirname(self.best_path), exist_ok=True)
            self.model.save(self.best_path)
            if self.verbose:
                print(f"[Eval] new best model saved to {self.best_path}")
        return True


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Pick the most recent checkpoint with a bias toward best/final, otherwise highest step count."""
    if not os.path.isdir(ckpt_dir):
        return None

    best: Tuple[int, int, float, str] | None = None  # (priority, steps, mtime, path)
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".zip"):
            continue
        path = os.path.join(ckpt_dir, fname)
        priority = 1
        steps = 0
        if fname == "best_model.zip":
            priority = 3
        elif fname == "final_model.zip":
            priority = 2
        else:
            m = re.search(r"_(\d+)_steps\\.zip$", fname)
            if m:
                steps = int(m.group(1))
        stat = os.stat(path)
        key = (priority, steps, stat.st_mtime, path)
        if best is None or key > best:
            best = key
    return best[3] if best else None


def parse_args():
    p = argparse.ArgumentParser(description="Train MaskablePPO on Reversi")
    p.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    p.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    p.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    p.add_argument("--checkpoints", type=str, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ, help="Timesteps between checkpoints")
    p.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ, help="Timesteps between eval runs")
    p.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES, help="Games per eval window")
    p.add_argument(
        "--bot-mix-prob",
        type=float,
        default=DEFAULT_BOT_MIX_PROB,
        help="Probability an env episode pits the agent against a fixed heuristic opponent instead of pure self-play",
    )
    p.add_argument(
        "--bot-mix-opponent",
        type=str,
        default=DEFAULT_BOT_MIX_OPP,
        choices=["heuristic"],
        help="Opponent used when bot-mix is enabled",
    )
    p.add_argument(
        "--eval-opponent",
        type=str,
        default="heuristic",
        choices=["random", "heuristic"],
        help="Opponent used for periodic eval",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-model", type=str, default=None, help="Path to an existing MaskablePPO .zip to resume training")
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--gae-lambda", type=float, default=DEFAULT_GAE_LAMBDA)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS, help="Rollout length per env before update")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Minibatch size for PPO updates")
    p.add_argument("--ent-coef", type=float, default=DEFAULT_ENT_COEF)
    p.add_argument("--clip-range", type=float, default=DEFAULT_CLIP_RANGE)
    p.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS, help="PPO epochs per update")
    p.add_argument(
        "--target-kl",
        type=float,
        default=DEFAULT_TARGET_KL,
        help="Target KL for early stopping PPO updates (lower values dampen unstable updates)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for training (e.g., 'cpu', 'cuda', 'mps', or 'auto')",
    )
    p.add_argument(
        "--no-subproc",
        action="store_true",
        help="Disable SubprocVecEnv and use in-process DummyVecEnv (single core)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Default per-process threading to 1 to avoid OpenMP/MKL oversubscription when using many envs.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)
    # Persist run configuration for reproducibility
    with open(os.path.join(args.logdir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    load_path = args.load_model
    if not load_path:
        load_path = find_latest_checkpoint(args.checkpoints)
        if load_path:
            print(f"Auto-resuming from checkpoint: {load_path}")
        else:
            print(f"No checkpoint found in {args.checkpoints}; starting fresh.")

    env = make_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        bot_mix_prob=args.bot_mix_prob,
        bot_name=args.bot_mix_opponent,
        use_subproc=not args.no_subproc,
    )

    if load_path:
        model = MaskablePPO.load(
            load_path,
            env=env,
            device=args.device,
            tensorboard_log=args.logdir,
        )
        model.set_random_seed(args.seed)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.logdir,
            seed=args.seed,
            device=args.device,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            n_epochs=args.n_epochs,
            target_kl=args.target_kl,
        )
        model_name = TB_LOG_NAME
        reset_steps = True

    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoints,
        name_prefix="maskable_ppo",
        save_replay_buffer=False,
    )

    eval_cb = EvalVsBotCallback(
        eval_freq=args.eval_freq,
        games=args.eval_games,
        opponent=args.eval_opponent,
        best_path=os.path.join(args.checkpoints, "best_model.zip"),
        log_path=os.path.join(args.logdir, "eval_history.csv"),
        verbose=1,
    )

    if load_path:
        model_name = TB_LOG_NAME
        reset_steps = False

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[ckpt_cb, eval_cb],
        tb_log_name=model_name,
        reset_num_timesteps=reset_steps,
    )

    final_path = os.path.join(args.checkpoints, "final_model.zip")
    model.save(final_path)
    print(f"Training complete. Saved final model to {final_path}")


if __name__ == "__main__":
    main()
