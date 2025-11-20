import argparse
import os
import json
from typing import Optional

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from bots import RandomBot, CornerAwareMobilityBot
from sb3_env_factory import make_vec_env
from env_reversi import ReversiEnv
from reversi import Player, ReversiGame


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
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.games = games
        self.opponent_name = opponent
        self.best_path = best_path
        self.best_winrate: float = -1.0

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
        if winrate > self.best_winrate:
            self.best_winrate = winrate
            os.makedirs(os.path.dirname(self.best_path), exist_ok=True)
            self.model.save(self.best_path)
            if self.verbose:
                print(f"[Eval] new best model saved to {self.best_path}")
        return True


def parse_args():
    p = argparse.ArgumentParser(description="Train MaskablePPO on Reversi")
    p.add_argument("--total-timesteps", type=int, default=250_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--logdir", type=str, default="logs/sb3")
    p.add_argument("--checkpoints", type=str, default="checkpoints/sb3")
    p.add_argument("--save-freq", type=int, default=50_000, help="Timesteps between checkpoints")
    p.add_argument("--eval-freq", type=int, default=50_000, help="Timesteps between eval runs")
    p.add_argument("--eval-games", type=int, default=10, help="Games per eval window")
    p.add_argument(
        "--eval-opponent",
        type=str,
        default="heuristic",
        choices=["random", "heuristic"],
        help="Opponent used for periodic eval",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024, help="Rollout length per env before update")
    p.add_argument("--batch-size", type=int, default=2048, help="Minibatch size for PPO updates")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per update")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for training (e.g., 'cpu', 'cuda', 'mps', or 'auto')",
    )
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)
    # Persist run configuration for reproducibility
    with open(os.path.join(args.logdir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    env = make_vec_env(n_envs=args.n_envs, seed=args.seed)

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
    )

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
        verbose=1,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=[ckpt_cb, eval_cb])

    final_path = os.path.join(args.checkpoints, "final_model.zip")
    model.save(final_path)
    print(f"Training complete. Saved final model to {final_path}")


if __name__ == "__main__":
    main()
