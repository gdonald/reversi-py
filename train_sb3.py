import argparse
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

from sb3_env_factory import make_vec_env


def parse_args():
    p = argparse.ArgumentParser(description="Train MaskablePPO on Reversi")
    p.add_argument("--total-timesteps", type=int, default=250_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--logdir", type=str, default="logs/sb3")
    p.add_argument("--checkpoints", type=str, default="checkpoints/sb3")
    p.add_argument("--save-freq", type=int, default=50_000, help="Timesteps between checkpoints")
    p.add_argument("--seed", type=int, default=42)
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

    env = make_vec_env(n_envs=args.n_envs, seed=args.seed)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
        device=args.device,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoints,
        name_prefix="maskable_ppo",
        save_replay_buffer=False,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb)

    final_path = os.path.join(args.checkpoints, "final_model.zip")
    model.save(final_path)
    print(f"Training complete. Saved final model to {final_path}")


if __name__ == "__main__":
    main()
