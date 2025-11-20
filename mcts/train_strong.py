#!/usr/bin/env python3

import os
import subprocess
import sys

def run_training():
    """Run training with optimized parameters for stronger AI"""

    cmd = [
        sys.executable, "train_loop.py",
        "--games-per-iter", "128",
        "--train-steps", "1500",
        "--batch-size", "512",
        "--mcts-sims", "400",
        "--temp-moves", "15",
        "--iters", "200",
        "--lr", "2e-4",
        "--replay-cap", "500000"
    ]

    print("Starting enhanced training with parameters:")
    print("- Model: 128 channels, 12 residual blocks")
    print("- Games per iteration: 128")
    print("- Training steps: 1500")
    print("- MCTS simulations: 400")
    print("- Temperature moves: 15")
    print("- Learning rate: 2e-4")
    print("- Replay buffer: 500k samples")
    print()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return False

    return True

if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    success = run_training()
    if success:
        print("\nTraining completed! Test your improved AI with:")
        print("python reversi.py --model checkpoints/best.pt --sims 400")
    else:
        print("\nTraining failed. Check error messages above.")
