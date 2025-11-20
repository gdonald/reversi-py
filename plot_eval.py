import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_history(path):
    rows = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"No eval history at {path}. Run training with eval callback first.")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "timesteps": int(float(r["timesteps"])),  # handle scientific notation
                    "winrate": float(r["winrate"]),
                    "opponent": r.get("opponent", ""),
                    "games": int(r.get("games", 0) or 0),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description="Plot eval winrate history from eval_history.csv")
    ap.add_argument(
        "--logdir",
        type=str,
        default="logs/sb3",
        help="Directory containing eval_history.csv (written during training)",
    )
    args = ap.parse_args()

    path = os.path.join(args.logdir, "eval_history.csv")
    hist = load_history(path)
    if not hist:
        print("No rows to plot.")
        return

    xs = [r["timesteps"] for r in hist]
    ys = [r["winrate"] * 100 for r in hist]
    opp = hist[0]["opponent"]

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o", label=f"vs {opp}")
    plt.xlabel("Timesteps")
    plt.ylabel("Winrate (%)")
    plt.ylim(0, 100)
    plt.title("Eval winrate over time")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
