from tbparse import SummaryReader
import pandas as pd

# Point at your SB3 log dir (e.g., logs/sb3)
reader = SummaryReader("logs/sb3", pivot=False)
df = reader.scalars  # columns: wall_time, step, run, tag, value

# Filter to the scalar tags you care about (optional)
tags = [
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
]
df = df[df["tag"].isin(tags)]

df.to_csv("logs/sb3/scalars_export.csv", index=False)
print("Wrote", df.shape, "to logs/sb3/scalars_export.csv")
