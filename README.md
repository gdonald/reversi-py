# reversi-py

Self-contained Reversi environment plus Stable Baselines3 (MaskablePPO) training harness with action masking and heuristic sparring.

## Install

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. Default device is CPU; override with `--device mps|cuda|auto` if you want GPU. MPS is often slower than CPU for this setup.

## Scripts

- Play the classic console game (no RL): `python reversi.py`
- Play the console game vs the SB3 agent: `python reversi.py --sb3-model checkpoints/sb3/best_model.zip` (add `--device cpu|mps|cuda` to force)
- Train agent (defaults below): `python train_sb3.py`
- Evaluate a checkpoint vs bots (get winrate): `python eval_agent_sb3.py --model checkpoints/sb3/best_model.zip --games 50 --opponents heuristic random`
- Head-to-head in a simple prompt (coords like `d3`/`pass`): `python play_vs_agent.py --model checkpoints/sb3/best_model.zip`
- Plot eval winrate history from training logs (uses `logs/sb3/eval_history.csv`): `python plot_eval.py --logdir logs/sb3`
- Baseline bots vs each other: `python eval_bots.py --games 50`

## Training defaults (tweak in `train_sb3.py`)

```
total_timesteps = 10_000_000
n_envs = 16
bot_mix_prob = 0.8        # fraction of episodes vs heuristic opponent
eval_freq = 40_000        # timesteps between evals
eval_games = 50
learning_rate = 5e-5
ent_coef = 0.003
n_steps = 3072
batch_size = 4096
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.15
n_epochs = 4
checkpoints = checkpoints/sb3
logdir = logs/sb3
```

 Overrides (examples):
- Auto-resume is on by default (uses `best_model.zip` > `final_model.zip` > newest step checkpoint)
- Continue from a specific checkpoint: `--load-model checkpoints/sb3/best_model.zip`
- Change sparring mix: `--bot-mix-prob 0.5`
- Force single-process vec env (if SubprocVecEnv blocked): `--no-subproc`
- Adjust device: `--device mps|cuda|cpu|auto` (default is `cpu`)

Outputs:
- Checkpoints saved every `--save-freq` to `checkpoints/sb3/` (`best_model.zip` from eval; `final_model.zip` at end).
- Logs and `eval_history.csv` in `logs/sb3/`
  - Winrate tracking: use `python eval_agent_sb3.py ...` for a one-off, or run `python plot_eval.py --logdir logs/sb3` to graph the eval callback history.
  - TensorBoard: `tensorboard --logdir logs/sb3` (opens http://localhost:6006) to see PPO losses/entropy/etc.

## Preview

![Reversi](https://raw.githubusercontent.com/gdonald/reversi-py/refs/heads/main/reversi.gif)
