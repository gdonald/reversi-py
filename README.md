# reversi-py

Self-contained Reversi environment plus Stable Baselines3 training harness with action masking.

## Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are pinned; Python 3.10+ recommended. MPS/CUDA selection is handled by SB3 (`--device auto`).

## Quickstart

Play the console game:

```
python reversi.py
```

Train a MaskablePPO agent (8 environments, checkpoints + TensorBoard logs):

```
python train_sb3.py --total-timesteps 500000 --n-envs 8 --checkpoints checkpoints/sb3 --logdir logs/sb3
```

Monitor training curves:

```
tensorboard --logdir logs/sb3
```

Evaluate a trained agent vs heuristic or random bots:

```
python eval_agent_sb3.py --model checkpoints/sb3/final_model.zip --games 50 --opponents heuristic random
python eval_agent_sb3.py --model checkpoints/sb3/best_model.zip --games 10 --render-games 1
```

Run heuristic vs random baseline:

```
python eval_bots.py --games 50
```

## Notes

- Action masking prevents illegal moves; pass action is explicit and two passes end a game.
- Training config is saved to `logs/sb3/training_config.json` for reproducibility.
- Checkpoints are written every `--save-freq`; best model (by periodic eval) is saved to `checkpoints/sb3/best_model.zip`.

## Preview

![Reversi](https://raw.githubusercontent.com/gdonald/reversi-py/refs/heads/main/reversi.gif)
