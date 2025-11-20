# reversi-py

## Play

With no training:

```
python reversi.py
```

After lots of training:

```
python reversi.py --model checkpoints/best.pt
```

## Bots and eval

Heuristic vs random (alternating colors):

```
python eval_bots.py --games 50
```

## SB3 training

MaskablePPO with MlpPolicy and action masking:

```
python train_sb3.py --total-timesteps 500000 --n-envs 8 --checkpoints checkpoints/sb3 --logdir logs/sb3
```

Evaluate a trained agent vs heuristic or random:

```
python eval_agent_sb3.py --model checkpoints/sb3/final_model.zip --games 50 --opponent heuristic
```

## Train

```
python train_loop.py \
  --games-per-iter 64 \
  --train-steps 1000 \
  --batch-size 512 \
  --mcts-sims 200
```

## Preview

![Reversi](https://raw.githubusercontent.com/gdonald/reversi-py/refs/heads/main/reversi.gif)
