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
