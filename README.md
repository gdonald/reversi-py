# reversi-py

## Train

Using my MBP M4 (cpu)

```
python train_loop.py \
  --games-per-iter 64 \
  --train-steps 1000 \
  --batch-size 512 \
  --mcts-sims 200
```

## Play

After lots of training...

```
python reversi.py --model checkpoints/best.pt
```
