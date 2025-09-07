import os, time, random, argparse, torch
import numpy as np

from model import OthelloNet
from selfplay import play_game
from train import Replay, train_step, save_ckpt, load_ckpt


def generate_selfplay(model, games, sims, temp_moves):
    buf = []
    for _ in range(games):
        buf.extend(play_game(model, sims=sims, temp_moves=temp_moves))
    return buf


@torch.no_grad()
def policy_agreement(model, replay, n=512, device="cpu"):
    n = min(n, len(replay.buf))
    if n == 0:
        return None, None
    batch = random.sample(replay.buf, n)
    xs = torch.from_numpy(np.stack([b[0] for b in batch]).astype(np.float32)).to(device)
    pis = torch.from_numpy(np.stack([b[1] for b in batch]).astype(np.float32)).to(
        device
    )
    logits, _ = model(xs)
    probs = torch.softmax(logits, dim=-1)

    # top-1 agreement with MCTS
    agree = (probs.argmax(dim=1) == pis.argmax(dim=1)).float().mean().item()

    # KL(MCTS || net) over support of MCTS (mask zeros to avoid nan)
    eps = 1e-8
    kl = (
        (pis * (torch.log(pis + eps) - torch.log(probs + eps))).sum(dim=1).mean().item()
    )

    return float(agree), float(kl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="checkpoints")

    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint to resume; defaults to checkpoints/latest.pt if present",
    )

    p.add_argument("--games-per-iter", type=int, default=64)
    p.add_argument("--train-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--replay-cap", type=int, default=200_000)
    p.add_argument("--mcts-sims", type=int, default=200)
    p.add_argument("--temp-moves", type=int, default=10)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OthelloNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.resume is None:
        cand = os.path.join(args.outdir, "latest.pt")
        if os.path.exists(cand):
            args.resume = cand

    if args.resume:
        replay = load_ckpt(args.resume, model, opt)
        print(f"Resumed from {args.resume} with {len(replay.buf)} samples")
    else:
        replay = Replay(cap=args.replay_cap)

    best_path = os.path.join(args.outdir, "best.pt")

    for it in range(1, args.iters + 1):
        t0 = time.time()
        # 1 Self play
        data = generate_selfplay(
            model, args.games_per_iter, args.mcts_sims, args.temp_moves
        )
        replay.add_many(data)

        # 2 Training
        losses = []
        for _ in range(args.train_steps):
            if len(replay.buf) < args.batch_size:
                break
            batch = replay.sample(args.batch_size)

            # loss = train_step(model, opt, batch)
            loss = train_step(model, opt, batch, device=device)

            losses.append(loss)

        dt = time.time() - t0
        avg_loss = sum(losses) / max(1, len(losses))

        agree, kl = policy_agreement(model, replay, n=512, device=device)

        stats = {
            "iter": it,
            "loss": float(avg_loss),
            "replay_size": len(replay.buf),
            "selfplay_positions": len(data),
            "secs": float(dt),
            "policy_top1_agree": agree,  # goes up
            "policy_kl": kl,  # goes down
        }

        # 3 Save checkpoint
        ckpt_path = os.path.join(args.outdir, f"run-{it:05d}.pt")
        best_path = os.path.join(args.outdir, "best.pt")
        latest_path = os.path.join(args.outdir, "latest.pt")

        save_ckpt(ckpt_path, model, opt, replay, stats=stats)
        save_ckpt(latest_path, model, opt, replay, stats=stats)
        save_ckpt(best_path, model, opt, replay, stats=stats)

        print(
            f"iter {it}  selfplay {len(data)} pos  replay {len(replay.buf)}  loss {avg_loss:.4f}  time {dt:.1f}s"
        )


if __name__ == "__main__":
    main()
