import os, time, random, argparse, torch
import numpy as np

from model import ReversiNet
from selfplay import play_game
from train import Replay, train_step, save_ckpt, load_ckpt
from device_utils import get_device, print_device_info


def generate_selfplay(model, games, sims, temp_moves, device="cpu"):
    buf = []
    for _ in range(games):
        buf.extend(play_game(model, sims=sims, temp_moves=temp_moves, device=device))
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

    agree = (probs.argmax(dim=1) == pis.argmax(dim=1)).float().mean().item()

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

    p.add_argument("--games-per-iter", type=int, default=128)
    p.add_argument("--train-steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--replay-cap", type=int, default=500_000)
    p.add_argument("--mcts-sims", type=int, default=400)
    p.add_argument("--temp-moves", type=int, default=15)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to use (auto=auto-detect, mps=Apple Silicon, cuda=NVIDIA GPU, cpu=CPU)",
    )
    p.add_argument(
        "--device-info",
        action="store_true",
        help="Print detailed device information and exit",
    )
    args = p.parse_args()

    if args.device_info:
        print_device_info()
        return

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_str = args.device if args.device and args.device != "auto" else None
    device = get_device(preferred_device=device_str, verbose=True)

    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    elif device.type == "mps":
        torch.mps.manual_seed(args.seed)

    model = ReversiNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.9)

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

        data = generate_selfplay(
            model, args.games_per_iter, args.mcts_sims, args.temp_moves, device=device
        )
        replay.add_many(data)

        losses = []

        for _ in range(args.train_steps):
            if len(replay.buf) < args.batch_size:
                break

            batch = replay.sample(args.batch_size)
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
            "policy_top1_agree": agree,
            "policy_kl": kl,
        }

        ckpt_path = os.path.join(args.outdir, f"run-{it:05d}.pt")
        best_path = os.path.join(args.outdir, "best.pt")
        latest_path = os.path.join(args.outdir, "latest.pt")

        save_ckpt(ckpt_path, model, opt, replay, stats=stats)
        save_ckpt(latest_path, model, opt, replay, stats=stats)
        save_ckpt(best_path, model, opt, replay, stats=stats)

        print(
            f"iter {it}  selfplay {len(data)} pos  replay {len(replay.buf)}  loss {avg_loss:.4f}  lr {opt.param_groups[0]['lr']:.2e}  time {dt:.1f}s"
        )

        scheduler.step()


if __name__ == "__main__":
    main()
