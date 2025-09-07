import random, torch, torch.nn.functional as F
import numpy as np
import os, tempfile, torch

from torch.optim import Adam
from model import OthelloNet
from selfplay import play_game


class Replay:
    def __init__(self, cap=200_000):
        self.buf, self.cap = [], cap

    def add_many(self, items):
        self.buf.extend(items)
        if len(self.buf) > self.cap:
            self.buf = self.buf[-self.cap :]

    def sample(self, n):
        return random.sample(self.buf, n)


def train_step(model, opt, batch, device="cpu"):
    # Convert lists -> contiguous NumPy -> torch
    xs = torch.from_numpy(
        np.stack([b[0] for b in batch], axis=0).astype(np.float32)
    ).to(
        device
    )  # [B,3,8,8]
    pis = torch.from_numpy(
        np.stack([b[1] for b in batch], axis=0).astype(np.float32)
    ).to(
        device
    )  # [B,65]
    zs = torch.from_numpy(np.asarray([b[2] for b in batch], dtype=np.float32)).to(
        device
    )  # [B]

    p_logits, v = model(xs)
    ce = F.cross_entropy(p_logits, pis.argmax(dim=1))
    mse = F.mse_loss(v.squeeze(1), zs)
    l2 = 1e-4 * sum((p.pow(2).sum() for p in model.parameters()))
    loss = ce + mse + l2

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())


def save_ckpt(path, model, opt, replay, stats=None):
    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "replay": replay.buf,
        "stats": stats or {},
    }
    d = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(dir=d, delete=False) as tmp:
        torch.save(payload, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def load_ckpt(path, model, opt):
    d = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(d["model"])
    try:
        opt.load_state_dict(d["opt"])
    except Exception:
        pass
    r = Replay()
    r.buf = d.get("replay", [])
    return r
