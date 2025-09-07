# model_ai.py
import numpy as np
import torch
from enum import Enum

# Map your Enum to ints for encoding
INT_BLACK, INT_WHITE, INT_EMPTY = 1, -1, 0


def encode_board(board, current_player):
    m = np.zeros((8, 8), dtype=np.int8)
    for r in range(8):
        for c in range(8):
            if board[r][c].name == "BLACK":
                m[r, c] = INT_BLACK
            elif board[r][c].name == "WHITE":
                m[r, c] = INT_WHITE
    b = (m == INT_BLACK).astype(np.float32)
    w = (m == INT_WHITE).astype(np.float32)
    side = np.full_like(b, 1.0 if current_player.name == "BLACK" else 0.0)
    return np.stack([b, w, side], axis=0)  # [3,8,8]


class MCTSSelector:
    def __init__(self, model, sims=200, cpuct=1.5, device="cpu"):
        self.model = model.eval().to(device)
        self.sims = sims
        self.cpuct = cpuct
        self.device = device

    @torch.no_grad()
    def policy_value(self, obs):
        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        logits, v = self.model(x)
        p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return p

    def pick(self, game):
        # Build legal mask over 65 actions
        legal_mask = np.zeros(65, dtype=np.float32)
        for r, c in game.get_valid_moves(game.current_player):
            legal_mask[r * 8 + c] = 1.0
        if legal_mask.sum() == 0:
            return None  # pass

        obs = encode_board(game.board, game.current_player)
        p = self.policy_value(obs) * legal_mask
        if p.sum() <= 0:
            # fallback if net is untrained
            p = legal_mask / legal_mask.sum()
        else:
            p /= p.sum()

        # One-step selection using policy (fast for UI). For stronger play, plug full MCTS here.
        a = int(np.argmax(p))
        return divmod(a, 8)  # (row, col)


class ModelAi:
    def __init__(self, model, sims=200, device="cpu"):
        self.sel = MCTSSelector(model, sims=sims, device=device)

    def get_move(self, board, current_player):
        # `board` is the same object held by the game; we access game state in pick()
        # So this adapter will be set on the ReversiGame instance directly
        return self.sel.pick(self.game_ref)
