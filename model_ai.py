import numpy as np
import torch
from enum import Enum

BLACK, WHITE, EMPTY = 1, -1, 0


def encode_board(board, current_player):
    m = np.zeros((8, 8), dtype=np.int8)

    for r in range(8):
        for c in range(8):
            if board[r][c].name == "BLACK":
                m[r, c] = BLACK
            elif board[r][c].name == "WHITE":
                m[r, c] = WHITE

    b = (m == BLACK).astype(np.float32)
    w = (m == WHITE).astype(np.float32)
    side = np.full_like(b, 1.0 if current_player.name == "BLACK" else 0.0)

    return np.stack([b, w, side], axis=0)


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
        legal_mask = np.zeros(65, dtype=np.float32)

        for r, c in game.get_valid_moves(game.current_player):
            legal_mask[r * 8 + c] = 1.0

        if legal_mask.sum() == 0:
            return None

        obs = encode_board(game.board, game.current_player)
        p = self.policy_value(obs) * legal_mask

        if p.sum() <= 0:
            p = legal_mask / legal_mask.sum()
        else:
            p /= p.sum()

        a = int(np.argmax(p))
        return divmod(a, 8)


class ModelAi:
    def __init__(self, model, sims=200, device="cpu"):
        self.sel = MCTSSelector(model, sims=sims, device=device)

    def get_move(self, board, current_player):
        return self.sel.pick(self.game_ref)
