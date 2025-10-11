import numpy as np
import torch
from types import SimpleNamespace


BLACK, WHITE = 1, -1
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def encode_board(board, current_player):
    m = np.zeros((8, 8), dtype=np.int8)
    for r in range(8):
        for c in range(8):
            n = board[r][c].name
            if n == "BLACK":
                m[r, c] = BLACK
            elif n == "WHITE":
                m[r, c] = WHITE
    b = (m == BLACK).astype(np.float32)
    w = (m == WHITE).astype(np.float32)
    side = np.full_like(b, 1.0 if current_player.name == "BLACK" else 0.0)

    corners = np.zeros((8, 8), dtype=np.float32)
    corners[0, 0] = corners[0, 7] = corners[7, 0] = corners[7, 7] = 1.0

    edges = np.zeros((8, 8), dtype=np.float32)
    edges[0, :] = edges[7, :] = edges[:, 0] = edges[:, 7] = 1.0
    edges[0, 0] = edges[0, 7] = edges[7, 0] = edges[7, 7] = 0.0

    return np.stack([b, w, side, corners, edges], axis=0)


def legal_mask_from(board, current_player):
    mask = np.zeros(65, dtype=np.float32)
    me = current_player.name
    opp = "WHITE" if me == "BLACK" else "BLACK"
    for r in range(8):
        for c in range(8):
            if board[r][c].name != "EMPTY":
                continue
            ok = False
            for dr, dc in DIRS:
                i, j = r + dr, c + dc
                seen = False
                while 0 <= i < 8 and 0 <= j < 8:
                    n = board[i][j].name
                    if n == opp:
                        seen = True
                    elif n == me and seen:
                        ok = True
                        break
                    else:
                        break
                    i += dr
                    j += dc
                if ok:
                    break
            if ok:
                mask[r * 8 + c] = 1.0
    if mask.sum() == 0:
        mask[64] = 1.0
    return mask


def _apply_move(board, r, c, current_player):
    me = current_player.name
    from reversi import Player

    opp = Player.WHITE if me == "BLACK" else Player.BLACK
    board[r][c] = current_player
    for dr, dc in DIRS:
        i, j = r + dr, c + dc
        flips = []
        while 0 <= i < 8 and 0 <= j < 8:
            if board[i][j] == opp:
                flips.append((i, j))
            elif board[i][j] == current_player:
                for u, v in flips:
                    board[u][v] = current_player
                break
            else:
                break
            i += dr
            j += dc


class ModelAi:
    def __init__(self, model, sims=0, device="cpu", temperature=0.1, add_noise=True):
        self.model = model.eval().to(device)
        self.device = device
        self.sims = sims
        self.temperature = temperature
        self.add_noise = add_noise

        self.use_mcts = sims and sims > 0
        if self.use_mcts:
            from mcts import MCTS

            self.mcts = MCTS(self.model, sims=sims, device=device)

    @torch.no_grad()
    def get_move(self, board, current_player):

        if not self.use_mcts:
            obs = encode_board(board, current_player)
            x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            logits, _ = self.model(x)
            p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            mask = legal_mask_from(board, current_player)
            p = p * mask
            p = p / p.sum() if p.sum() > 0 else mask / max(1.0, mask.sum())
            a = int(np.argmax(p))
            if a == 64:
                return None
            return divmod(a, 8)

        from reversi import Player

        state = {"board": [row[:] for row in board], "player": current_player}

        def obs():
            return encode_board(state["board"], state["player"])

        def legal_mask():
            return legal_mask_from(state["board"], state["player"])

        def step(a):
            if a == 64:
                state["player"] = (
                    Player.WHITE if state["player"] == Player.BLACK else Player.BLACK
                )
                return
            r, c = divmod(a, 8)
            _apply_move(state["board"], r, c, state["player"])
            state["player"] = (
                Player.WHITE if state["player"] == Player.BLACK else Player.BLACK
            )

        def terminal():
            if legal_mask().sum() > 0:
                return False
            other = Player.WHITE if state["player"] == Player.BLACK else Player.BLACK
            saved = state["player"]
            state["player"] = other
            has = legal_mask().sum() > 0
            state["player"] = saved
            return not has

        env = SimpleNamespace(
            obs=obs, legal_mask=legal_mask, step=step, terminal=terminal
        )

        visits = self.mcts.run(env, temp_moves=0, add_noise=self.add_noise)

        if self.temperature > 0:
            from mcts import softmax_t

            probs = softmax_t(np.log(visits + 1e-10), self.temperature)

            if self.temperature > 0.5:
                a = int(np.random.choice(65, p=probs))
            else:
                a = int(np.argmax(probs))
        else:
            a = int(visits.argmax())

        if a == 64:
            return None

        return divmod(a, 8)
