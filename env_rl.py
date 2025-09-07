import numpy as np
from copy import deepcopy
from reversi import ReversiGame, Player

BLACK, WHITE, EMPTY = 1, -1, 0


def encode_board(board, player_to_move):
    b = (board == BLACK).astype(np.float32)
    w = (board == WHITE).astype(np.float32)
    side = np.full_like(b, 1.0 if player_to_move == BLACK else 0.0)
    x = np.stack([b, w, side], axis=0)
    return x


class ReversiEnv:
    def __init__(self):
        self.g = ReversiGame(ai=None)
        self._player = BLACK

    def reset(self):
        self.g.board = [[Player.EMPTY for _ in range(8)] for _ in range(8)]
        self.g.current_player = Player.BLACK
        self.g.game_over = False
        self.g.initialize_board()
        self._player = BLACK
        return self.obs(), self.legal_mask()

    def obs(self):
        m = np.zeros((8, 8), dtype=np.int8)

        for r in range(8):
            for c in range(8):
                if self.g.board[r][c] == Player.BLACK:
                    m[r, c] = BLACK
                elif self.g.board[r][c] == Player.WHITE:
                    m[r, c] = WHITE

        return encode_board(m, self._player)

    def legal_moves_list(self):
        pl = Player.BLACK if self._player == BLACK else Player.WHITE
        return self.g.get_valid_moves(pl)

    def legal_mask(self):
        mask = np.zeros(65, dtype=np.float32)
        for r, c in self.legal_moves_list():
            mask[r * 8 + c] = 1.0
        if mask.sum() == 0:
            mask[64] = 1.0
        return mask

    def terminal(self):
        return self.g.is_game_over()

    def outcome(self):
        w = self.g.get_winner()
        if w is None:
            return 0
        return 1 if w == Player.BLACK else -1

    def step(self, a):
        if a != 64:
            r, c = divmod(a, 8)
            pl = Player.BLACK if self._player == BLACK else Player.WHITE
            ok = self.g.make_move(r, c, pl)

            if not ok:
                raise ValueError("Illegal move")

        self._player = -self._player
        self.g.current_player = Player.BLACK if self._player == BLACK else Player.WHITE
        return self.obs(), self.legal_mask(), self.terminal()
