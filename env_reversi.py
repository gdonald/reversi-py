import numpy as np
import gymnasium as gym
from gymnasium import spaces

from reversi import ReversiGame, Player


class ReversiEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        self.board_size = 8
        self.action_space = spaces.Discrete(self.board_size * self.board_size + 1)
        # Planes: player stones, opponent stones, side-to-move, corners, edges
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5, self.board_size, self.board_size),
            dtype=np.float32,
        )
        self.game = ReversiGame(ai=None)
        self.current_player = Player.BLACK

    def _encode_board(self):
        m = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.game.board[r][c] == Player.BLACK:
                    m[r, c] = 1
                elif self.game.board[r][c] == Player.WHITE:
                    m[r, c] = -1

        b = (m == 1).astype(np.float32)
        w = (m == -1).astype(np.float32)
        side = np.full_like(b, 1.0 if self.current_player == Player.BLACK else 0.0)

        corners = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        corners[0, 0] = corners[0, -1] = corners[-1, 0] = corners[-1, -1] = 1.0

        edges = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        edges[0, :] = edges[-1, :] = edges[:, 0] = edges[:, -1] = 1.0
        edges[0, 0] = edges[0, -1] = edges[-1, 0] = edges[-1, -1] = 0.0

        return np.stack([b, w, side, corners, edges], axis=0)

    def _legal_mask(self):
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        pl = self.current_player
        for r, c in self.game.get_valid_moves(pl):
            mask[r * self.board_size + c] = 1.0
        if mask.sum() == 0:
            mask[-1] = 1.0  # pass
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = ReversiGame(ai=None)
        self.current_player = Player.BLACK
        return self._encode_board(), {"action_mask": self._legal_mask()}

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.0
        terminated = False
        truncated = False

        mask = self._legal_mask()
        if mask[action] == 0.0:
            # Illegal actions are not allowed; end episode with penalty.
            reward = -1.0
            terminated = True
            info = {"action_mask": mask, "illegal_action": True}
            return self._encode_board(), reward, terminated, truncated, info

        if action != self.board_size * self.board_size:
            r, c = divmod(action, self.board_size)
            self.game.make_move(r, c, self.current_player)
        # pass is no-op on board

        # terminal check before switching
        if self.game.is_game_over():
            terminated = True
            winner = self.game.get_winner()
            if winner == self.current_player:
                reward = 1.0
            elif winner is None:
                reward = 0.0
            else:
                reward = -1.0
            return (
                self._encode_board(),
                reward,
                terminated,
                truncated,
                {"action_mask": self._legal_mask()},
            )

        # swap player and continue
        self.current_player = (
            Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        )
        obs = self._encode_board()
        info = {"action_mask": self._legal_mask()}
        return obs, reward, terminated, truncated, info

    def render(self):
        # Simple ANSI render of board
        rows = []
        for r in range(self.board_size):
            row = []
            for c in range(self.board_size):
                v = self.game.board[r][c]
                if v == Player.BLACK:
                    row.append("B")
                elif v == Player.WHITE:
                    row.append("W")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)
