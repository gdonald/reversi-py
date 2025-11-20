import os
import sys
import random
import time
import termios
import tty
import argparse
import torch

from enum import Enum
from model import ReversiNet
import numpy as np

from model_ai import ModelAi
from device_utils import get_device, print_device_info
from sb3_contrib import MaskablePPO

# When run as a script, ensure imports that reference the module name "reversi"
# resolve to this instance (avoids duplicate module leading to mismatched Player enums).
if "reversi" not in sys.modules:
    sys.modules["reversi"] = sys.modules[__name__]


class Ai:
    def __init__(self):
        pass

    def get_move(self, board, current_player):
        valid_moves = self._get_valid_moves(board, current_player)
        if valid_moves:
            return random.choice(valid_moves)
        return None

    def _get_valid_moves(self, board, player):
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self._is_valid_move(board, row, col, player):
                    valid_moves.append((row, col))
        return valid_moves

    def _is_valid_move(self, board, row, col, player):
        if board[row][col] != Player.EMPTY:
            return False

        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        opponent = Player.WHITE if player == Player.BLACK else Player.BLACK

        for dr, dc in directions:
            r, c = row + dr, col + dc
            found_opponent = False

            while 0 <= r < 8 and 0 <= c < 8:
                if board[r][c] == opponent:
                    found_opponent = True
                elif board[r][c] == player and found_opponent:
                    return True
                else:
                    break
                r += dr
                c += dc

        return False


class Player(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


class PlayerCount(Enum):
    ZERO = 0
    ONE = 1


class SB3MaskableAi(Ai):
    def __init__(self, model_path: str, device: str = "auto", debug: bool = False):
        super().__init__()
        # Deferred import to avoid circular dependency at module import time.
        from env_reversi import ReversiEnv

        self.env = ReversiEnv()
        self.model = MaskablePPO.load(model_path, device=device)
        self.debug = debug

    def get_move(self, board, current_player):
        # sync env with current game state (defensive copy to avoid aliasing)
        self.env.game.board = [[cell for cell in row] for row in board]
        self.env.game.current_player = current_player
        self.env.game.game_over = False
        self.env.current_player = current_player
        self.env.pass_count = 0
        obs = self.env._encode_board()
        mask = self.env._legal_mask()
        if self.debug:
            valid = self.env.game.get_valid_moves(current_player)
            # Inspect board types to catch mismatches (e.g., ints instead of Player Enum)
            types = {type(cell).__name__ for row in self.env.game.board for cell in row}
            if not hasattr(self, "_debug_once"):
                counts = {}
                for row in self.env.game.board:
                    for cell in row:
                        counts[cell] = counts.get(cell, 0) + 1
                center = self.env.game.board[3][3], self.env.game.board[3][4]
                print(f"[SB3-debug] counts={counts} center={center}")
                self._debug_once = True
            print(f"[SB3] mask_sum={mask.sum()} valid_moves={valid} cell_types={types}")
        if mask.sum() == 0:
            fallback_moves = self.env.game.get_valid_moves(current_player)
            if fallback_moves:
                return fallback_moves[0]
            return None
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)
        pass_idx = self.env.board_size * self.env.board_size
        if action == pass_idx:
            legal_moves = np.where(mask[:-1] == 1.0)[0]
            if legal_moves.size > 0:
                action = int(legal_moves[0])  # fall back to first legal move
            else:
                return None
        r, c = divmod(action, self.env.board_size)
        return (r, c)


class ReversiGame:
    def __init__(self, ai=None, ai_debug: bool = False):
        self.board = [[Player.EMPTY for _ in range(8)] for _ in range(8)]
        self.current_player = Player.BLACK
        self.cursor_x = 3
        self.cursor_y = 2
        self.game_mode = PlayerCount.ONE
        self.human_player = Player.BLACK
        self.game_over = False
        self.ai = ai if ai is not None else Ai()
        self.ai_debug = ai_debug
        self.simulation_mode = False
        self.initialize_board()

    def initialize_board(self):
        self.board[3][3] = Player.WHITE
        self.board[3][4] = Player.BLACK
        self.board[4][3] = Player.BLACK
        self.board[4][4] = Player.WHITE

    def clear_screen(self):
        os.system("export TERM=linux; clear")
        print("Reversi")

    def display_board(self):
        self.clear_screen()

        print("\nUse WASD or arrow keys to move")
        print("Press enter to place piece\n")
        print(
            f"Current Player: {'BLACK ○' if self.current_player == Player.BLACK else 'WHITE ●'}"
        )

        if self.game_mode == PlayerCount.ZERO:
            print("Game Mode: Computer vs Computer")
        else:
            human_color = "BLACK" if self.human_player == Player.BLACK else "WHITE"
            print(f"Game Mode: Human ({human_color}) vs Computer")

        print()
        print("    A   B   C   D   E   F   G   H")
        print("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")

        for row in range(8):
            print(f"{row + 1} │", end="")

            for col in range(8):
                if row == self.cursor_y and col == self.cursor_x:
                    if self.board[row][col] == Player.EMPTY:
                        print("[·]", end="")
                    elif self.board[row][col] == Player.BLACK:
                        print("[○]", end="")
                    else:
                        print("[●]", end="")
                else:
                    if self.board[row][col] == Player.EMPTY:
                        print("   ", end="")
                    elif self.board[row][col] == Player.BLACK:
                        print(" ○ ", end="")
                    else:
                        print(" ● ", end="")

                if col < 7:
                    print("│", end="")
                else:
                    print("│")

            if row < 7:
                print("  ├───┼───┼───┼───┼───┼───┼───┼───┤")
            else:
                print("  └───┴───┴───┴───┴───┴───┴───┴───┘")

    def is_valid_move(self, row, col, player):
        if self.board[row][col] != Player.EMPTY:
            return False

        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        opponent = Player.WHITE if player == Player.BLACK else Player.BLACK

        for dr, dc in directions:
            r, c = row + dr, col + dc
            found_opponent = False

            while 0 <= r < 8 and 0 <= c < 8:
                if self.board[r][c] == opponent:
                    found_opponent = True
                elif self.board[r][c] == player and found_opponent:
                    return True
                else:
                    break
                r += dr
                c += dc

        return False

    def get_valid_moves(self, player):
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(row, col, player):
                    valid_moves.append((row, col))
        return valid_moves

    def make_move(self, row, col, player):
        if not self.is_valid_move(row, col, player):
            return False

        self.board[row][col] = player
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        opponent = Player.WHITE if player == Player.BLACK else Player.BLACK

        for dr, dc in directions:
            r, c = row + dr, col + dc
            pieces_to_flip = []

            while 0 <= r < 8 and 0 <= c < 8:
                if self.board[r][c] == opponent:
                    pieces_to_flip.append((r, c))
                elif self.board[r][c] == player:
                    for flip_r, flip_c in pieces_to_flip:
                        self.board[flip_r][flip_c] = player
                    break
                else:
                    break
                r += dr
                c += dc

        return True

    def count_pieces(self):
        black_count = sum(row.count(Player.BLACK) for row in self.board)
        white_count = sum(row.count(Player.WHITE) for row in self.board)
        return black_count, white_count

    def is_game_over(self):
        black_moves = self.get_valid_moves(Player.BLACK)
        white_moves = self.get_valid_moves(Player.WHITE)
        return len(black_moves) == 0 and len(white_moves) == 0

    def get_winner(self):
        black_count, white_count = self.count_pieces()
        if black_count > white_count:
            return Player.BLACK
        elif white_count > black_count:
            return Player.WHITE
        else:
            return None

    def play_game(self):
        while not self.game_over:
            self.display_board()

            valid_moves = self.get_valid_moves(self.current_player)

            if not valid_moves:
                opponent = (
                    Player.WHITE
                    if self.current_player == Player.BLACK
                    else Player.BLACK
                )
                opponent_moves = self.get_valid_moves(opponent)

                if not opponent_moves:
                    self.game_over = True
                    break
                else:
                    print(
                        f"\nNo valid moves for {'BLACK' if self.current_player == Player.BLACK else 'WHITE'}. Skipping turn."
                    )
                    if (
                        self.game_mode == PlayerCount.ZERO
                        or self.current_player != self.human_player
                    ):
                        time.sleep(1)
                    else:
                        input("Press Enter to continue...")
                    self.current_player = opponent
                    continue

            if self.game_mode == PlayerCount.ZERO or (
                self.game_mode == PlayerCount.ONE
                and self.current_player != self.human_player
            ):
                move = self.ai.get_move(self.board, self.current_player)
                proposed = move
                row_col = None
                if move and self.is_valid_move(move[0], move[1], self.current_player):
                    row_col = move
                elif valid_moves:
                    # Fallback to first legal move to avoid bogus skips when the AI outputs pass/invalid.
                    row_col = valid_moves[0]

                if self.ai_debug:
                    print(
                        f"[AI] {('BLACK' if self.current_player == Player.BLACK else 'WHITE')} proposed {proposed}; using {row_col}"
                    )

                if row_col:
                    row, col = row_col
                    self.make_move(row, col, self.current_player)
                else:
                    print(
                        f"\nComputer ({('BLACK' if self.current_player == Player.BLACK else 'WHITE')}) failed to produce a legal move. Skipping turn."
                    )
                    time.sleep(1)

                self.current_player = (
                    Player.WHITE
                    if self.current_player == Player.BLACK
                    else Player.BLACK
                )
            else:
                if self.handle_player_input():
                    self.current_player = (
                        Player.WHITE
                        if self.current_player == Player.BLACK
                        else Player.BLACK
                    )

        self.show_game_over()

    def handle_player_input(self):
        while True:
            try:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)

                try:
                    tty.setraw(fd)
                    key = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                if key == "\x1b":  # ESC sequence
                    key = sys.stdin.read(2)
                    if key == "[A":  # Up arrow
                        self.cursor_y = max(0, self.cursor_y - 1)
                        self.display_board()
                        continue
                    elif key == "[B":  # Down arrow
                        self.cursor_y = min(7, self.cursor_y + 1)
                        self.display_board()
                        continue
                    elif key == "[D":  # Left arrow
                        self.cursor_x = max(0, self.cursor_x - 1)
                        self.display_board()
                        continue
                    elif key == "[C":  # Right arrow
                        self.cursor_x = min(7, self.cursor_x + 1)
                        self.display_board()
                        continue

                # Handle WASD and other keys
                if key.lower() == "w":
                    self.cursor_y = max(0, self.cursor_y - 1)
                    self.display_board()
                elif key.lower() == "s":
                    self.cursor_y = min(7, self.cursor_y + 1)
                    self.display_board()
                elif key.lower() == "a":
                    self.cursor_x = max(0, self.cursor_x - 1)
                    self.display_board()
                elif key.lower() == "d":
                    self.cursor_x = min(7, self.cursor_x + 1)
                    self.display_board()
                elif key == "\r" or key == "\n":
                    if self.is_valid_move(
                        self.cursor_y, self.cursor_x, self.current_player
                    ):
                        self.make_move(
                            self.cursor_y, self.cursor_x, self.current_player
                        )
                        return True
                    else:
                        print("\nInvalid move! Press any key to continue...")

                        fd = sys.stdin.fileno()
                        old_settings = termios.tcgetattr(fd)
                        try:
                            tty.setraw(fd)
                            sys.stdin.read(1)
                        finally:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        self.display_board()

                elif key.lower() == "q":
                    self.confirm_quit()
            except KeyboardInterrupt:
                sys.exit(0)

    def confirm_quit(self):
        print("\nAre you sure you want to quit? [y/n]: ", end="", flush=True)

        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)

            try:
                tty.setraw(fd)
                key = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            print(key)

            if key.lower() == "y":
                print("\nThanks for playing!")
                sys.exit(0)
            else:
                self.display_board()

        except KeyboardInterrupt:
            sys.exit(0)

    def show_game_over(self):
        self.display_board()
        black_count, white_count = self.count_pieces()
        winner = self.get_winner()

        print(f"\nGame Over!")
        print(f"Black: {black_count} pieces")
        print(f"White: {white_count} pieces")

        if winner is None:
            print("It's a tie!")
        else:
            print(f"Winner: {'BLACK' if winner == Player.BLACK else 'WHITE'}!")

    def start_new_game(self):
        # Reset board/state for a fresh game
        self.board = [[Player.EMPTY for _ in range(8)] for _ in range(8)]
        self.initialize_board()
        self.current_player = Player.BLACK
        self.cursor_x = 3
        self.cursor_y = 2
        self.game_over = False

        self.clear_screen()

        while True:
            print("\nSelect game mode:\n")
            print("1. Human against computer (play as black)")
            print("2. Human against computer (play as white)")
            print("3. Computer versus computer")
            print("q. Quit")

            choice = input("\nEnter your choice: ").lower()

            if choice == "1":
                self.game_mode = PlayerCount.ONE
                self.human_player = Player.BLACK
                break
            elif choice == "2":
                self.game_mode = PlayerCount.ONE
                self.human_player = Player.WHITE
                break
            elif choice == "3":
                self.game_mode = PlayerCount.ZERO
                break
            elif choice == "q":
                sys.exit(0)
            else:
                print("Invalid choice. Please try again.")

        self.board = [[Player.EMPTY for _ in range(8)] for _ in range(8)]
        self.current_player = Player.BLACK
        self.cursor_x = 3
        self.cursor_y = 2
        self.game_over = False
        self.initialize_board()
        self.play_game()

    def simulate_game(self):
        self.board = [[Player.EMPTY for _ in range(8)] for _ in range(8)]
        self.current_player = Player.BLACK
        self.game_over = False
        self.initialize_board()

        while not self.game_over:
            valid_moves = self.get_valid_moves(self.current_player)

            if not valid_moves:
                opponent = (
                    Player.WHITE
                    if self.current_player == Player.BLACK
                    else Player.BLACK
                )
                opponent_moves = self.get_valid_moves(opponent)

                if not opponent_moves:
                    self.game_over = True
                    break
                else:
                    self.current_player = opponent
                    continue

            move = self.ai.get_move(self.board, self.current_player)
            proposed = move
            row_col = None
            if move and self.is_valid_move(move[0], move[1], self.current_player):
                row_col = move
            elif valid_moves:
                row_col = valid_moves[0]

            if self.ai_debug:
                print(
                    f"[AI] {('BLACK' if self.current_player == Player.BLACK else 'WHITE')} proposed {proposed}; using {row_col}"
                )

            if row_col:
                row, col = row_col
                self.make_move(row, col, self.current_player)

            self.current_player = (
                Player.WHITE
                if self.current_player == Player.BLACK
                else Player.BLACK
            )

        return self.get_winner()

    def run_simulations(self, num_games):
        black_wins = 0
        white_wins = 0
        ties = 0

        print(f"Running {num_games} simulations...")

        for i in range(num_games):
            if (i + 1) % max(1, num_games // 10) == 0:
                print(f"Completed {i + 1}/{num_games} games...")

            winner = self.simulate_game()

            if winner == Player.BLACK:
                black_wins += 1
            elif winner == Player.WHITE:
                white_wins += 1
            else:
                ties += 1

        print(f"\nSimulation Results ({num_games} games):")
        print(f"Black wins: {black_wins} ({black_wins/num_games*100:.1f}%)")
        print(f"White wins: {white_wins} ({white_wins/num_games*100:.1f}%)")
        print(f"Ties: {ties} ({ties/num_games*100:.1f}%)")

        return {
            "black_wins": black_wins,
            "white_wins": white_wins,
            "ties": ties,
            "total_games": num_games,
        }


def load_model(path):
    m = ReversiNet()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


def main():
    parser = argparse.ArgumentParser(description="Reversi - Terminal Edition")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode (computer vs computer, no UI)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of games to simulate (default: 1000, only used with --simulate)",
    )
    parser.add_argument(
        "--model", type=str, help="Path to trained model checkpoint (optional)"
    )
    parser.add_argument(
        "--sb3-model",
        type=str,
        help="Path to MaskablePPO .zip checkpoint (uses new RL agent)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to use (auto=auto-detect, mps=Apple Silicon, cuda=NVIDIA GPU, cpu=CPU)",
    )
    parser.add_argument(
        "--device-info",
        action="store_true",
        help="Print detailed device information and exit",
    )
    parser.add_argument(
        "--debug-ai",
        action="store_true",
        help="Print the AI's proposed move and the move actually played (helps confirm inference is used)",
    )

    args = parser.parse_args()

    if args.device_info:
        print_device_info()
        return

    device_str = args.device if args.device and args.device != "auto" else None
    device = get_device(preferred_device=device_str, verbose=True)

    if args.model and args.sb3_model:
        print("Specify only one of --model or --sb3-model")
        sys.exit(1)

    if args.sb3_model:
        ai = SB3MaskableAi(args.sb3_model, device=str(device), debug=args.debug_ai)
    elif args.model:
        model = load_model(args.model)
        ai = ModelAi(model, device=str(device))
    else:
        ai = Ai()

    game = ReversiGame(ai, ai_debug=args.debug_ai)
    ai.game_ref = game

    if hasattr(ai, "game_ref"):
        ai.game_ref = game

    if args.simulate:
        if args.games <= 0:
            print("Number of games must be positive")
            sys.exit(1)

        game.run_simulations(args.games)
        return

    while True:
        game.start_new_game()

        while True:
            print("\nPlay again? (y/n): ", end="", flush=True)

            try:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    key = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                print(key)

                if key.lower() == "y":
                    break
                elif key.lower() == "n":
                    print("\nThanks for playing!")
                    sys.exit(0)
                else:
                    print("Please enter 'y' for yes or 'n' for no.")

            except KeyboardInterrupt:
                print("\nThanks for playing!")
                sys.exit(0)


if __name__ == "__main__":
    main()
