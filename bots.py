import random
from typing import Optional, Tuple

from reversi import Player, ReversiGame

Move = Optional[Tuple[int, int]]


def _deepcopy_board(board):
    return [[cell for cell in row] for row in board]


class RandomBot:
    def select_move(self, game: ReversiGame, player: Player) -> Move:
        moves = game.get_valid_moves(player)
        return random.choice(moves) if moves else None


class CornerAwareMobilityBot:
    """
    Heuristic bot that mirrors expert intuition:
    - Prioritize corners, avoid X/C-squares when corner is open.
    - Favor maintaining mobility and keeping piece count low early;
      transition to higher flips later.
    """

    def __init__(self, noise: float = 0.0):
        self.noise = noise
        self.corners = {(0, 0), (0, 7), (7, 0), (7, 7)}
        self.x_squares = {(1, 1), (1, 6), (6, 1), (6, 6)}
        self.c_squares = {(0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (6, 7), (7, 6)}

    def select_move(self, game: ReversiGame, player: Player) -> Move:
        moves = game.get_valid_moves(player)
        if not moves:
            return None

        opponent = Player.WHITE if player == Player.BLACK else Player.BLACK
        best_score = float("-inf")
        best_moves = []

        for r, c in moves:
            score = 0.0

            if (r, c) in self.corners:
                score += 100.0
            else:
                # Penalize risky squares if the adjacent corner is open.
                if (r, c) in self.x_squares:
                    corner = (0, 0) if r < 4 and c < 4 else (0, 7) if r < 4 else (7, 0) if c < 4 else (7, 7)
                    if game.board[corner[0]][corner[1]] == Player.EMPTY:
                        score -= 60.0
                if (r, c) in self.c_squares:
                    corner = (0, 0) if r <= 1 and c <= 1 else (0, 7) if r <= 1 else (7, 0) if c <= 1 else (7, 7)
                    if game.board[corner[0]][corner[1]] == Player.EMPTY:
                        score -= 25.0

            sim = ReversiGame(ai=None)
            sim.board = _deepcopy_board(game.board)
            sim.current_player = player
            sim.make_move(r, c, player)

            my_moves_next = len(sim.get_valid_moves(player))
            opp_moves_next = len(sim.get_valid_moves(opponent))
            score += (my_moves_next - 2 * opp_moves_next)

            black_count, white_count = sim.count_pieces()
            my_count = black_count if player == Player.BLACK else white_count
            opp_count = white_count if player == Player.BLACK else black_count
            total = black_count + white_count
            piece_delta = my_count - opp_count

            # Early game: keep piece count modest; late game: flips matter more.
            if total < 40:
                score -= 0.5 * piece_delta
            else:
                score += 0.5 * piece_delta

            if self.noise > 0:
                score += random.uniform(-self.noise, self.noise)

            if score > best_score:
                best_score = score
                best_moves = [(r, c)]
            elif score == best_score:
                best_moves.append((r, c))

        return random.choice(best_moves)


def play_game(bot_black, bot_white, verbose: bool = False) -> Player:
    game = ReversiGame(ai=None)
    current = Player.BLACK

    while True:
        moves = game.get_valid_moves(current)
        if moves:
            move = bot_black.select_move(game, current) if current == Player.BLACK else bot_white.select_move(game, current)
            if move is not None:
                game.make_move(move[0], move[1], current)
        else:
            other = Player.WHITE if current == Player.BLACK else Player.BLACK
            if not game.get_valid_moves(other):
                break
        current = Player.WHITE if current == Player.BLACK else Player.BLACK

    winner = game.get_winner()
    if verbose:
        black_count, white_count = game.count_pieces()
        print(f"Final: B {black_count} vs W {white_count} -> {winner}")
    return winner


def play_matches(bot_a, bot_b, games: int = 20) -> dict:
    results = {"bot_a_as_black": 0, "bot_a_as_white": 0, "bot_b_as_black": 0, "bot_b_as_white": 0, "ties": 0}
    for i in range(games):
        if i % 2 == 0:
            w = play_game(bot_a, bot_b)
            if w == Player.BLACK:
                results["bot_a_as_black"] += 1
            elif w == Player.WHITE:
                results["bot_b_as_white"] += 1
            else:
                results["ties"] += 1
        else:
            w = play_game(bot_b, bot_a)
            if w == Player.BLACK:
                results["bot_b_as_black"] += 1
            elif w == Player.WHITE:
                results["bot_a_as_white"] += 1
            else:
                results["ties"] += 1
    return results
