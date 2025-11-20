import pytest

from reversi import ReversiGame, Player


def test_initial_board_setup():
    game = ReversiGame(ai=None)
    assert game.board[3][3] == Player.WHITE
    assert game.board[3][4] == Player.BLACK
    assert game.board[4][3] == Player.BLACK
    assert game.board[4][4] == Player.WHITE
    assert game.is_game_over() is False


def test_initial_legal_moves_black_and_white():
    game = ReversiGame(ai=None)

    black_moves = sorted(game.get_valid_moves(Player.BLACK))
    white_moves = sorted(game.get_valid_moves(Player.WHITE))

    assert black_moves == [(2, 3), (3, 2), (4, 5), (5, 4)]
    assert white_moves == [(2, 4), (3, 5), (4, 2), (5, 3)]


def test_game_over_when_no_moves_for_either_side():
    game = ReversiGame(ai=None)
    game.board = [[Player.BLACK for _ in range(8)] for _ in range(8)]
    game.current_player = Player.BLACK

    assert game.get_valid_moves(Player.BLACK) == []
    assert game.get_valid_moves(Player.WHITE) == []
    assert game.is_game_over() is True
    assert game.get_winner() == Player.BLACK
