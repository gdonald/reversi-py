import numpy as np

from env_reversi import ReversiEnv


def test_reset_provides_mask_and_obs_shape():
    env = ReversiEnv()
    obs, info = env.reset()
    assert obs.shape == (5, 8, 8)
    assert "action_mask" in info
    mask = info["action_mask"]
    assert mask.shape == (65,)
    # Initial legal moves should be 4 for the starting player
    assert np.isclose(mask.sum(), 4)


def test_step_pass_action_when_no_moves():
    env = ReversiEnv()
    env.reset()
    # Force state to no legal moves by filling board with a single color (current player)
    from reversi import Player

    env.game.board = [[Player.BLACK for _ in range(8)] for _ in range(8)]
    env.current_player = Player.BLACK

    obs, reward, terminated, truncated, info = env.step(env.action_space.n - 1)
    assert terminated is True
    assert truncated is False
    assert reward == 1.0
    assert obs.shape == (5, 8, 8)


def test_illegal_action_penalty():
    env = ReversiEnv()
    env.reset()
    # Pick an obviously illegal action (corner on full board start is legal; choose a middle empty beyond mask)
    illegal_action = 0  # likely illegal at start (0,0) is legal? ensure actual mask check)
    if env._legal_mask()[illegal_action] == 1.0:
        illegal_action = 1
    obs, reward, terminated, truncated, info = env.step(illegal_action)
    assert reward == -1.0
    assert terminated is True
    assert info.get("illegal_action") is True


def test_corner_aware_bot_respects_legality():
    from bots import CornerAwareMobilityBot
    from reversi import Player

    env = ReversiEnv()
    game = env.game
    bot = CornerAwareMobilityBot()
    move = bot.select_move(game, Player.BLACK)
    if move is not None:
        assert move in game.get_valid_moves(Player.BLACK)
