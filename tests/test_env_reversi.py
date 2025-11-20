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
    assert mask[-1] == 0.0  # pass should not be legal at the starting position


def test_action_mask_matches_valid_moves():
    from reversi import Player

    env = ReversiEnv()
    env.reset()
    mask = env._legal_mask()
    moves_from_game = set(env.game.get_valid_moves(env.current_player))
    encoded_moves = {
        divmod(idx, env.board_size)
        for idx, allowed in enumerate(mask[:-1])
        if allowed == 1.0
    }
    assert moves_from_game == encoded_moves
    assert mask[-1] == 0.0


def test_step_pass_action_when_no_moves():
    env = ReversiEnv()
    env.reset()
    # Force state to no legal moves by filling board with a single color (current player)
    from reversi import Player

    env.game.board = [[Player.BLACK for _ in range(8)] for _ in range(8)]
    env.current_player = Player.BLACK

    obs, reward, terminated, truncated, info = env.step(env.action_space.n - 1)
    assert terminated is False
    assert truncated is False
    assert reward == 0.0
    assert info["action_mask"][-1] == 1.0
    assert info["action_mask"].sum() == 1.0


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


def test_margin_reward_mode():
    from reversi import Player

    env = ReversiEnv(reward_mode="margin", margin_scale=0.5)
    env.reset()
    # Fill board with BLACK wins by a margin
    env.game.board = [[Player.BLACK for _ in range(8)] for _ in range(8)]
    env.current_player = Player.WHITE

    env.step(env.action_space.n - 1)  # first pass
    obs, reward, terminated, truncated, info = env.step(env.action_space.n - 1)
    assert terminated is True
    assert truncated is False
    assert reward > 1.0


def test_consecutive_passes_end_game():
    from reversi import Player

    env = ReversiEnv()
    env.reset()
    env.game.board = [[Player.BLACK for _ in range(8)] for _ in range(8)]
    env.current_player = Player.BLACK

    # First forced pass should keep episode alive but expose pass mask
    obs, reward, terminated, truncated, info = env.step(env.action_space.n - 1)
    assert terminated is False
    assert truncated is False
    assert reward == 0.0
    assert info["action_mask"][-1] == 1.0
    assert info["action_mask"].sum() == 1.0

    # Second consecutive pass ends the game per Reversi rules
    obs, reward, terminated, truncated, info = env.step(env.action_space.n - 1)
    assert terminated is True
    assert truncated is False
    assert reward == -1.0
    assert obs.shape == (5, 8, 8)


def test_pass_mask_only_when_no_moves():
    from reversi import Player

    env = ReversiEnv()
    env.reset()
    start_mask = env._legal_mask()
    assert start_mask[-1] == 0.0

    env.game.board = [[Player.BLACK for _ in range(8)] for _ in range(8)]
    env.current_player = Player.BLACK
    no_move_mask = env._legal_mask()
    assert no_move_mask[-1] == 1.0
    assert no_move_mask.sum() == 1.0


def test_corner_aware_bot_respects_legality():
    from bots import CornerAwareMobilityBot
    from reversi import Player

    env = ReversiEnv()
    game = env.game
    bot = CornerAwareMobilityBot()
    move = bot.select_move(game, Player.BLACK)
    if move is not None:
        assert move in game.get_valid_moves(Player.BLACK)
