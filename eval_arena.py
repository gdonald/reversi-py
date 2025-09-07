# eval_arena.py
from selfplay import play_game


def head_to_head(model_new, model_best, games=200):
    # alternate colors by swapping perspectives via env reset order
    import itertools

    wins = 0
    draws = 0
    for i in range(games):
        # play two games by swapping who is queried first if you implement dual-agent loop
        data = play_game(model_new)  # quick proxy: compare values is not enough
        # For a proper arena: implement a match loop where each model drives MCTS for its own turns.
        # Omitted here for brevity.
        # As a placeholder, track loss rate trend via training loss and simple baseline matches.
    return wins, draws
