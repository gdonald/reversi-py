import argparse
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO

from bots import RandomBot, CornerAwareMobilityBot
from env_reversi import ReversiEnv
from reversi import Player, ReversiGame


def make_bot(name: str):
    if name == "random":
        return RandomBot()
    if name == "heuristic":
        return CornerAwareMobilityBot()
    raise ValueError(f"Unknown bot '{name}'")


def agent_action(model: MaskablePPO, env: ReversiEnv) -> Optional[tuple]:
    obs = env._encode_board()
    mask = env._legal_mask()
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    action = int(action)
    if action == env.board_size * env.board_size:
        return None
    r, c = divmod(action, env.board_size)
    return r, c


def play_game(agent: MaskablePPO, bot, agent_as: Player) -> Player:
    game = ReversiGame(ai=None)
    env = ReversiEnv()
    env.game = game

    current = Player.BLACK

    while True:
        env.current_player = current
        moves = game.get_valid_moves(current)

        if moves:
            if current == agent_as:
                move = agent_action(agent, env)
            else:
                move = bot.select_move(game, current)
            if move is not None:
                game.make_move(move[0], move[1], current)
        else:
            other = Player.WHITE if current == Player.BLACK else Player.BLACK
            if not game.get_valid_moves(other):
                break

        current = Player.WHITE if current == Player.BLACK else Player.BLACK

    return game.get_winner()


def main():
    p = argparse.ArgumentParser(description="Evaluate a MaskablePPO Reversi agent vs a bot")
    p.add_argument("--model", required=True, help="Path to MaskablePPO .zip checkpoint")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--opponent", type=str, choices=["random", "heuristic"], default="heuristic")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    model = MaskablePPO.load(args.model, device=args.device)
    bot = make_bot(args.opponent)

    results = {"agent_black": 0, "agent_white": 0, "ties": 0}
    for i in range(args.games):
        agent_as = Player.BLACK if i % 2 == 0 else Player.WHITE
        winner = play_game(model, bot, agent_as)
        if winner == agent_as:
            if agent_as == Player.BLACK:
                results["agent_black"] += 1
            else:
                results["agent_white"] += 1
        elif winner is None:
            results["ties"] += 1

    total = args.games
    wins = results["agent_black"] + results["agent_white"]
    win_rate = wins / total if total else 0.0
    print(f"Games: {total} vs {args.opponent}")
    print(f"Agent wins: {wins} (black {results['agent_black']}, white {results['agent_white']})")
    print(f"Ties: {results['ties']}")
    print(f"Win rate: {win_rate:.2%}")


if __name__ == "__main__":
    main()
