import argparse
from typing import Optional, Sequence

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


def play_game(agent: MaskablePPO, bot, agent_as: Player, capture_render: bool = False):
    game = ReversiGame(ai=None)
    env = ReversiEnv()
    env.game = game

    current = Player.BLACK
    frames = []

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
            if capture_render:
                frames.append(env.render())
        else:
            other = Player.WHITE if current == Player.BLACK else Player.BLACK
            if not game.get_valid_moves(other):
                break

        current = Player.WHITE if current == Player.BLACK else Player.BLACK

    return game.get_winner(), frames


def eval_vs_bot(model: MaskablePPO, bot, games: int) -> dict:
    results = {"agent_black": 0, "agent_white": 0, "ties": 0}
    for i in range(games):
        agent_as = Player.BLACK if i % 2 == 0 else Player.WHITE
        winner, _ = play_game(model, bot, agent_as, capture_render=False)
        if winner == agent_as:
            if agent_as == Player.BLACK:
                results["agent_black"] += 1
            else:
                results["agent_white"] += 1
        elif winner is None:
            results["ties"] += 1
    return results


def summarize(res: dict) -> tuple:
    total = sum(res.values())
    wins = res["agent_black"] + res["agent_white"]
    ties = res["ties"]
    win_rate = wins / total if total else 0.0
    return wins, ties, total, win_rate


def main():
    p = argparse.ArgumentParser(description="Evaluate a MaskablePPO Reversi agent vs baseline bots")
    p.add_argument("--model", required=True, help="Path to MaskablePPO .zip checkpoint")
    p.add_argument("--games", type=int, default=20)
    p.add_argument(
        "--opponents",
        nargs="+",
        choices=["random", "heuristic"],
        default=["heuristic", "random"],
        help="List of opponents to evaluate against",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--render-games", type=int, default=0, help="Number of games to render to stdout starting from the first matchup")
    p.add_argument("--render-opponent", type=str, default=None, choices=["random", "heuristic"], help="Render against this opponent (defaults to first in list)")
    args = p.parse_args()

    model = MaskablePPO.load(args.model, device=args.device)

    for idx, opp_name in enumerate(args.opponents):
        bot = make_bot(opp_name)
        res = eval_vs_bot(model, bot, games=args.games)
        wins, ties, total, win_rate = summarize(res)
        print(f"Opponent: {opp_name}")
        print(f"  Games: {total}  Wins: {wins} (black {res['agent_black']}, white {res['agent_white']})  Ties: {ties}  Winrate: {win_rate:.2%}")
        should_render = args.render_games and (
            (args.render_opponent is None and idx == 0)
            or (args.render_opponent == opp_name)
        )
        if should_render:
            to_render = min(args.render_games, args.games)
            print(f"\nRendering {to_render} sample game(s) vs {opp_name}")
            for i in range(to_render):
                agent_as = Player.BLACK if i % 2 == 0 else Player.WHITE
                winner, frames = play_game(
                    model,
                    bot,
                    agent_as,
                    capture_render=True,
                )
                label = "BLACK" if agent_as == Player.BLACK else "WHITE"
                print(f"\nGame {i+1} (agent as {label}): winner={winner}")
                for t, frame in enumerate(frames, 1):
                    print(f"\nTurn {t} ({'B' if t % 2 == 1 else 'W'} to move before action)")
                    print(frame)
            print()


if __name__ == "__main__":
    main()
