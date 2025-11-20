import argparse

from bots import CornerAwareMobilityBot, RandomBot, play_matches


def main():
    p = argparse.ArgumentParser(description="Evaluate heuristic vs random bots on Reversi")
    p.add_argument("--games", type=int, default=20, help="Number of games to play (will alternate colors)")
    p.add_argument("--noise", type=float, default=0.0, help="Heuristic bot tie-break noise")
    args = p.parse_args()

    heuristic = CornerAwareMobilityBot(noise=args.noise)
    random_bot = RandomBot()

    results = play_matches(heuristic, random_bot, games=args.games)

    total = args.games
    h_wins = results["bot_a_as_black"] + results["bot_a_as_white"]
    r_wins = results["bot_b_as_black"] + results["bot_b_as_white"]
    ties = results["ties"]

    print(f"Games: {total}")
    print(f"Heuristic wins: {h_wins}")
    print(f"Random wins: {r_wins}")
    print(f"Ties: {ties}")
    print("Breakdown:", results)


if __name__ == "__main__":
    main()
