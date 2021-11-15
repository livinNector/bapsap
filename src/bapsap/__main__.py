from .image.data_extraction import extract_ball_color_codes
from .tubestate import TubeState
from . import adb_player
import matplotlib.pyplot as plt
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        add_help="Solves the ballsort puzzle game using various stategies using adb_player or from an image."
    )
    parser.add_argument(
        "-f",
        "--file",
        required=False,
        help="filename of the image file to be used as the level image.",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        default="dfs",
        help="The search strategy to be used for searching the solution.",
    )
    parser.add_argument(
        "-n",
        "--balls-per-color",
        required=False,
        default=4,
        help="Number of balls per color",
    )
    parser.add_argument(
        "--list-strategies",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="list all the available strategies",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to plot the extracted balls",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.list_strategies:
        for i, strategy in enumerate(TubeState().search_strategies.keys()):
            print(f"{i+1}. {strategy}")
    elif args.file:
        tube_state = extract_ball_color_codes(args.file, args.balls_per_color)
        print(tube_state)
        solution = TubeState(tube_state).find_solution(args.strategy)
        print(*solution, sep=" -> ")
        print("Length of the solution:", len(solution))
    else:
        adb_player.main()
