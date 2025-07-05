import os
import argparse
from snake_class import Snake
import pandas as pd

STATS_FILE_MANUAL = "stats/snake_stats_man.csv"
STATS_FILE_AUTO = "stats/snake_stats_auto.csv"
STATS_FILE_LEARNING = "stats/snake_stats_learn.csv"
STATISTICS_FIELDS = ["score", "moves", "green_apples", "red_apples"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Snake Game Options')
    parser.add_argument('-f',
                        '--file_model',
                        type=str,
                        help='File name to save the model.')
    parser.add_argument('-c',
                        '--cells',
                        type=int, help='Number of cells in the grid.')
    parser.add_argument('-t',
                        '--time_frequency',
                        type=int,
                        help='Time frequency for the game loop.')
    args = parser.parse_args()
    if not args.file_model:
        print("No model name provided.Default name will be used.")
        modelname = None
    else:
        modelname = args.file_model
    if not os.path.exists(modelname):
        print(f"Model file {modelname} does not exist.")
        print("Starting with a new model.")
        modelname = None
    if not os.path.exists(STATS_FILE_MANUAL):
        df = pd.DataFrame(columns=[STATISTICS_FIELDS])
        df.to_csv(STATS_FILE_MANUAL, index=False)
    if not os.path.exists(STATS_FILE_AUTO):
        df = pd.DataFrame(columns=[STATISTICS_FIELDS])
        df.to_csv(STATS_FILE_AUTO, index=False)
    if not os.path.exists(STATS_FILE_LEARNING):
        df = pd.DataFrame(columns=[STATISTICS_FIELDS])
        df.to_csv(STATS_FILE_LEARNING, index=False)
    if not args.cells:
        print("No number of cells provided. Defaulting to 10x10 grid.")
        args.cells = 10
    else:
        print(f"Using {args.cells}x{args.cells} grid.")
    if args.cells < 5:
        print("Number of cells is too small. Setting to 5x5 grid.")
        args.cells = 5
    if args.cells > 50:
        print("Number of cells is too large. Setting to 50x50 grid.")
        args.cells = 50
    if not args.time_frequency:
        print("No time frequency provided. Defaulting to 10 hertz.")
        frequency = 10
    else:
        print(f"Using {args.time_frequency} seconds time frequency.")
        frequency = args.time_frequency

    game = Snake(1600, 1200,
                 [args.cells, args.cells],
                 modelname=modelname,
                 time_frequency=frequency,
                 stats_man=STATS_FILE_MANUAL,
                 stats_auto=STATS_FILE_AUTO,
                 stats_learn=STATS_FILE_LEARNING)
    game.load_grass("./icons/grass.png")
    game.load_green_apple("./icons/green-apple-48.png")
    game.load_red_apple("./icons/red-apple-48.png")
    game.load_head_worn("./icons/head1.png")
    game.load_body_worn("./icons/Body.png")
    game.load_tail_worn("./icons/Tail.png")
    game.load_corner_worn("./icons/corner1.png")
    game._select_mode()
