import os
import argparse
from snake_class import Snake

MODEL_DEFAULT_NAME = "model.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN agent for the Snake game.')
    parser.add_argument('-f', '--file_model', type=str, help='File name to save the model.')
    parser.add_argument('-c', '--cells', type=int, help='Number of cells in the grid.')
    args = parser.parse_args()
    if not args.file_model:
        print("No model name provided.Default name will be used.")
        args.file_model = MODEL_DEFAULT_NAME
        modelname = None
    else:
        modelname = args.file_model
    if  not os.path.exists(modelname):
        print(f"Model file {modelname} does not exist. Starting with a new model.")
        modelname = None
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

    game = Snake(1600, 1200, [args.cells, args.cells], modelname=modelname)
    game.load_grass("./icons/grass.png")
    game.load_green_apple("./icons/green-apple-48.png")
    game.load_red_apple("./icons/red-apple-48.png")
    game.load_head_worn("./icons/head1.png")
    game.load_body_worn("./icons/Body.png")
    game.load_tail_worn("./icons/Tail.png")
    game.load_corner_worn("./icons/corner1.png")
    game._select_mode()

