import pandas as pd
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View lengths of snake.")
    parser.add_argument('-f',
                        '--file_lengths',
                        type=str,
                        default='lengths.csv',
                        help='File containing the lengths history of the \
                            snake over all learning episodes.')
    args = parser.parse_args()

    # Load the lengths from the CSV file
    filename_lengths = args.file_lengths
    try:
        lengths_df = pd.read_csv(filename_lengths)
        lengths = lengths_df['length'].values
    except FileNotFoundError:
        print(f"File {filename_lengths} not found.")
        exit(1)

    # Plot the lengths
    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.plot(lengths, label='Length of Snake')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.title('Length of Snake Over Episodes')
    plt.legend()
    plt.show()
