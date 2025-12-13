import numpy as np

BOARD_DIM = 11

def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    # columns: winner, moves_left, c0..c120
    winners = data[:, 0].astype(np.int32)
    moves_left = data[:, 1].astype(np.int32)
    cells = data[:, 2:].astype(np.int8)

    # reshape cells into (N, 11, 11)
    boards = cells.reshape(-1, BOARD_DIM, BOARD_DIM)
    return boards, winners, moves_left

def main():
    boards, winners, moves_left = load_csv("hex_11x11_dataset.csv")

    print("Loaded CSV:")
    print("  boards:", boards.shape)
    print("  winners:", winners.shape)
    print("  moves_left:", moves_left.shape)

    np.savez("hex_11x11.npz", boards=boards, labels=winners, moves_left=moves_left)
    print("Saved hex_11x11.npz")

if __name__ == "__main__":
    main()
