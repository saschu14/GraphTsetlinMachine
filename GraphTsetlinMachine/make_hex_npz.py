import argparse
import numpy as np

def load_csv(path: str, board_dim: int):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    # columns: winner, moves_left, c0..c(board_dim*board_dim-1)
    winners = data[:, 0].astype(np.int32)
    moves_left = data[:, 1].astype(np.int32)
    cells = data[:, 2:].astype(np.int8)

    expected = board_dim * board_dim
    if cells.shape[1] != expected:
        raise ValueError(f"Expected {expected} cell columns, got {cells.shape[1]}")

    boards = cells.reshape(-1, board_dim, board_dim)
    return boards, winners, moves_left

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV file from hex_dataset")
    p.add_argument("--output", required=True, help="Output .npz filename")
    p.add_argument("--board-dim", type=int, required=True, help="Board size, e.g. 7 or 11")
    args = p.parse_args()

    boards, winners, moves_left = load_csv(args.input, args.board_dim)

    print("Loaded CSV:")
    print("  boards:", boards.shape)
    print("  winners:", winners.shape)
    print("  moves_left:", moves_left.shape)

    np.savez(args.output, boards=boards, labels=winners, moves_left=moves_left)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
