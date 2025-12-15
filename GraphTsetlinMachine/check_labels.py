import numpy as np
from collections import deque

BOARD_DIM = 11

# Hex neighbor directions (same as your graph)
DIRS = [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]

def winner_from_board(board):
    """
    board values: 0 empty, 1 player0 (top-bottom), 2 player1 (left-right)
    returns 0 or 1
    """
    n = board.shape[0]

    # Player0: top -> bottom using cells == 1
    q = deque()
    seen = set()
    for c in range(n):
        if board[0, c] == 1:
            q.append((0, c))
            seen.add((0, c))
    while q:
        r, c = q.popleft()
        if r == n - 1:
            return 0
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and board[rr, cc] == 1 and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))

    # Player1: left -> right using cells == 2
    q = deque()
    seen = set()
    for r in range(n):
        if board[r, 0] == 2:
            q.append((r, 0))
            seen.add((r, 0))
    while q:
        r, c = q.popleft()
        if c == n - 1:
            return 1
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and board[rr, cc] == 2 and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))

    # If neither connects, your “completed” boards aren’t actually terminal-win boards.
    return -1

def main(path="hex_11x11.npz", max_check=2000):
    data = np.load(path)
    boards = data["boards"]
    labels = data["labels"].astype(np.int64)
    moves_left = data["moves_left"].astype(np.int64) if "moves_left" in data else None

    print("boards:", boards.shape, "labels:", labels.shape)
    print("label counts:", np.bincount(labels))
    if moves_left is not None:
        print("moves_left counts:", {k:int(v) for k,v in zip(*np.unique(moves_left, return_counts=True))})

    idx = np.arange(len(labels))
    if moves_left is not None:
        idx = idx[moves_left == 0]  # check “completed” only

    idx = idx[:max_check]
    bad = 0
    undecided = 0

    for i in idx:
        w = winner_from_board(boards[i])
        if w == -1:
            undecided += 1
        elif w != labels[i]:
            bad += 1

    print(f"Checked {len(idx)} completed boards")
    print("Undecided (no winner found):", undecided)
    print("Label mismatches:", bad, f"({bad/len(idx)*100:.2f}%)")

if __name__ == "__main__":
    main()
