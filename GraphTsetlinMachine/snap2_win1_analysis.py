import numpy as np
from GraphTsetlinMachine.hex_graph import _winning_moves_one  # uses the helper we added

def main(npz_path: str, snap: int = 2):
    d = np.load(npz_path)
    boards = d["boards"]
    labels = d["labels"].astype(int)
    moves_left = d["moves_left"].astype(int)

    mask = (moves_left == snap)
    boards = boards[mask]
    labels = labels[mask]

    n = len(labels)
    print(f"snap={snap} boards: {n}")
    print("label counts:", np.bincount(labels))

    p0_has = 0
    p1_has = 0
    both_has = 0
    none_has = 0

    # conditional counts by label
    p0_has_by_label = [0, 0]
    p1_has_by_label = [0, 0]

    for b, y in zip(boards, labels):
        p0_win1 = _winning_moves_one(b, 1)
        p1_win1 = _winning_moves_one(b, 2)
        has0 = bool(p0_win1.any())
        has1 = bool(p1_win1.any())

        if has0: p0_has += 1
        if has1: p1_has += 1
        if has0 and has1: both_has += 1
        if (not has0) and (not has1): none_has += 1

        if has0: p0_has_by_label[y] += 1
        if has1: p1_has_by_label[y] += 1

    print(f"P0 has WIN1: {p0_has}/{n} = {p0_has/n:.3f}")
    print(f"P1 has WIN1: {p1_has}/{n} = {p1_has/n:.3f}")
    print(f"Both have WIN1: {both_has}/{n} = {both_has/n:.3f}")
    print(f"Neither has WIN1: {none_has}/{n} = {none_has/n:.3f}")

    for y in [0, 1]:
        count_y = np.sum(labels == y)
        print(f"\nGiven label={y} (n={count_y}):")
        print(f"  P0 has WIN1: {p0_has_by_label[y]}/{count_y} = {p0_has_by_label[y]/count_y:.3f}")
        print(f"  P1 has WIN1: {p1_has_by_label[y]}/{count_y} = {p1_has_by_label[y]/count_y:.3f}")

if __name__ == "__main__":
    # Example:
    # python3 analyze_snap2_win1.py GraphTsetlinMachine/hex_11x11.npz
    import sys
    path = sys.argv[1]
    main(path, snap=2)
