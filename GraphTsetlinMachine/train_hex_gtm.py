import argparse
import time
import numpy as np

from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine  # or GraphTsetlinMachine for strict binary
from GraphTsetlinMachine.hex_graph import boards_to_graphs

BOARD_DIM = 11


def load_hex_dataset_npz(path: str):
    """
    Expected .npz format:
        boards: (N, 11, 11) with values {0,1,2}
        labels: (N,) with values {0,1}  (winner player index)
    """
    data = np.load(path)
    boards = data["boards"]
    labels = data["labels"].astype(np.int32)
    moves_left = data["moves_left"].astype(np.int32)

    mask = (moves_left == 0)   # only completed positions
    boards = boards[mask]
    labels = labels[mask]

    if boards.ndim != 3 or boards.shape[1] != BOARD_DIM or boards.shape[2] != BOARD_DIM:
        raise ValueError(
            f"Expected 'boards' with shape (N,{BOARD_DIM},{BOARD_DIM}), got {boards.shape}"
        )

    if labels.ndim != 1:
        raise ValueError(f"Expected 'labels' with shape (N,), got {labels.shape}")

    return boards, labels


def train_and_evaluate(
    dataset_path: str,
    clauses: int = 4000,
    T: float = 20.0,
    s: float = 10.0,
    depth: int = 3,
    epochs: int = 50,
    message_size: int = 256,
    message_bits: int = 2,
    hypervector_size: int = 128,
    hypervector_bits: int = 2,
    seed: int = 1,
    log_every: int = 1,
    eval_every: int = 0,  # set to e.g. 10 to evaluate every 10 epochs (costs time)
):
    # --- Load dataset ---
    boards, labels = load_hex_dataset_npz(dataset_path)
    n_samples = boards.shape[0]

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    boards = boards[indices]
    labels = labels[indices]

    split = int(0.8 * n_samples)
    boards_train, boards_test = boards[:split], boards[split:]
    y_train, y_test = labels[:split], labels[split:]

    print(f"Loaded {n_samples} positions.")
    print(f"Train: {boards_train.shape[0]}, Test: {boards_test.shape[0]}")

    # --- Build Graphs objects ---
    print("Encoding training graphs...")
    graphs_train = boards_to_graphs(
        boards_train,
        board_dim=BOARD_DIM,
        base_graphs=None,
        hypervector_size=hypervector_size,
        hypervector_bits=hypervector_bits,
    )

    print("Encoding test graphs (reusing symbol hypervectors)...")
    graphs_test = boards_to_graphs(
        boards_test,
        board_dim=BOARD_DIM,
        base_graphs=graphs_train,
        hypervector_size=hypervector_size,
        hypervector_bits=hypervector_bits,
    )

    # --- Initialize GTM ---
    print("Initializing Graph Tsetlin Machine...")
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=clauses,
        T=T,
        s=s,
        depth=depth,
        message_size=message_size,
        message_bits=message_bits,
        double_hashing=False,
        one_hot_encoding=False,
    )

    # --- Training with timing ---
    print(
        f"Training GTM: clauses={clauses}, T={T}, s={s}, depth={depth}, "
        f"epochs={epochs}, message_size={message_size}"
    )

    start = time.time()
    epoch_times = []

    for e in range(epochs):
        t0 = time.time()
        tm.fit(graphs_train, y_train, epochs=1)
        t1 = time.time()

        dt = t1 - t0
        epoch_times.append(dt)

        # ETA based on average epoch time so far (includes first-epoch overhead)
        avg = sum(epoch_times) / len(epoch_times)
        remaining = avg * (epochs - (e + 1))

        if log_every > 0 and ((e + 1) % log_every == 0):
            msg = f"Epoch {e+1}/{epochs} | {dt:.2f}s | avg {avg:.2f}s | ETA ~ {remaining/60:.1f} min"

            if eval_every and ((e + 1) % eval_every == 0):
                # Optional: evaluation is expensive, so do it periodically
                y_pred_test = tm.predict(graphs_test)
                test_acc = (y_pred_test == y_test).mean()
                msg += f" | test_acc {test_acc*100:.2f}%"

            print(msg)

    total = time.time() - start
    print(f"Training time total: {total/60:.2f} minutes")

    # --- Final Evaluation ---
    print("Evaluating...")
    y_pred_train = tm.predict(graphs_train)
    y_pred_test = tm.predict(graphs_test)

    train_acc = (y_pred_train == y_train).mean()
    test_acc = (y_pred_test == y_test).mean()

    print(f"Train accuracy: {train_acc * 100:.2f}%")
    print(f"Test accuracy:  {test_acc * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train Graph Tsetlin Machine on Hex positions.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to .npz dataset file containing 'boards' and 'labels'.",
    )
    parser.add_argument("--clauses", type=int, default=4000)
    parser.add_argument("--T", type=float, default=20.0)
    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--message-size", type=int, default=256)
    parser.add_argument("--message-bits", type=int, default=2)
    parser.add_argument("--hypervector-size", type=int, default=128)
    parser.add_argument("--hypervector-bits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)

    # new flags
    parser.add_argument("--log-every", type=int, default=1, help="Print timing every N epochs (default: 1)")
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate test accuracy every N epochs (0=off)")

    args = parser.parse_args()

    train_and_evaluate(
        dataset_path=args.data,
        clauses=args.clauses,
        T=args.T,
        s=args.s,
        depth=args.depth,
        epochs=args.epochs,
        message_size=args.message_size,
        message_bits=args.message_bits,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
