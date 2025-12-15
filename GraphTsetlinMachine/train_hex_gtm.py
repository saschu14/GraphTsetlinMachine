import argparse
import time
import numpy as np

from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from GraphTsetlinMachine.hex_graph import boards_to_graphs

BOARD_DIM = None

def load_hex_dataset_npz(path: str, only_completed: bool = True, limit: int | None = None):
    """
    Expected .npz format:
        boards: (N, 11, 11) with values {0,1,2}
        labels: (N,) with values {0,1}
        moves_left: (N,) with values like {0,2,5}
    """
    data = np.load(path)
    boards = data["boards"]
    labels = data["labels"].astype(np.uint32)
    global BOARD_DIM
    BOARD_DIM = boards.shape[1]

    if only_completed:
        if "moves_left" not in data:
            raise ValueError("Dataset missing 'moves_left' array but only_completed=True.")
        moves_left = data["moves_left"].astype(np.int32)
        mask = (moves_left == 0)
        boards = boards[mask]
        labels = labels[mask]

    if limit is not None:
        boards = boards[:limit]
        labels = labels[:limit]

    if boards.ndim != 3 or boards.shape[1] != BOARD_DIM or boards.shape[2] != BOARD_DIM:
        raise ValueError(f"Expected 'boards' with shape (N,{BOARD_DIM},{BOARD_DIM}), got {boards.shape}")

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
    eval_every: int = 0,
    balance_train: bool = True,
    only_completed: bool = True,
    limit_samples: int | None = None,
):
    # --- Load dataset ---
    boards, labels = load_hex_dataset_npz(dataset_path, only_completed=only_completed, limit=limit_samples)
    n_samples = boards.shape[0]

    counts = np.bincount(labels.astype(np.int64), minlength=2)
    print("Label counts:", counts, "majority baseline:", counts.max() / counts.sum())

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    boards = boards[indices]
    labels = labels[indices]

    split = int(0.8 * n_samples)
    boards_train, boards_test = boards[:split], boards[split:]
    y_train, y_test = labels[:split], labels[split:]

    # Optional: balance training set (helps avoid single-class collapse)
    if balance_train:
        idx0 = np.where(y_train == 0)[0]
        idx1 = np.where(y_train == 1)[0]
        m = min(len(idx0), len(idx1))
        keep = np.concatenate([
            rng.choice(idx0, m, replace=False),
            rng.choice(idx1, m, replace=False),
        ])
        rng.shuffle(keep)
        boards_train = boards_train[keep]
        y_train = y_train[keep]
        print("Balanced train counts:", np.bincount(y_train.astype(np.int64), minlength=2))

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

    print(
        f"Training GTM: clauses={clauses}, T={T}, s={s}, depth={depth}, "
        f"epochs={epochs}, message_size={message_size}"
    )

    start = time.time()

    # Train in chunks (avoids the problematic pattern fit(..., epochs=1) in a tight loop)
    chunk = eval_every if (eval_every and eval_every > 0) else log_every
    chunk = chunk if (chunk and chunk > 0) else 1

    done = 0
    epoch_times = []
    best_test = -1.0
    best_epoch = -1
    best_state = None
    patience = 6          # number of evals with no improvement
    bad = 0
    
    w_before = None

    while done < epochs:
        n = min(chunk, epochs - done)

        t0 = time.time()
        tm.fit(graphs_train, y_train, epochs=n)
        dt = time.time() - t0
        epoch_times.append(dt)

        if done == n:  # after first chunk
            w_before = tm.get_state()[1].copy()
        elif done == 2*n:  # after second chunk
            w_after = tm.get_state()[1].copy()
            print("weights changed after 2 chunks:", np.any(w_before != w_after))

        done += n

        # logging
        if log_every and log_every > 0:
            avg = sum(epoch_times) / len(epoch_times)
            remaining = avg * ((epochs - done) / max(n, 1))
            msg = f"Epoch {done}/{epochs} | {dt:.2f}s | avg_chunk {avg:.2f}s | ETA ~ {remaining/60:.1f} min"

            if eval_every and eval_every > 0:
                y_pred_train = tm.predict(graphs_train)
                y_pred_test = tm.predict(graphs_test)
                train_acc = (y_pred_train == y_train).mean()
                test_acc = (y_pred_test == y_test).mean()
                if test_acc > best_test + 1e-6:
                    best_test = test_acc
                    best_epoch = done
                    best_state = tm.get_state()
                    bad = 0
                else:
                    bad += 1

                msg += f" | best_test {best_test*100:.2f}% @ {best_epoch}"

                if bad >= patience:
                    print(f"Early stopping at epoch {done} (best @ {best_epoch})")
                    break

                msg += f" | train_acc {train_acc*100:.2f}% | test_acc {test_acc*100:.2f}%"

                # Detailed prediction counts
                pred_counts_train = np.bincount(y_pred_train.astype(np.int64), minlength=2)
                pred_counts_test  = np.bincount(y_pred_test.astype(np.int64), minlength=2)
                msg += f" | pred_train {pred_counts_train.tolist()} | pred_test {pred_counts_test.tolist()}"

            print(msg)
    if best_state is not None:
        tm.set_state(best_state)
        print(f"Restored best model @ epoch {best_epoch} with best_test {best_test*100:.2f}%")


    total = time.time() - start
    print(f"Training time total: {total/60:.2f} minutes")

    # Final evaluation
    print("Evaluating...")
    y_pred_train = tm.predict(graphs_train)
    y_pred_test = tm.predict(graphs_test)

    train_acc = (y_pred_train == y_train).mean()
    test_acc = (y_pred_test == y_test).mean()

    print(f"Train accuracy: {train_acc*100:.2f}%")
    print(f"Test accuracy:  {test_acc*100:.2f}%")
    pred_counts = np.bincount(y_pred_test.astype(np.int64), minlength=2)
    print("Pred counts (test):", pred_counts)
    print(f"Best_test {best_test*100:.2f}% @ {best_epoch}")

def main():
    parser = argparse.ArgumentParser(description="Train Graph Tsetlin Machine on Hex positions.")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz dataset file.")
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

    parser.add_argument("--log-every", type=int, default=1, help="Print timing every N chunks (default: 1)")
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluate accuracy every N epochs (0=off)")

    parser.add_argument("--no-balance", action="store_true", help="Disable balancing of training set")
    parser.add_argument("--all-snaps", action="store_true", help="Use all positions (moves_left 0/2/5), not only completed")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (debug/overfit test)")

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
        balance_train=(not args.no_balance),
        only_completed=(not args.all_snaps),
        limit_samples=args.limit,
    )


if __name__ == "__main__":
    main()
