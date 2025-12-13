import numpy as np
from graphs import Graphs

# Directions for Hex neighbors (axial-ish on 2D grid):
# Up, Up-Right, Left, Right, Down-Left, Down
_HEX_DIRECTIONS = [
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
]


def build_hex_adjacency(board_dim: int):
    """
    Build adjacency list for a board_dim x board_dim Hex board.

    Nodes are indexed linearly:
        node_id = r * board_dim + c
    """
    num_nodes = board_dim * board_dim
    adj = [[] for _ in range(num_nodes)]

    def idx(r, c):
        return r * board_dim + c

    for r in range(board_dim):
        for c in range(board_dim):
            u = idx(r, c)
            for dr, dc in _HEX_DIRECTIONS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < board_dim and 0 <= cc < board_dim:
                    v = idx(rr, cc)
                    adj[u].append(v)

    return adj


def make_symbols(board_dim: int):
    """
    Define the symbol vocabulary for node properties.

    - Occupancy: Empty / Player0 / Player1
    - Position: Row0..Row10, Col0..Col10 (for 11x11)
    """
    symbols = ["Empty", "Player0", "Player1"]
    symbols += [f"Row{r}" for r in range(board_dim)]
    symbols += [f"Col{c}" for c in range(board_dim)]
    return symbols


def boards_to_graphs(
    boards: np.ndarray,
    board_dim: int,
    base_graphs: Graphs | None = None,
    hypervector_size: int = 128,
    hypervector_bits: int = 2,
) -> Graphs:
    """
    Convert a batch of Hex boards into a Graphs object.

    Parameters
    ----------
    boards : np.ndarray
        Array of shape (num_positions, board_dim, board_dim)
        Values:
            0 = empty
            1 = player 0
            2 = player 1
    board_dim : int
        Board dimension, e.g., 11 for an 11x11 board.
    base_graphs : Graphs or None
        If None: create a new Graphs object and generate symbol hypervectors.
        If given: create a Graphs object initialized with the same symbols and
                  hypervectors as base_graphs (used for test set).
    hypervector_size : int
        Length of hypervectors used to represent symbols.
    hypervector_bits : int
        Number of active bits per symbol hypervector.

    Returns
    -------
    graphs : Graphs
        Populated Graphs instance ready for GTM training/inference.
    """
    boards = np.asarray(boards)
    assert boards.ndim == 3, "Expected boards with shape (N, board_dim, board_dim)"
    n_graphs = boards.shape[0]

    if base_graphs is None:
        symbols = make_symbols(board_dim)
        graphs = Graphs(
            number_of_graphs=n_graphs,
            symbols=symbols,
            hypervector_size=hypervector_size,
            hypervector_bits=hypervector_bits,
        )
    else:
        # Reuse symbol IDs and hypervectors from base_graphs
        graphs = Graphs(
            number_of_graphs=n_graphs,
            init_with=base_graphs,
        )

    # --- Node counts and basic structure ---
    num_nodes = board_dim * board_dim
    for g in range(n_graphs):
        graphs.set_number_of_graph_nodes(g, num_nodes)

    graphs.prepare_node_configuration()

    # Precompute adjacency (same for all graphs)
    adjacency = build_hex_adjacency(board_dim)

    # Add nodes
    for g in range(n_graphs):
        for node in range(num_nodes):
            degree = len(adjacency[node])
            graphs.add_graph_node(
                graph_id=g,
                node_name=str(node),
                number_of_graph_node_edges=degree,
                node_type_name="Cell",
            )

    # Add edges
    graphs.prepare_edge_configuration()
    for g in range(n_graphs):
        for u in range(num_nodes):
            for v in adjacency[u]:
                graphs.add_graph_node_edge(
                    graph_id=g,
                    source_node_name=str(u),
                    destination_node_name=str(v),
                    edge_type_name="Adjacent",
                )

    # --- Node properties: occupancy + (row, col) ---
    for g in range(n_graphs):
        board = boards[g]
        if board.shape != (board_dim, board_dim):
            raise ValueError(f"Board has wrong shape {board.shape}, expected {(board_dim, board_dim)}")

        for r in range(board_dim):
            for c in range(board_dim):
                node_id = r * board_dim + c
                node_name = str(node_id)

                value = int(board[r, c])

                if value == 0:
                    occ_symbol = "Empty"
                elif value == 1:
                    occ_symbol = "Player0"
                elif value == 2:
                    occ_symbol = "Player1"
                else:
                    raise ValueError(f"Unexpected board value {value} at ({r}, {c})")

                # Occupancy
                graphs.add_graph_node_property(g, node_name, occ_symbol)
                # Position
                graphs.add_graph_node_property(g, node_name, f"Row{r}")
                graphs.add_graph_node_property(g, node_name, f"Col{c}")

    # Final consistency check + signature
    graphs.encode()
    return graphs
