import numpy as np
from GraphTsetlinMachine.graphs import Graphs

# 6 Hex directions (outgoing)
_HEX_DIRECTIONS = [
    (-1, 0),   # Up
    (-1, 1),   # Up-right
    (0, -1),   # Left
    (0, 1),    # Right
    (1, -1),   # Down-left
    (1, 0),    # Down
]
_DIR_NAMES = ["D0", "D1", "D2", "D3", "D4", "D5"]


def boards_to_graphs(
    boards: np.ndarray,
    board_dim: int,
    base_graphs: Graphs | None = None,
    hypervector_size: int = 1024,
    hypervector_bits: int = 8,
) -> Graphs:
    boards = np.asarray(boards)
    n_graphs = boards.shape[0]

    # ---- Symbols (vocabulary) ----
    if base_graphs is None:
        symbols = ["Empty", "Player0", "Player1"]
        symbols += [f"Row{r}" for r in range(board_dim)]
        symbols += [f"Col{c}" for c in range(board_dim)]
        symbols += ["P0_TOP", "P0_BOTTOM", "P1_LEFT", "P1_RIGHT"]

        graphs = Graphs(
            number_of_graphs=n_graphs,
            symbols=symbols,
            hypervector_size=hypervector_size,
            hypervector_bits=hypervector_bits,
        )
    else:
        graphs = Graphs(number_of_graphs=n_graphs, init_with=base_graphs)

    # ---- Node naming ----
    num_board_nodes = board_dim * board_dim
    goal_nodes = ["P0_TOP", "P0_BOTTOM", "P1_LEFT", "P1_RIGHT"]
    all_nodes = [str(i) for i in range(num_board_nodes)] + goal_nodes
    num_nodes = len(all_nodes)

    # ---- Precompute outgoing edges for THIS board_dim (static across graphs) ----
    # out_edges[src] = list of (dst, edge_type)
    out_edges: dict[str, list[tuple[str, str]]] = {name: [] for name in all_nodes}

    def idx(r, c):
        return r * board_dim + c

    # Cell -> neighbors (directional)
    for r in range(board_dim):
        for c in range(board_dim):
            u = str(idx(r, c))

            for d, (dr, dc) in enumerate(_HEX_DIRECTIONS):
                rr, cc = r + dr, c + dc
                if 0 <= rr < board_dim and 0 <= cc < board_dim:
                    v = str(idx(rr, cc))
                    out_edges[u].append((v, _DIR_NAMES[d]))

            # Cell <-> goal nodes (bidirectional edges so goal nodes also have outgoing edges)
            if r == 0:
                out_edges[u].append(("P0_TOP", "P0_GOAL"))
                out_edges["P0_TOP"].append((u, "P0_GOAL"))
            if r == board_dim - 1:
                out_edges[u].append(("P0_BOTTOM", "P0_GOAL"))
                out_edges["P0_BOTTOM"].append((u, "P0_GOAL"))
            if c == 0:
                out_edges[u].append(("P1_LEFT", "P1_GOAL"))
                out_edges["P1_LEFT"].append((u, "P1_GOAL"))
            if c == board_dim - 1:
                out_edges[u].append(("P1_RIGHT", "P1_GOAL"))
                out_edges["P1_RIGHT"].append((u, "P1_GOAL"))

    # ---- Tell Graphs how many nodes per graph ----
    for g in range(n_graphs):
        graphs.set_number_of_graph_nodes(g, num_nodes)
    graphs.prepare_node_configuration()

    # ---- Add nodes with the CORRECT edge counts ----
    for g in range(n_graphs):
        for name in all_nodes:
            graphs.add_graph_node(
                graph_id=g,
                node_name=name,
                number_of_graph_node_edges=len(out_edges[name]),
                node_type_name="Cell" if name.isdigit() else "Goal",
            )

    # ---- Add edges (must match the counts above exactly) ----
    graphs.prepare_edge_configuration()
    for g in range(n_graphs):
        for src, edges in out_edges.items():
            for dst, etype in edges:
                graphs.add_graph_node_edge(
                    graph_id=g,
                    source_node_name=src,
                    destination_node_name=dst,
                    edge_type_name=etype,
                )

    # ---- Node properties (only for board cells) ----
    for g in range(n_graphs):
        board = boards[g]
        if board.shape != (board_dim, board_dim):
            raise ValueError(f"Board shape {board.shape} != {(board_dim, board_dim)}")

        for r in range(board_dim):
            for c in range(board_dim):
                node = str(idx(r, c))
                v = int(board[r, c])

                if v == 0:
                    occ = "Empty"
                elif v == 1:
                    occ = "Player0"
                elif v == 2:
                    occ = "Player1"
                else:
                    raise ValueError(f"Invalid board value {v} at {(r, c)}")

                graphs.add_graph_node_property(g, node, occ)
                graphs.add_graph_node_property(g, node, f"Row{r}")
                graphs.add_graph_node_property(g, node, f"Col{c}")

    graphs.encode()
    return graphs
