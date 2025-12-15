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

    # ---- Symbols ----
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

    num_board_nodes = board_dim * board_dim
    goal_nodes = ["P0_TOP", "P0_BOTTOM", "P1_LEFT", "P1_RIGHT"]
    all_nodes = [str(i) for i in range(num_board_nodes)] + goal_nodes
    num_nodes = len(all_nodes)

    def idx(r: int, c: int) -> int:
        return r * board_dim + c

    # ---- Precompute static neighbor lists (directional) for each cell ----
    # neighbors[u] = list of (v, edge_type)
    neighbors: list[list[tuple[str, str]]] = [[] for _ in range(num_board_nodes)]
    for r in range(board_dim):
        for c in range(board_dim):
            u = idx(r, c)
            for d, (dr, dc) in enumerate(_HEX_DIRECTIONS):
                rr, cc = r + dr, c + dc
                if 0 <= rr < board_dim and 0 <= cc < board_dim:
                    v = idx(rr, cc)
                    neighbors[u].append((str(v), _DIR_NAMES[d]))

    # ---- Number of nodes per graph ----
    for g in range(n_graphs):
        graphs.set_number_of_graph_nodes(g, num_nodes)
    graphs.prepare_node_configuration()

    # ---- Add nodes with EXACT out-degree per graph ----
    for g in range(n_graphs):
        board = boards[g]
        if board.shape != (board_dim, board_dim):
            raise ValueError(f"Board shape {board.shape} != {(board_dim, board_dim)}")

        # Add all cell nodes
        for r in range(board_dim):
            for c in range(board_dim):
                u = idx(r, c)
                u_name = str(u)

                # base: neighbor edges
                deg = len(neighbors[u])

                # conditional goal edges (outgoing from cell -> goal)
                v = int(board[r, c])
                if v == 1:  # Player0
                    if r == 0:
                        deg += 1
                    if r == board_dim - 1:
                        deg += 1
                elif v == 2:  # Player1
                    if c == 0:
                        deg += 1
                    if c == board_dim - 1:
                        deg += 1

                graphs.add_graph_node(
                    graph_id=g,
                    node_name=u_name,
                    number_of_graph_node_edges=deg,
                    node_type_name="Cell",
                )

        # Add goal nodes (NO outgoing edges from goals)
        for goal in goal_nodes:
            graphs.add_graph_node(
                graph_id=g,
                node_name=goal,
                number_of_graph_node_edges=0,
                node_type_name="Goal",
            )

    graphs.prepare_edge_configuration()

    # ---- Add edges + properties ----
    for g in range(n_graphs):
        board = boards[g]

        for r in range(board_dim):
            for c in range(board_dim):
                u = idx(r, c)
                u_name = str(u)
                v = int(board[r, c])

                # neighbor edges
                for v_name, etype in neighbors[u]:
                    graphs.add_graph_node_edge(g, u_name, v_name, etype)

                # node properties
                if v == 0:
                    occ = "Empty"
                elif v == 1:
                    occ = "Player0"
                elif v == 2:
                    occ = "Player1"
                else:
                    raise ValueError(f"Invalid board value {v} at {(r, c)}")

                graphs.add_graph_node_property(g, u_name, occ)
                graphs.add_graph_node_property(g, u_name, f"Row{r}")
                graphs.add_graph_node_property(g, u_name, f"Col{c}")

                # conditional goal edges (only correct player's stones)
                if v == 1:
                    if r == 0:
                        graphs.add_graph_node_edge(g, u_name, "P0_TOP", "P0_GOAL")
                    if r == board_dim - 1:
                        graphs.add_graph_node_edge(g, u_name, "P0_BOTTOM", "P0_GOAL")
                elif v == 2:
                    if c == 0:
                        graphs.add_graph_node_edge(g, u_name, "P1_LEFT", "P1_GOAL")
                    if c == board_dim - 1:
                        graphs.add_graph_node_edge(g, u_name, "P1_RIGHT", "P1_GOAL")

    graphs.encode()
    return graphs
