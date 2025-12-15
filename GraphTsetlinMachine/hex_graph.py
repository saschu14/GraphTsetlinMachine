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

# Occupancy-pair edge types
P0_CONN = "P0_CONN"
P1_CONN = "P1_CONN"
OPP_CONN = "OPP_CONN"
EMPTY_CONN = "EMPTY_CONN"


def boards_to_graphs(
    boards: np.ndarray,
    board_dim: int,
    base_graphs: Graphs | None = None,
    hypervector_size: int = 1024,
    hypervector_bits: int = 8,
) -> Graphs:
    """
    Hex -> Graphs encoding with:
      - Cells as nodes "0".."N-1"
      - Goal nodes: P0_TOP, P0_BOTTOM, P1_LEFT, P1_RIGHT
      - Directional neighbor edges D0..D5 for all adjacent cells (always present)
      - Additional occupancy-pair edges between neighbors:
          * P0_CONN, P1_CONN, OPP_CONN, EMPTY_CONN
      - Conditional goal edges (only correct player's stones connect to their goals)
      - Node properties: occupancy + Row/Col
    """

    boards = np.asarray(boards)
    n_graphs = boards.shape[0]

    # ---- Symbols (node properties vocabulary) ----
    if base_graphs is None:
        symbols = ["Empty", "Player0", "Player1"]
        symbols += [f"Row{r}" for r in range(board_dim)]
        symbols += [f"Col{c}" for c in range(board_dim)]
        # These are node NAMES, not properties, but keeping them in symbols is harmless
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

    def idx(r: int, c: int) -> int:
        return r * board_dim + c

    # ---- Precompute directional neighbor lists for each cell ----
    # neighbors[u] = list of (v_index, edge_type_name)
    neighbors: list[list[tuple[int, str]]] = [[] for _ in range(num_board_nodes)]
    for r in range(board_dim):
        for c in range(board_dim):
            u = idx(r, c)
            for d, (dr, dc) in enumerate(_HEX_DIRECTIONS):
                rr, cc = r + dr, c + dc
                if 0 <= rr < board_dim and 0 <= cc < board_dim:
                    v = idx(rr, cc)
                    neighbors[u].append((v, _DIR_NAMES[d]))

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
                u_val = int(board[r, c])

                # Base: one directional edge per neighbor
                deg = len(neighbors[u])

                # Extra: one occupancy-pair edge per neighbor IF it matches one of the 4 types
                # (can be up to len(neighbors[u]) more)
                for v, _dir_type in neighbors[u]:
                    vr, vc = divmod(v, board_dim)
                    v_val = int(board[vr, vc])

                    if u_val == 0 and v_val == 0:
                        deg += 1
                    elif u_val == 1 and v_val == 1:
                        deg += 1
                    elif u_val == 2 and v_val == 2:
                        deg += 1
                    elif (u_val in (1, 2)) and (v_val in (1, 2)) and (u_val != v_val):
                        deg += 1
                    # else: no additional pair edge

                # Conditional goal edges (outgoing from cell -> goal), max 2
                if u_val == 1:  # Player0 (top-bottom)
                    if r == 0:
                        deg += 1
                    if r == board_dim - 1:
                        deg += 1
                elif u_val == 2:  # Player1 (left-right)
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

        # Add goal nodes (NO outgoing edges from goals in this encoding)
        for goal in goal_nodes:
            graphs.add_graph_node(
                graph_id=g,
                node_name=goal,
                number_of_graph_node_edges=0,
                node_type_name="Goal",
            )

    graphs.prepare_edge_configuration()

    # ---- Add edges + properties (must match declared degrees exactly) ----
    for g in range(n_graphs):
        board = boards[g]

        for r in range(board_dim):
            for c in range(board_dim):
                u = idx(r, c)
                u_name = str(u)
                u_val = int(board[r, c])

                # 1) Directional neighbor edges (always)
                for v, dir_type in neighbors[u]:
                    v_name = str(v)
                    graphs.add_graph_node_edge(g, u_name, v_name, dir_type)

                # 2) Occupancy-pair edges (conditional per neighbor)
                for v, _dir_type in neighbors[u]:
                    vr, vc = divmod(v, board_dim)
                    v_val = int(board[vr, vc])
                    v_name = str(v)

                    if u_val == 0 and v_val == 0:
                        graphs.add_graph_node_edge(g, u_name, v_name, EMPTY_CONN)
                    elif u_val == 1 and v_val == 1:
                        graphs.add_graph_node_edge(g, u_name, v_name, P0_CONN)
                    elif u_val == 2 and v_val == 2:
                        graphs.add_graph_node_edge(g, u_name, v_name, P1_CONN)
                    elif (u_val in (1, 2)) and (v_val in (1, 2)) and (u_val != v_val):
                        graphs.add_graph_node_edge(g, u_name, v_name, OPP_CONN)
                    # else: no extra edge

                # 3) Node properties
                if u_val == 0:
                    occ = "Empty"
                elif u_val == 1:
                    occ = "Player0"
                elif u_val == 2:
                    occ = "Player1"
                else:
                    raise ValueError(f"Invalid board value {u_val} at {(r, c)}")

                graphs.add_graph_node_property(g, u_name, occ)
                graphs.add_graph_node_property(g, u_name, f"Row{r}")
                graphs.add_graph_node_property(g, u_name, f"Col{c}")

                # 4) Conditional goal edges (only correct player's stones)
                if u_val == 1:
                    if r == 0:
                        graphs.add_graph_node_edge(g, u_name, "P0_TOP", "P0_GOAL")
                    if r == board_dim - 1:
                        graphs.add_graph_node_edge(g, u_name, "P0_BOTTOM", "P0_GOAL")
                elif u_val == 2:
                    if c == 0:
                        graphs.add_graph_node_edge(g, u_name, "P1_LEFT", "P1_GOAL")
                    if c == board_dim - 1:
                        graphs.add_graph_node_edge(g, u_name, "P1_RIGHT", "P1_GOAL")

    graphs.encode()
    return graphs
