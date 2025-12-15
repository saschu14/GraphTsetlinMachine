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

# Goal edge types
P0_GOAL_EDGE = "P0_GOAL"
P1_GOAL_EDGE = "P1_GOAL"


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
      - Directional neighbor edges D0..D5
      - Occupancy-pair edges: P0_CONN, P1_CONN, OPP_CONN, EMPTY_CONN
      - Bidirectional, conditional goal edges
      - Node properties:
          * occupancy (Empty/Player0/Player1)
          * Row/Col
          * local neighbor statistics:
              - N0_k: k player0 neighbors (k=0..6)
              - N1_k: k player1 neighbors (k=0..6)
              - NE_k: k empty neighbors (k=0..6)
      - Node types depend on occupancy: EmptyCell / P0Cell / P1Cell
      - Exact per-node edge counts
    """

    boards = np.asarray(boards)
    n_graphs = boards.shape[0]

    # ---- Symbols (node property vocabulary) ----
    if base_graphs is None:
        symbols = ["Empty", "Player0", "Player1"]
        symbols += [f"Row{r}" for r in range(board_dim)]
        symbols += [f"Col{c}" for c in range(board_dim)]

        # Neighbor-stat symbols
        symbols += [f"N0_{k}" for k in range(7)]
        symbols += [f"N1_{k}" for k in range(7)]
        symbols += [f"NE_{k}" for k in range(7)]

        # Goal node names (harmless to include)
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
    num_nodes = num_board_nodes + len(goal_nodes)

    def idx(r: int, c: int) -> int:
        return r * board_dim + c

    # ---- Precompute directional neighbor lists for each cell (static) ----
    # neighbors[u] = list of (v_index, direction_edge_type)
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

        # Goal out-degrees for bidirectional goal edges (outgoing from goals)
        p0_top_deg = int(np.sum(board[0, :] == 1))
        p0_bottom_deg = int(np.sum(board[board_dim - 1, :] == 1))
        p1_left_deg = int(np.sum(board[:, 0] == 2))
        p1_right_deg = int(np.sum(board[:, board_dim - 1] == 2))

        # Cell nodes
        for r in range(board_dim):
            for c in range(board_dim):
                u = idx(r, c)
                u_name = str(u)
                u_val = int(board[r, c])

                if u_val == 0:
                    node_type = "EmptyCell"
                elif u_val == 1:
                    node_type = "P0Cell"
                elif u_val == 2:
                    node_type = "P1Cell"
                else:
                    raise ValueError(f"Invalid board value {u_val} at {(r, c)}")

                # Base: directional neighbor edges
                deg = len(neighbors[u])

                # Extra: occupancy-pair edges
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

                # Conditional cell->goal edges (only correct player's stones)
                if u_val == 1:
                    if r == 0:
                        deg += 1
                    if r == board_dim - 1:
                        deg += 1
                elif u_val == 2:
                    if c == 0:
                        deg += 1
                    if c == board_dim - 1:
                        deg += 1

                graphs.add_graph_node(
                    graph_id=g,
                    node_name=u_name,
                    number_of_graph_node_edges=deg,
                    node_type_name=node_type,
                )

        # Goal nodes
        graphs.add_graph_node(g, "P0_TOP", p0_top_deg, "P0Goal")
        graphs.add_graph_node(g, "P0_BOTTOM", p0_bottom_deg, "P0Goal")
        graphs.add_graph_node(g, "P1_LEFT", p1_left_deg, "P1Goal")
        graphs.add_graph_node(g, "P1_RIGHT", p1_right_deg, "P1Goal")

    graphs.prepare_edge_configuration()

    # ---- Add edges + properties ----
    for g in range(n_graphs):
        board = boards[g]

        # Goal -> cell edges (bidirectional goal links: outgoing from goals)
        for c in range(board_dim):
            if int(board[0, c]) == 1:
                graphs.add_graph_node_edge(g, "P0_TOP", str(idx(0, c)), P0_GOAL_EDGE)
            if int(board[board_dim - 1, c]) == 1:
                graphs.add_graph_node_edge(g, "P0_BOTTOM", str(idx(board_dim - 1, c)), P0_GOAL_EDGE)

        for r in range(board_dim):
            if int(board[r, 0]) == 2:
                graphs.add_graph_node_edge(g, "P1_LEFT", str(idx(r, 0)), P1_GOAL_EDGE)
            if int(board[r, board_dim - 1]) == 2:
                graphs.add_graph_node_edge(g, "P1_RIGHT", str(idx(r, board_dim - 1)), P1_GOAL_EDGE)

        # Cells: edges + properties
        for r in range(board_dim):
            for c in range(board_dim):
                u = idx(r, c)
                u_name = str(u)
                u_val = int(board[r, c])

                # 1) Directional neighbor edges
                for v, dir_type in neighbors[u]:
                    graphs.add_graph_node_edge(g, u_name, str(v), dir_type)

                # 2) Occupancy-pair edges + count neighbors by occupancy
                n0 = n1 = ne = 0
                for v, _dir_type in neighbors[u]:
                    vr, vc = divmod(v, board_dim)
                    v_val = int(board[vr, vc])
                    v_name = str(v)

                    if v_val == 0:
                        ne += 1
                    elif v_val == 1:
                        n0 += 1
                    elif v_val == 2:
                        n1 += 1

                    if u_val == 0 and v_val == 0:
                        graphs.add_graph_node_edge(g, u_name, v_name, EMPTY_CONN)
                    elif u_val == 1 and v_val == 1:
                        graphs.add_graph_node_edge(g, u_name, v_name, P0_CONN)
                    elif u_val == 2 and v_val == 2:
                        graphs.add_graph_node_edge(g, u_name, v_name, P1_CONN)
                    elif (u_val in (1, 2)) and (v_val in (1, 2)) and (u_val != v_val):
                        graphs.add_graph_node_edge(g, u_name, v_name, OPP_CONN)

                # 3) Node properties: occupancy + position
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

                # 4) Node properties: local neighbor statistics
                graphs.add_graph_node_property(g, u_name, f"N0_{n0}")
                graphs.add_graph_node_property(g, u_name, f"N1_{n1}")
                graphs.add_graph_node_property(g, u_name, f"NE_{ne}")

                # 5) Cell -> goal edges (bidirectional goal links: incoming to goals)
                if u_val == 1:
                    if r == 0:
                        graphs.add_graph_node_edge(g, u_name, "P0_TOP", P0_GOAL_EDGE)
                    if r == board_dim - 1:
                        graphs.add_graph_node_edge(g, u_name, "P0_BOTTOM", P0_GOAL_EDGE)
                elif u_val == 2:
                    if c == 0:
                        graphs.add_graph_node_edge(g, u_name, "P1_LEFT", P1_GOAL_EDGE)
                    if c == board_dim - 1:
                        graphs.add_graph_node_edge(g, u_name, "P1_RIGHT", P1_GOAL_EDGE)

    # Optional debug: check a few valid node ids near the end of the board
    debug_nodes = [str(num_board_nodes - k) for k in (1, 2, 3, 4) if num_board_nodes - k >= 0]
    for g in [0, n_graphs - 1]:
        for name in debug_nodes:
            nid = graphs.graph_node_id[g][name]
            abs_id = graphs.node_index[g] + nid
            print(g, name,
                "expected", graphs.number_of_graph_node_edges[abs_id],
                "added", graphs.graph_node_edge_counter[abs_id])

    graphs.encode()
    return graphs
