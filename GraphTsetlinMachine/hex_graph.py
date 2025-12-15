import numpy as np
from collections import deque
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


def _idx(r: int, c: int, n: int) -> int:
    return r * n + c


def _bfs_reach(board: np.ndarray, player_val: int, starts: list[tuple[int, int]]) -> np.ndarray:
    """
    BFS on the hex grid restricted to cells == player_val.
    Returns boolean array reach[r,c] = True if reachable from any start cell.
    """
    n = board.shape[0]
    reach = np.zeros((n, n), dtype=bool)
    q = deque()

    for (r, c) in starts:
        if board[r, c] == player_val and not reach[r, c]:
            reach[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in _HEX_DIRECTIONS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and not reach[rr, cc] and board[rr, cc] == player_val:
                reach[rr, cc] = True
                q.append((rr, cc))

    return reach


def boards_to_graphs(
    boards: np.ndarray,
    board_dim: int,
    base_graphs: Graphs | None = None,
    hypervector_size: int = 1024,
    hypervector_bits: int = 8,
) -> Graphs:
    """
    Builds one graph per board.

    Nodes:
      - One node per cell: "0".."N-1"
      - 4 goal nodes: P0_TOP, P0_BOTTOM, P1_LEFT, P1_RIGHT

    Edges:
      - 6 directional neighbor edge types D0..D5 (static)
      - Conditional goal edges:
          * P0 stones connect to P0_TOP/P0_BOTTOM if on those borders
          * P1 stones connect to P1_LEFT/P1_RIGHT if on those borders

    Node properties:
      - occupancy: Empty / Player0 / Player1
      - row/col symbols: Row0.., Col0..
      - reachability (completed-board-friendly features):
          * P0_RT, P0_RB, P0_BOTH
          * P1_RL, P1_RR, P1_BOTH
    """
    boards = np.asarray(boards)
    n_graphs = boards.shape[0]

    # ---- Symbols (vocabulary) ----
    if base_graphs is None:
        symbols = ["Empty", "Player0", "Player1"]
        symbols += [f"Row{r}" for r in range(board_dim)]
        symbols += [f"Col{c}" for c in range(board_dim)]
        symbols += ["P0_TOP", "P0_BOTTOM", "P1_LEFT", "P1_RIGHT"]

        # Reachability properties
        symbols += ["P0_RT", "P0_RB", "P0_BOTH", "P1_RL", "P1_RR", "P1_BOTH"]

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

    # out_edges[src] = list of (dst, edge_type) for STATIC neighbor edges
    out_edges: dict[str, list[tuple[str, str]]] = {name: [] for name in all_nodes}

    for r in range(board_dim):
        for c in range(board_dim):
            u = str(_idx(r, c, board_dim))
            for d, (dr, dc) in enumerate(_HEX_DIRECTIONS):
                rr, cc = r + dr, c + dc
                if 0 <= rr < board_dim and 0 <= cc < board_dim:
                    v = str(_idx(rr, cc, board_dim))
                    out_edges[u].append((v, _DIR_NAMES[d]))

    # ---- Tell Graphs how many nodes per graph ----
    num_nodes = len(all_nodes)
    for g in range(n_graphs):
        graphs.set_number_of_graph_nodes(g, num_nodes)
    graphs.prepare_node_configuration()

    # ---- Add nodes with safe edge counts ----
    # Each cell has 6 neighbor edges + up to 2 conditional goal edges => max 8
    for g in range(n_graphs):
        for name in all_nodes:
            if name.isdigit():
                edge_slots = 8
                node_type = "Cell"
            else:
                edge_slots = 0
                node_type = "Goal"

            graphs.add_graph_node(
                graph_id=g,
                node_name=name,
                number_of_graph_node_edges=edge_slots,
                node_type_name=node_type,
            )

    graphs.prepare_edge_configuration()

    # ---- Add static neighbor edges ----
    for g in range(n_graphs):
        for src, edges in out_edges.items():
            if not src.isdigit():
                continue
            for dst, etype in edges:
                graphs.add_graph_node_edge(
                    graph_id=g,
                    source_node_name=src,
                    destination_node_name=dst,
                    edge_type_name=etype,
                )

    # ---- Add node properties + conditional goal edges + reachability props ----
    for g in range(n_graphs):
        board = boards[g]
        if board.shape != (board_dim, board_dim):
            raise ValueError(f"Board shape {board.shape} != {(board_dim, board_dim)}")

        # Reachability sets
        # P0 (value 1): top and bottom
        p0_top_starts = [(0, c) for c in range(board_dim)]
        p0_bot_starts = [(board_dim - 1, c) for c in range(board_dim)]
        p0_rt = _bfs_reach(board, 1, p0_top_starts)
        p0_rb = _bfs_reach(board, 1, p0_bot_starts)
        p0_both = p0_rt & p0_rb

        # P1 (value 2): left and right
        p1_left_starts = [(r, 0) for r in range(board_dim)]
        p1_right_starts = [(r, board_dim - 1) for r in range(board_dim)]
        p1_rl = _bfs_reach(board, 2, p1_left_starts)
        p1_rr = _bfs_reach(board, 2, p1_right_starts)
        p1_both = p1_rl & p1_rr

        for r in range(board_dim):
            for c in range(board_dim):
                node = str(_idx(r, c, board_dim))
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

                # Reachability properties (only meaningful on stones)
                if v == 1:
                    if p0_rt[r, c]:
                        graphs.add_graph_node_property(g, node, "P0_RT")
                    if p0_rb[r, c]:
                        graphs.add_graph_node_property(g, node, "P0_RB")
                    if p0_both[r, c]:
                        graphs.add_graph_node_property(g, node, "P0_BOTH")

                    # Conditional goal edges
                    if r == 0:
                        graphs.add_graph_node_edge(g, node, "P0_TOP", "P0_GOAL")
                    if r == board_dim - 1:
                        graphs.add_graph_node_edge(g, node, "P0_BOTTOM", "P0_GOAL")

                elif v == 2:
                    if p1_rl[r, c]:
                        graphs.add_graph_node_property(g, node, "P1_RL")
                    if p1_rr[r, c]:
                        graphs.add_graph_node_property(g, node, "P1_RR")
                    if p1_both[r, c]:
                        graphs.add_graph_node_property(g, node, "P1_BOTH")

                    # Conditional goal edges
                    if c == 0:
                        graphs.add_graph_node_edge(g, node, "P1_LEFT", "P1_GOAL")
                    if c == board_dim - 1:
                        graphs.add_graph_node_edge(g, node, "P1_RIGHT", "P1_GOAL")

    graphs.encode()
    return graphs
