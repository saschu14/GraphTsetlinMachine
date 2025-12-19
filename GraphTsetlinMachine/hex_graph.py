import numpy as np
from collections import deque
from GraphTsetlinMachine.graphs import Graphs

# 6 Hex directions (r, c) in axial-style layout on a rhombus grid
_HEX_DIRS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
_DIR_NAMES = ["D0", "D1", "D2", "D3", "D4", "D5"]


def _idx(r: int, c: int, n: int) -> int:
    return r * n + c


def _bfs_reach(board: np.ndarray, player_val: int, starts: list[tuple[int, int]]) -> np.ndarray:
    """BFS restricted to stones == player_val. Returns reach mask."""
    n = board.shape[0]
    reach = np.zeros((n, n), dtype=bool)
    q = deque()

    for r, c in starts:
        if board[r, c] == player_val and not reach[r, c]:
            reach[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in _HEX_DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and (not reach[rr, cc]) and board[rr, cc] == player_val:
                reach[rr, cc] = True
                q.append((rr, cc))

    return reach

class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return ra


def _winning_moves_one(board: np.ndarray, player_val: int) -> np.ndarray:
    """
    Returns winmask[r,c]=True for EMPTY cells that are immediate winning moves for player_val.
    board values: 0 empty, 1 P0, 2 P1
    P0 aims top-bottom, P1 aims left-right.
    """
    n = board.shape[0]
    N = n * n

    dsu = _DSU(N)

    # Union adjacent stones of this player
    for r in range(n):
        for c in range(n):
            if int(board[r, c]) != player_val:
                continue
            u = r * n + c
            for dr, dc in _HEX_DIRS:  # uses your existing hex dirs
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and int(board[rr, cc]) == player_val:
                    v = rr * n + cc
                    dsu.union(u, v)

    # Track border-touch per component root
    touch_a = np.zeros(N, dtype=bool)  # top (P0) or left (P1)
    touch_b = np.zeros(N, dtype=bool)  # bottom (P0) or right (P1)

    for r in range(n):
        for c in range(n):
            if int(board[r, c]) != player_val:
                continue
            root = dsu.find(r * n + c)

            if player_val == 1:
                if r == 0:
                    touch_a[root] = True
                if r == n - 1:
                    touch_b[root] = True
            else:  # player_val == 2
                if c == 0:
                    touch_a[root] = True
                if c == n - 1:
                    touch_b[root] = True

    # Evaluate each empty cell: if placing here connects border A and B, it's a WIN1 move
    win = np.zeros((n, n), dtype=bool)

    for r in range(n):
        for c in range(n):
            if int(board[r, c]) != 0:
                continue

            neigh_roots = set()
            for dr, dc in _HEX_DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and int(board[rr, cc]) == player_val:
                    neigh_roots.add(dsu.find(rr * n + cc))

            # Borders touched if we place at (r,c):
            a = False
            b = False

            # The placed stone itself may touch a border
            if player_val == 1:
                if r == 0:
                    a = True
                if r == n - 1:
                    b = True
            else:
                if c == 0:
                    a = True
                if c == n - 1:
                    b = True

            # Neighbor components may touch borders
            for root in neigh_roots:
                a = a or touch_a[root]
                b = b or touch_b[root]
                if a and b:
                    win[r, c] = True
                    break

    return win


def boards_to_graphs(
    boards: np.ndarray,
    board_dim: int,
    feature_mode: str = "domain",  # "baseline", "domain", "domain_turn"
    base_graphs: Graphs | None = None,
    hypervector_size: int = 1024,
    hypervector_bits: int = 8,
) -> Graphs:
    """
    One graph per board.

    Nodes:
      - Cell nodes: "0".."N-1"
      - Goal nodes: P0_TOP, P0_BOTTOM, P1_LEFT, P1_RIGHT

    Edges:
      - Static neighbor edges D0..D5 (only where neighbor exists)
      - Conditional goal edges (only from stones on relevant borders)

    Node properties:
      - Occupancy: Empty / Player0 / Player1
      - Row/Col: Row0.., Col0..
      - Reachability props (computed deterministically from the finished board):
          P0_RT, P0_RB, P0_BOTH, P1_RL, P1_RR, P1_BOTH
    """
    boards = np.asarray(boards)
    n_graphs = boards.shape[0]

    num_board_nodes = board_dim * board_dim
    goal_nodes = ["P0_TOP", "P0_BOTTOM", "P1_LEFT", "P1_RIGHT"]
    all_nodes = [str(i) for i in range(num_board_nodes)] + goal_nodes

    # ---- Build static neighbor edge lists per cell (same for all graphs) ----
    neighbor_out = {str(i): [] for i in range(num_board_nodes)}
    for r in range(board_dim):
        for c in range(board_dim):
            u = str(_idx(r, c, board_dim))
            for d, (dr, dc) in enumerate(_HEX_DIRS):
                rr, cc = r + dr, c + dc
                if 0 <= rr < board_dim and 0 <= cc < board_dim:
                    v = str(_idx(rr, cc, board_dim))
                    neighbor_out[u].append((v, _DIR_NAMES[d]))

    # ---- Symbols (vocabulary) ----
    if base_graphs is None:
        symbols = ["Empty", "Player0", "Player1"]

        if feature_mode != "baseline":
            symbols += [f"Row{r}" for r in range(board_dim)]
            symbols += [f"Col{c}" for c in range(board_dim)]
            symbols += goal_nodes

            symbols += [
                "P0_RT", "P0_RB", "P0_BOTH",
                "P1_RL", "P1_RR", "P1_BOTH",
            ]

            symbols += [
                "P0_WIN1", "P1_WIN1",
                "P0_BLOCK1", "P1_BLOCK1",
            ]

            if feature_mode == "domain_turn":
                symbols += ["P0_TO_MOVE", "P1_TO_MOVE"]

        graphs = Graphs(
            number_of_graphs=n_graphs,
            symbols=symbols,
            hypervector_size=hypervector_size,
            hypervector_bits=hypervector_bits,
        )
    else:
        graphs = Graphs(number_of_graphs=n_graphs, init_with=base_graphs)

    # ---- Configure number of nodes per graph ----
    for g in range(n_graphs):
        graphs.set_number_of_graph_nodes(g, len(all_nodes))
    graphs.prepare_node_configuration()

    # ---- Add nodes with EXACT outgoing edge counts per graph ----
    # For each board/cell: expected edges = number of neighbors + number of goal edges (0..2) depending on stone + border
    for g in range(n_graphs):
        board = boards[g]
        if board.shape != (board_dim, board_dim):
            raise ValueError(f"Board shape {board.shape} != {(board_dim, board_dim)}")

        for node_name in all_nodes:
            if node_name in goal_nodes:
                graphs.add_graph_node(
                    graph_id=g,
                    node_name=node_name,
                    number_of_graph_node_edges=0,
                    node_type_name="Goal",
                )
                continue

            # cell node
            u = int(node_name)
            r, c = divmod(u, board_dim)
            v = int(board[r, c])  # 0 empty, 1 P0, 2 P1

            goal_edges = 0
            if v == 1:
                if r == 0:
                    goal_edges += 1
                if r == board_dim - 1:
                    goal_edges += 1
            elif v == 2:
                if c == 0:
                    goal_edges += 1
                if c == board_dim - 1:
                    goal_edges += 1

            expected = len(neighbor_out[node_name]) + goal_edges

            graphs.add_graph_node(
                graph_id=g,
                node_name=node_name,
                number_of_graph_node_edges=expected,
                node_type_name="Cell",
            )

    graphs.prepare_edge_configuration()

    # ---- Add neighbor edges (static) + goal edges (conditional) ----
    for g in range(n_graphs):
        board = boards[g]

        # Neighbor edges
        for src, edges in neighbor_out.items():
            for dst, etype in edges:
                graphs.add_graph_node_edge(
                    graph_id=g,
                    source_node_name=src,
                    destination_node_name=dst,
                    edge_type_name=etype,
                )
        
        if feature_mode == "domain_turn":
            num_p0 = np.sum(board == 1)
            num_p1 = np.sum(board == 2)

            if num_p0 == num_p1:
                turn = "P0_TO_MOVE"
            else:
                turn = "P1_TO_MOVE"

            # Goal edges (only for stones on relevant borders)
            for u in range(num_board_nodes):
                # Add to-move property
                graphs.add_graph_node_property(g, str(u), turn)
                
                # Cell position
                r, c = divmod(u, board_dim)
                v = int(board[r, c])
                name = str(u)

                if v == 1:
                    if r == 0:
                        graphs.add_graph_node_edge(g, name, "P0_TOP", "P0_GOAL")
                    if r == board_dim - 1:
                        graphs.add_graph_node_edge(g, name, "P0_BOTTOM", "P0_GOAL")
                elif v == 2:
                    if c == 0:
                        graphs.add_graph_node_edge(g, name, "P1_LEFT", "P1_GOAL")
                    if c == board_dim - 1:
                        graphs.add_graph_node_edge(g, name, "P1_RIGHT", "P1_GOAL")

    # ---- Add properties (incl. reachability) ----
    for g in range(n_graphs):
        board = boards[g]

        # Reachability masks (deterministic)
        p0_rt = _bfs_reach(board, 1, [(0, c) for c in range(board_dim)])
        p0_rb = _bfs_reach(board, 1, [(board_dim - 1, c) for c in range(board_dim)])
        p0_both = p0_rt & p0_rb

        p1_rl = _bfs_reach(board, 2, [(r, 0) for r in range(board_dim)])
        p1_rr = _bfs_reach(board, 2, [(r, board_dim - 1) for r in range(board_dim)])
        p1_both = p1_rl & p1_rr

        p0_win1 = _winning_moves_one(board, 1)
        p1_win1 = _winning_moves_one(board, 2)

        for u in range(num_board_nodes):
            r, c = divmod(u, board_dim)
            v = int(board[r, c])
            node = str(u)

            if v == 0:
                graphs.add_graph_node_property(g, node, "Empty")
                if v == 0:
                    # If P0 can win in one move by placing here
                    if p0_win1[r, c]:
                        graphs.add_graph_node_property(g, node, "P0_WIN1")

                    # If P1 can win in one move by placing here
                    if p1_win1[r, c]:
                        graphs.add_graph_node_property(g, node, "P1_WIN1")
                    
                    # blocking moves (block opponent's immediate win)
                    if p1_win1[r, c]:
                        graphs.add_graph_node_property(g, node, "P0_BLOCK1")

                    if p0_win1[r, c]:
                        graphs.add_graph_node_property(g, node, "P1_BLOCK1")


            elif v == 1:
                graphs.add_graph_node_property(g, node, "Player0")
            elif v == 2:
                graphs.add_graph_node_property(g, node, "Player1")
            else:
                raise ValueError(f"Invalid cell value {v} at {(r, c)}")

            if feature_mode != "baseline":
                graphs.add_graph_node_property(g, node, f"Row{r}")
                graphs.add_graph_node_property(g, node, f"Col{c}")

            if v == 1:
                if p0_rt[r, c]:
                    graphs.add_graph_node_property(g, node, "P0_RT")
                if p0_rb[r, c]:
                    graphs.add_graph_node_property(g, node, "P0_RB")
                if p0_both[r, c]:
                    graphs.add_graph_node_property(g, node, "P0_BOTH")

            elif v == 2:
                if p1_rl[r, c]:
                    graphs.add_graph_node_property(g, node, "P1_RL")
                if p1_rr[r, c]:
                    graphs.add_graph_node_property(g, node, "P1_RR")
                if p1_both[r, c]:
                    graphs.add_graph_node_property(g, node, "P1_BOTH")

    graphs.encode()
    return graphs
