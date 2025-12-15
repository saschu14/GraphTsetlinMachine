import numpy as np
from GraphTsetlinMachine.graphs import Graphs

# 6 Hex directions
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
    """
    Convert Hex boards to GraphTsetlinMachine Graphs with:
      - 6 directional edge types
      - 4 goal nodes (P0_TOP, P0_BOTTOM, P1_LEFT, P1_RIGHT)
    """

    boards = np.asarray(boards)
    n_graphs = boards.shape[0]

    # ----- Symbol vocabulary -----
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

    # ----- Node counts -----
    num_board_nodes = board_dim * board_dim
    goal_nodes = ["P0_TOP", "P0_BOTTOM", "P1_LEFT", "P1_RIGHT"]
    num_nodes = num_board_nodes + len(goal_nodes)

    for g in range(n_graphs):
        graphs.set_number_of_graph_nodes(g, num_nodes)

    graphs.prepare_node_configuration()

    # ----- Add nodes -----
    for g in range(n_graphs):
        # Board cells
        for node in range(num_board_nodes):
            graphs.add_graph_node(
                graph_id=g,
                node_name=str(node),
                number_of_graph_node_edges=6,
                node_type_name="Cell",
            )

        # Goal nodes
        for goal in goal_nodes:
            graphs.add_graph_node(
                graph_id=g,
                node_name=goal,
                number_of_graph_node_edges=board_dim,
                node_type_name="Goal",
            )

    # ----- Add edges -----
    graphs.prepare_edge_configuration()

    def idx(r, c):
        return r * board_dim + c

    for g in range(n_graphs):
        for r in range(board_dim):
            for c in range(board_dim):
                u = idx(r, c)

                # Neighbor edges (directional)
                for d, (dr, dc) in enumerate(_HEX_DIRECTIONS):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < board_dim and 0 <= cc < board_dim:
                        v = idx(rr, cc)
                        graphs.add_graph_node_edge(
                            graph_id=g,
                            source_node_name=str(u),
                            destination_node_name=str(v),
                            edge_type_name=_DIR_NAMES[d],
                        )

                # Goal connections
                if r == 0:
                    graphs.add_graph_node_edge(g, str(u), "P0_TOP", "P0_GOAL")
                if r == board_dim - 1:
                    graphs.add_graph_node_edge(g, str(u), "P0_BOTTOM", "P0_GOAL")
                if c == 0:
                    graphs.add_graph_node_edge(g, str(u), "P1_LEFT", "P1_GOAL")
                if c == board_dim - 1:
                    graphs.add_graph_node_edge(g, str(u), "P1_RIGHT", "P1_GOAL")

    # ----- Node properties -----
    for g in range(n_graphs):
        board = boards[g]
        for r in range(board_dim):
            for c in range(board_dim):
                node = str(idx(r, c))
                v = board[r, c]

                if v == 0:
                    occ = "Empty"
                elif v == 1:
                    occ = "Player0"
                elif v == 2:
                    occ = "Player1"
                else:
                    raise ValueError("Invalid board value")

                graphs.add_graph_node_property(g, node, occ)
                graphs.add_graph_node_property(g, node, f"Row{r}")
                graphs.add_graph_node_property(g, node, f"Col{c}")

    graphs.encode()
    return graphs
