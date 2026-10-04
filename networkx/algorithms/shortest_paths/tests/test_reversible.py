import pytest

import networkx as nx


def _walk_path(G, source, target, path, reversible):
    """Check that `path` is a valid walk from source to target and return
    the number of reversed edges it uses."""
    assert path[0][0] == source
    assert path[-1][1] == target
    reversals = 0
    for u, v, key, rev in path:
        if rev:
            # Traversed from u to v against the edge direction, so the
            # underlying edge runs from v to u.
            assert G.has_edge(v, u)
            if G.is_multigraph():
                assert key in G[v][u]
                assert G[v][u][key].get(reversible, False)
            else:
                assert key is None
                assert G[v][u].get(reversible, False)
            reversals += 1
        else:
            assert G.has_edge(u, v)
            if G.is_multigraph():
                assert key in G[u][v]
            else:
                assert key is None
    return reversals


def test_basic_reversal():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("c", "b", reversible=True)
    n, path = nx.shortest_path_with_reversals(G, "a", "c")
    assert n == 1
    assert path == [("a", "b", None, False), ("b", "c", None, True)]
    assert _walk_path(G, "a", "c", path, "reversible") == n


def test_disabled_reversible_no_path():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("c", "b", reversible=False)
    with pytest.raises(nx.NetworkXNoPath):
        nx.shortest_path_with_reversals(G, "a", "c")
    # Same when the attribute is missing entirely (defaults to False).
    G["c"]["b"].pop("reversible")
    with pytest.raises(nx.NetworkXNoPath):
        nx.shortest_path_with_reversals(G, "a", "c")


def test_forward_preferred_over_reverse():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("c", "b", reversible=True)
    G.add_edge("a", "c")
    n, path = nx.shortest_path_with_reversals(G, "a", "c")
    assert n == 0
    assert path == [("a", "c", None, False)]
    assert _walk_path(G, "a", "c", path, "reversible") == n


def test_same_source_target():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    n, path = nx.shortest_path_with_reversals(G, "a", "a")
    assert n == 0
    assert path == []


def test_parallel_edges_key():
    G = nx.MultiDiGraph()
    G.add_edge("a", "b")  # key 0
    G.add_edge("a", "b")  # key 1
    G.add_edge("b", "c")  # key 0
    n, path = nx.shortest_path_with_reversals(G, "a", "c")
    assert n == 0
    assert path == [("a", "b", 0, False), ("b", "c", 0, False)]
    assert _walk_path(G, "a", "c", path, "reversible") == n


def test_parallel_edges_reverse_key():
    G = nx.MultiDiGraph()
    G.add_edge("a", "b")  # key 0, not reversible
    G.add_edge("a", "b", reversible=True)  # key 1
    n, path = nx.shortest_path_with_reversals(G, "b", "a")
    assert n == 1
    assert path == [("b", "a", 1, True)]
    assert _walk_path(G, "b", "a", path, "reversible") == n


def test_multigraph_reverse_picks_first_reversible_key():
    G = nx.MultiDiGraph()
    G.add_edge("b", "a", reversible=True)  # key 0
    G.add_edge("b", "a", reversible=True)  # key 1
    n, path = nx.shortest_path_with_reversals(G, "a", "b")
    assert n == 1
    assert path == [("a", "b", 0, True)]
    assert _walk_path(G, "a", "b", path, "reversible") == n


def test_cycle():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    G.add_edge("c", "a")
    n, path = nx.shortest_path_with_reversals(G, "a", "a")
    assert n == 0
    assert path == []
    n, path = nx.shortest_path_with_reversals(G, "a", "c")
    assert n == 0
    assert path == [("a", "b", None, False), ("b", "c", None, False)]
    assert _walk_path(G, "a", "c", path, "reversible") == n


def test_cost_zero_cycle_does_not_loop():
    # a<->b form a cost-0 cycle; a also points to the target t.
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "a")
    G.add_edge("a", "t")
    n, path = nx.shortest_path_with_reversals(G, "a", "t")
    assert n == 0
    assert path == [("a", "t", None, False)]
    # From b to t: b->a->t, no loop.
    n, path = nx.shortest_path_with_reversals(G, "b", "t")
    assert n == 0
    assert path == [("b", "a", None, False), ("a", "t", None, False)]


def test_self_loop_does_not_move():
    G = nx.DiGraph()
    G.add_edge("a", "a", reversible=True)
    G.add_edge("a", "b")
    n, path = nx.shortest_path_with_reversals(G, "a", "b")
    assert n == 0
    assert path == [("a", "b", None, False)]
    n, path = nx.shortest_path_with_reversals(G, "a", "a")
    assert n == 0
    assert path == []


def test_self_loop_reverse_cannot_reach_new_node():
    G = nx.DiGraph()
    G.add_edge("a", "a", reversible=True)
    G.add_node("b")
    with pytest.raises(nx.NetworkXNoPath):
        nx.shortest_path_with_reversals(G, "a", "b")


def test_multiple_reversals():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("c", "b", reversible=True)
    G.add_edge("c", "d")
    G.add_edge("e", "d", reversible=True)
    n, path = nx.shortest_path_with_reversals(G, "a", "e")
    assert n == 2
    assert path == [
        ("a", "b", None, False),
        ("b", "c", None, True),
        ("c", "d", None, False),
        ("d", "e", None, True),
    ]
    assert _walk_path(G, "a", "e", path, "reversible") == n


def test_tie_break_node_insertion_order():
    G = nx.DiGraph()
    G.add_node("x")
    G.add_node("y")
    G.add_edge("s", "x")
    G.add_edge("s", "y")
    G.add_edge("x", "t")
    G.add_edge("y", "t")
    n, path = nx.shortest_path_with_reversals(G, "s", "t")
    assert n == 0
    assert path == [("s", "x", None, False), ("x", "t", None, False)]


def test_tie_break_edge_insertion_order():
    G = nx.MultiDiGraph()
    G.add_edge("s", "t")  # key 0
    G.add_edge("s", "t")  # key 1
    n, path = nx.shortest_path_with_reversals(G, "s", "t")
    assert path == [("s", "t", 0, False)]


def test_tie_break_lexicographic_not_last_predecessor():
    # Two equal-cost paths. The lexicographically smallest path has the
    # earliest inserted *first* node (a < b), even though the other path's
    # last predecessor (y) settles before x in a plain first-settler
    # Dijkstra.
    G = nx.DiGraph()
    G.add_node("s")
    G.add_node("a")
    G.add_node("b")
    G.add_node("y")
    G.add_node("x")
    G.add_node("t")
    G.add_edge("s", "a")
    G.add_edge("a", "x")
    G.add_edge("x", "t")
    G.add_edge("s", "b")
    G.add_edge("b", "y")
    G.add_edge("y", "t")
    n, path = nx.shortest_path_with_reversals(G, "s", "t")
    assert n == 0
    assert path == [
        ("s", "a", None, False),
        ("a", "x", None, False),
        ("x", "t", None, False),
    ]


def test_graph_not_modified():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("c", "b", reversible=True)
    edges_before = [(u, v, d.copy()) for u, v, d in G.edges(data=True)]
    nodes_before = list(G.nodes)
    nx.shortest_path_with_reversals(G, "a", "c")
    assert [(u, v, d.copy()) for u, v, d in G.edges(data=True)] == edges_before
    assert list(G.nodes) == nodes_before


def test_undirected_raises():
    G = nx.Graph()
    G.add_edge("a", "b")
    with pytest.raises(nx.NetworkXError):
        nx.shortest_path_with_reversals(G, "a", "b")


def test_node_not_found():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    with pytest.raises(nx.NodeNotFound):
        nx.shortest_path_with_reversals(G, "a", "z")
    with pytest.raises(nx.NodeNotFound):
        nx.shortest_path_with_reversals(G, "z", "a")


def test_custom_reversible_attribute():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("c", "b", allow_reverse=True)
    n, path = nx.shortest_path_with_reversals(G, "a", "c", reversible="allow_reverse")
    assert n == 1
    assert path == [("a", "b", None, False), ("b", "c", None, True)]
    assert _walk_path(G, "a", "c", path, "allow_reverse") == n


def test_reversible_edge_used_in_natural_direction_is_free():
    G = nx.DiGraph()
    G.add_edge("a", "b", reversible=True)
    G.add_edge("b", "c", reversible=True)
    n, path = nx.shortest_path_with_reversals(G, "a", "c")
    assert n == 0
    assert path == [("a", "b", None, False), ("b", "c", None, False)]
