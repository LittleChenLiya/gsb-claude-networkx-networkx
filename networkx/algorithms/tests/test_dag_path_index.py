import random
from itertools import pairwise

import pytest

import networkx as nx


def layered_graph_with_four_paths():
    """Layered DAG with exactly four paths from node 0 to node 5.

    Neighbors keep the node insertion order (1 before 2, 3 before 4).
    """
    return nx.DiGraph(
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 5),
            (4, 5),
        ]
    )


def test_four_path_layered_graph_ranking():
    G = layered_graph_with_four_paths()
    index = nx.DAGPathIndex(G, 5)

    assert index.number_of_paths(0) == 4
    expected = list(nx.all_simple_paths(G, 0, 5))
    assert expected == [
        [0, 1, 3, 5],
        [0, 1, 4, 5],
        [0, 2, 3, 5],
        [0, 2, 4, 5],
    ]
    assert [index.path_at(0, i) for i in range(4)] == expected


def test_four_path_layered_graph_roundtrip():
    G = layered_graph_with_four_paths()
    index = nx.DAGPathIndex(G, 5)
    for i in range(index.number_of_paths(0)):
        path = index.path_at(0, i)
        assert index.index_of_path(path) == i


def test_path_at_returns_a_copy():
    index = nx.DAGPathIndex(layered_graph_with_four_paths(), 5)
    path = index.path_at(0, 0)
    path.append(99)
    assert index.path_at(0, 0) == [0, 1, 3, 5]


def test_empty_source_target_path():
    G = layered_graph_with_four_paths()
    index = nx.DAGPathIndex(G, 5)
    assert index.number_of_paths(5) == 1
    assert index.path_at(5, 0) == [5]
    assert index.index_of_path([5]) == 0


def test_counts_for_inner_nodes():
    index = nx.DAGPathIndex(layered_graph_with_four_paths(), 5)
    assert index.number_of_paths(1) == 2
    assert index.number_of_paths(2) == 2
    assert index.number_of_paths(3) == 1
    assert index.path_at(1, 1) == [1, 4, 5]
    assert index.index_of_path([2, 3, 5]) == 0


def test_unreachable_node_has_zero_paths():
    G = layered_graph_with_four_paths()
    G.add_node(6)
    index = nx.DAGPathIndex(G, 5)
    assert index.number_of_paths(6) == 0
    with pytest.raises(IndexError):
        index.path_at(6, 0)
    with pytest.raises(nx.NetworkXError, match="cannot reach"):
        index.index_of_path([6, 5])


def test_path_at_index_errors():
    index = nx.DAGPathIndex(layered_graph_with_four_paths(), 5)
    with pytest.raises(IndexError):
        index.path_at(0, 4)
    with pytest.raises(IndexError):
        index.path_at(0, -1)
    with pytest.raises(TypeError):
        index.path_at(0, 1.0)
    with pytest.raises(TypeError):
        index.path_at(0, True)
    with pytest.raises(nx.NodeNotFound):
        index.path_at(99, 0)


def test_number_of_paths_unknown_node():
    index = nx.DAGPathIndex(layered_graph_with_four_paths(), 5)
    with pytest.raises(nx.NodeNotFound):
        index.number_of_paths(99)


@pytest.mark.parametrize("bad_path", [[], [0, 1, 3, 4], [0, 1, 1, 3, 5]])
def test_index_of_path_invalid_paths(bad_path):
    index = nx.DAGPathIndex(layered_graph_with_four_paths(), 5)
    with pytest.raises(nx.NetworkXError):
        index.index_of_path(bad_path)


def test_index_of_path_wrong_target():
    index = nx.DAGPathIndex(layered_graph_with_four_paths(), 5)
    with pytest.raises(nx.NetworkXError, match="end at the target"):
        index.index_of_path([0, 1, 3])


def test_index_of_path_missing_edge():
    index = nx.DAGPathIndex(layered_graph_with_four_paths(), 5)
    with pytest.raises(nx.NetworkXError, match="no edge"):
        index.index_of_path([0, 3, 5])


def test_index_of_path_unknown_node():
    index = nx.DAGPathIndex(layered_graph_with_four_paths(), 5)
    with pytest.raises(nx.NodeNotFound):
        index.index_of_path([0, 99, 5])


def test_neighbor_order_uses_node_insertion_order():
    G = nx.DiGraph()
    G.add_edge(0, "b")
    G.add_edge(0, "a")
    G.add_edge("b", 2)
    G.add_edge("a", 2)
    index = nx.DAGPathIndex(G, 2)
    assert index.path_at(0, 0) == [0, "b", 2]
    assert index.path_at(0, 1) == [0, "a", 2]
    assert index.index_of_path([0, "a", 2]) == 1


def test_cyclic_graph_is_rejected():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3)])
    with pytest.raises(nx.HasACycle):
        nx.DAGPathIndex(G, 3)


def test_self_loop_is_rejected():
    G = nx.DiGraph([(0, 0), (0, 1)])
    with pytest.raises(nx.HasACycle):
        nx.DAGPathIndex(G, 1)


def test_undirected_graph_is_rejected():
    with pytest.raises(nx.NetworkXError, match="directed"):
        nx.DAGPathIndex(nx.Graph([(0, 1)]), 1)


def test_multidigraph_is_rejected():
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.DAGPathIndex(nx.MultiDiGraph([(0, 1)]), 1)


def test_target_not_in_graph():
    with pytest.raises(nx.NodeNotFound):
        nx.DAGPathIndex(nx.DiGraph([(0, 1)]), 2)


def test_empty_graph():
    with pytest.raises(nx.NodeNotFound):
        nx.DAGPathIndex(nx.DiGraph(), 0)

    G = nx.empty_graph(1, create_using=nx.DiGraph)
    index = nx.DAGPathIndex(G, 0)
    assert index.number_of_paths(0) == 1
    assert index.path_at(0, 0) == [0]
    assert index.index_of_path([0]) == 0


def test_index_is_a_snapshot():
    G = layered_graph_with_four_paths()
    index = nx.DAGPathIndex(G, 5)

    # Removing edges and nodes after construction does not change the index.
    G.remove_edge(0, 2)
    G.remove_node(1)
    assert index.number_of_paths(0) == 4
    assert index.path_at(0, 3) == [0, 2, 4, 5]
    assert index.index_of_path([0, 2, 4, 5]) == 3

    # Adding nodes and edges does not extend the index either.
    G.add_edge(0, 5)
    assert index.number_of_paths(0) == 4
    with pytest.raises(nx.NodeNotFound):
        index.number_of_paths(7)


def test_more_than_2_to_the_64_paths():
    # A chain of diamond layers has 2 ** n_layers paths.
    n_layers = 70
    G = nx.DiGraph()
    layers = [[0]] + [[2 * layer + 1, 2 * layer + 2] for layer in range(n_layers)]
    target = 2 * n_layers + 1
    layers.append([target])
    for left_layer, right_layer in pairwise(layers):
        for u in left_layer:
            G.add_edges_from((u, v) for v in right_layer)
    index = nx.DAGPathIndex(G, target)

    total = index.number_of_paths(0)
    assert total == 2**n_layers
    assert total > 2**64

    # Only the first and last paths are requested; no enumeration.
    first = index.path_at(0, 0)
    last = index.path_at(0, total - 1)
    assert first[0] == 0
    assert first[-1] == target
    assert len(first) == n_layers + 2
    assert last[0] == 0
    assert last[-1] == target
    assert len(last) == n_layers + 2
    assert first != last
    assert index.index_of_path(first) == 0
    assert index.index_of_path(last) == total - 1
    assert all(n % 2 == 1 for n in first[1:])
    assert all(n % 2 == 0 for n in last[1:-1])


def test_roundtrip_against_all_simple_paths_random_dags():
    random.seed(42)
    for _ in range(20):
        raw = nx.gnp_random_graph(8, 0.3, seed=random, directed=True)
        # Insert nodes first and then edges in node order so insertion
        # order matches the DFS ordering used by all_simple_paths.
        G = nx.DiGraph()
        G.add_nodes_from(range(8))
        G.add_edges_from(
            (u, v) for u in range(8) for v in range(u + 1, 8) if raw.has_edge(u, v)
        )
        if 0 not in G or nx.has_path(G, 0, 7) is False:
            G.add_edge(0, 7)
        index = nx.DAGPathIndex(G, 7)
        expected = list(nx.all_simple_paths(G, 0, 7))
        assert index.number_of_paths(0) == len(expected)
        for i, path in enumerate(expected):
            assert index.path_at(0, i) == path
            assert index.index_of_path(path) == i
