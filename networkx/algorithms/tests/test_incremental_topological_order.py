import heapq
import random

import pytest

import networkx as nx
from networkx.algorithms.incremental_topological_order import (
    IncrementalTopologicalOrder,
    IncrementalTopologicalOrderBatchResult,
)


def reference_topological_order(G, rank):
    """Test-only full-graph Kahn sort using explicit insertion ranks."""
    indegree = dict(G.in_degree())
    ready = [(rank[node], node) for node in G if indegree[node] == 0]
    heapq.heapify(ready)
    order = []

    while ready:
        _, node = heapq.heappop(ready)
        order.append(node)
        for child in G.successors(node):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (rank[child], child))

    if len(order) != len(G):
        raise nx.HasACycle
    return order


def assert_topological(order, G):
    position = {node: i for i, node in enumerate(order)}
    assert set(position) == set(G.nodes)
    for u, v in G.edges:
        assert position[u] < position[v]


def test_acceptance_order_and_cyclic_batch_rollback():
    original = nx.DiGraph([("A", "C"), ("B", "C")])
    index = IncrementalTopologicalOrder(original)

    assert index.order == ("A", "B", "C")
    assert index.position("A") == 0
    assert index.position("C") == 2

    result = index.apply_batch(edges=[("B", "A")])
    assert result == IncrementalTopologicalOrderBatchResult(True)
    accepted, cycle = result
    assert accepted is True
    assert cycle is None
    assert index.order == ("B", "A", "C")
    assert [index.position(node) for node in ("B", "A", "C")] == [0, 1, 2]

    result = index.apply_batch(edges=[("C", "B")])
    rejected, witness = result
    assert rejected is False
    assert witness in (
        [("B", "C"), ("C", "B")],
        [("B", "A"), ("A", "C"), ("C", "B")],
    )
    assert witness[0][0] == witness[-1][1]
    assert index.order == ("B", "A", "C")
    assert list(index.graph.edges()) == [("A", "C"), ("B", "C"), ("B", "A")]
    assert set(index.graph.nodes()) == {"A", "B", "C"}

    original.add_edge("A", "B")
    assert not index.graph.has_edge("A", "B")


def test_empty_batch_has_no_effect():
    G = nx.DiGraph([(1, 2)])
    index = IncrementalTopologicalOrder(G)

    result = index.apply_batch()

    assert result.accepted is True
    assert result.cycle is None
    assert index.order == (1, 2)
    assert list(index.graph.edges()) == [(1, 2)]


def test_mixed_node_types_are_not_compared():
    unordered = object()
    G = nx.DiGraph()
    index = IncrementalTopologicalOrder(G)

    result = index.apply_batch(nodes=[0, "x", (1, 2), unordered])

    assert result.accepted is True
    assert index.order == (0, "x", (1, 2), unordered)
    assert index.position(unordered) == 3

    result = index.apply_batch(edges=[(unordered, 0)])
    assert result.accepted is True
    assert index.order == ("x", (1, 2), unordered, 0)


def test_self_loop_is_rejected_and_new_nodes_roll_back():
    G = nx.DiGraph([("A", "B")])
    index = IncrementalTopologicalOrder(G)
    previous_order = index.order
    previous_edges = set(index.graph.edges())

    result = index.apply_batch(nodes=["C"], edges=[("B", "B")])

    assert result.accepted is False
    assert result.cycle == [("B", "B")]
    assert index.order == previous_order
    assert set(index.graph.edges()) == previous_edges
    assert not index.graph.has_node("C")


def test_node_can_be_declared_before_it_is_referenced():
    index = IncrementalTopologicalOrder(nx.DiGraph())

    result = index.apply_batch(nodes=["A", "B"], edges=[("A", "B")])

    assert result.accepted is True
    assert index.order == ("A", "B")
    assert list(index.graph.edges()) == [("A", "B")]


def test_duplicate_edges_have_no_side_effect():
    G = nx.DiGraph([(1, 2)])
    index = IncrementalTopologicalOrder(G)

    result = index.apply_batch(edges=[(1, 2), (1, 2), (2, 3)], nodes=[3, 3])

    assert result.accepted is True
    assert index.order == (1, 2, 3)
    assert list(index.graph.edges()) == [(1, 2), (2, 3)]

    result = index.apply_batch(edges=[(1, 2), (3, 1)])
    assert result.accepted is False
    assert index.order == (1, 2, 3)
    assert set(index.graph.edges()) == {(1, 2), (2, 3)}


def test_unknown_endpoint_must_be_declared_in_batch():
    G = nx.DiGraph([("A", "B")])
    index = IncrementalTopologicalOrder(G)

    with pytest.raises(nx.NetworkXError, match="must be declared"):
        index.apply_batch(edges=[("B", "C")])

    assert index.order == ("A", "B")
    assert set(index.graph.nodes()) == {"A", "B"}
    assert set(index.graph.edges()) == {("A", "B")}


def test_cycle_rolls_back_new_nodes_and_all_edges():
    G = nx.DiGraph([("A", "B")])
    index = IncrementalTopologicalOrder(G)

    result = index.apply_batch(nodes=["C"], edges=[("C", "A"), ("B", "C")])

    assert result.accepted is False
    assert result.cycle[0][0] == result.cycle[-1][1]
    assert not index.graph.has_node("C")
    assert set(index.graph.edges()) == {("A", "B")}
    assert index.order == ("A", "B")


def test_initial_graph_validation():
    with pytest.raises(nx.HasACycle):
        IncrementalTopologicalOrder(nx.DiGraph([(1, 2), (2, 1)]))

    with pytest.raises(nx.NetworkXNotImplemented):
        IncrementalTopologicalOrder(nx.Graph([(1, 2)]))

    with pytest.raises(nx.NetworkXNotImplemented):
        IncrementalTopologicalOrder(nx.MultiDiGraph([(1, 2)]))


def test_graph_view_and_position_queries_are_read_only():
    index = IncrementalTopologicalOrder(nx.DiGraph([(1, 2)]))

    assert len(index) == 2
    assert tuple(index) == (1, 2)
    with pytest.raises(nx.NodeNotFound):
        index.position(3)
    with pytest.raises(nx.NetworkXError):
        index.graph.add_edge(2, 1)


def test_api_is_exported_from_top_level_namespace():
    assert nx.IncrementalTopologicalOrder is IncrementalTopologicalOrder
    assert (
        nx.IncrementalTopologicalOrderBatchResult
        is IncrementalTopologicalOrderBatchResult
    )


def test_local_update_does_not_scan_unrelated_branch(monkeypatch):
    G = nx.DiGraph(
        [("A", "C"), ("B", "C"), ("U", "V")],
    )
    G.add_node("W")
    index = IncrementalTopologicalOrder(G)
    graph = index._graph
    visited = []
    old_predecessors = graph.predecessors

    def counting_predecessors(node):
        visited.append(node)
        return old_predecessors(node)

    monkeypatch.setattr(graph, "predecessors", counting_predecessors)

    result = index.apply_batch(edges=[("B", "A")])

    assert result.accepted is True
    assert index.order == ("B", "A", "C", "U", "V", "W")
    assert visited == ["A", "B"]
    assert "U" not in visited
    assert "V" not in visited
    assert "C" not in visited
    assert "W" not in visited


def test_random_batches_against_full_reference():
    rng = random.Random(20260930)
    index = IncrementalTopologicalOrder(nx.DiGraph())
    rank = {}
    next_rank = 0

    for _ in range(200):
        declared = []
        if len(rank) < 25 and rng.random() < 0.7:
            declared = [
                (len(rank) + offset, object())
                if (len(rank) + offset) % 2
                else str(len(rank) + offset)
                for offset in range(rng.randrange(3))
            ]

        available = list(index.graph.nodes) + declared
        edges = []
        if available:
            edges = [
                (rng.choice(available), rng.choice(available))
                for _ in range(rng.randrange(5))
            ]

        candidate = nx.DiGraph(index.graph)
        candidate.add_nodes_from(declared)
        candidate.add_edges_from(edges)
        saved_order = index.order
        saved_nodes = set(index.graph.nodes)
        saved_edges = set(index.graph.edges)

        result = index.apply_batch(declared, edges)
        try:
            expected = reference_topological_order(
                candidate,
                {**rank, **{node: next_rank + i for i, node in enumerate(declared)}},
            )
        except nx.HasACycle:
            expected = None

        assert result.accepted is (expected is not None)
        if expected is not None:
            assert_topological(index.order, index.graph)
            assert len(index.order) == len(index.graph)
            for node in declared:
                if node not in rank:
                    rank[node] = next_rank
                    next_rank += 1
        else:
            assert index.order == saved_order
            assert set(index.graph.nodes) == saved_nodes
            assert set(index.graph.edges) == saved_edges
            assert result.cycle[0][0] == result.cycle[-1][1]
            for u, v in result.cycle:
                assert candidate.has_edge(u, v)
