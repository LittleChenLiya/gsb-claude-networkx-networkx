"""Tests for temporal_connected_components."""

import random

import pytest

import networkx as nx
from networkx.algorithms.connectivity import temporal_connected_components


def brute_force(G, times, attr="intervals"):
    """Per-query rebuild + connected_components as the reference."""
    out = []
    for t in times:
        H = nx.Graph()
        H.add_nodes_from(G.nodes)
        if G.is_multigraph():
            edges = G.edges(data=True)
        else:
            edges = G.edges(data=True)
        for u, v, data in edges:
            intervals = data.get(attr)
            if intervals is None or any(s <= t < e for s, e in intervals):
                H.add_edge(u, v)
        out.append(
            [sorted(c, key=list(G.nodes).index) for c in nx.connected_components(H)]
        )
    return out


def as_frozensets(result):
    return [
        sorted((frozenset(c) for c in comps), key=lambda c: min(map(id, c)))
        for comps in result
    ]


def same_components(result, expected):
    return [sorted(map(frozenset, r), key=repr) for r in result] == [
        sorted(map(frozenset, e), key=repr) for e in expected
    ]


class TestAcceptance:
    def test_three_node_chain(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[(0, 4)])
        G.add_edge("b", "c", intervals=[(2, 6)])
        assert temporal_connected_components(G, [4, 1, 2, 2]) == [
            [["a"], ["b", "c"]],
            [["a", "b"], ["c"]],
            [["a", "b", "c"]],
            [["a", "b", "c"]],
        ]

    def test_empty_times(self):
        G = nx.Graph([(1, 2)])
        assert temporal_connected_components(G, []) == []


class TestSemantics:
    def test_parallel_edges_fill_gap(self):
        G = nx.MultiGraph()
        G.add_edge("x", "y", intervals=[(0, 2)])
        G.add_edge("x", "y", intervals=[(2, 4)])  # gap-free together
        G.add_edge("x", "y", intervals=[])  # never active
        result = temporal_connected_components(G, [0, 1, 2, 3, 4])
        assert result == [
            [["x", "y"]],
            [["x", "y"]],
            [["x", "y"]],
            [["x", "y"]],
            [["x"], ["y"]],
        ]

    def test_overlapping_intervals_merged(self):
        G = nx.Graph()
        G.add_edge(1, 2, intervals=[(0, 3), (2, 5), (10, 12)])
        result = temporal_connected_components(G, [1, 4, 5, 11])
        assert result == [[[1, 2]], [[1, 2]], [[1], [2]], [[1, 2]]]

    def test_missing_attribute_always_active(self):
        G = nx.Graph()
        G.add_edge("a", "b")  # no intervals attribute
        G.add_node("iso")
        result = temporal_connected_components(G, [-(10**18), 0, 10**18])
        assert result == [[["a", "b"], ["iso"]]] * 3

    def test_empty_interval_list_never_active(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[])
        assert temporal_connected_components(G, [0]) == [[["a"], ["b"]]]

    def test_self_loop_ignored(self):
        G = nx.Graph()
        G.add_edge("a", "a", intervals=[(0, 10)])
        G.add_edge("a", "b", intervals=[(0, 10)])
        assert temporal_connected_components(G, [5]) == [[["a", "b"]]]

    def test_isolated_nodes_present_everywhere(self):
        G = nx.Graph()
        G.add_nodes_from(["z", "y"])
        G.add_edge("a", "b", intervals=[(0, 1)])
        result = temporal_connected_components(G, [0, 1])
        assert result == [
            [["z"], ["y"], ["a", "b"]],
            [["z"], ["y"], ["a"], ["b"]],
        ]

    def test_mixed_unorderable_nodes(self):
        marker = object()
        G = nx.Graph()
        for node in ["s", 7, (1, 2), marker, 3.5]:
            G.add_node(node)
        G.add_edge("s", 7, intervals=[(0, 5)])
        G.add_edge(7, (1, 2), intervals=[(0, 5)])
        result = temporal_connected_components(G, [0, 5])
        assert result[0] == [["s", 7, (1, 2)], [marker], [3.5]]
        assert result[1] == [["s"], [7], [(1, 2)], [marker], [3.5]]

    def test_far_negative_times(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[(-(10**12), -(10**6))])
        result = temporal_connected_components(
            G, [-(10**12), -(10**6) - 1, -(10**6), 0]
        )
        assert result == [
            [["a", "b"]],
            [["a", "b"]],
            [["a"], ["b"]],
            [["a"], ["b"]],
        ]

    def test_stable_insertion_order(self):
        G = nx.Graph()
        G.add_nodes_from(["c", "a", "b"])
        G.add_edge("b", "c", intervals=[(0, 10)])
        (result,) = temporal_connected_components(G, [5])
        assert result == [["c", "b"], ["a"]]

    def test_duplicate_times_preserved(self):
        G = nx.Graph()
        G.add_edge(1, 2, intervals=[(0, 2)])
        result = temporal_connected_components(G, [1, 1, 1])
        assert result == [[[1, 2]], [[1, 2]], [[1, 2]]]

    def test_multigraph_any_active_parallel_edge_connects(self):
        G = nx.MultiGraph()
        G.add_edge("u", "v", intervals=[(0, 1)])
        G.add_edge("u", "v", intervals=[(5, 6)])
        result = temporal_connected_components(G, [0, 3, 5])
        assert result == [[["u", "v"]], [["u"], ["v"]], [["u", "v"]]]

    def test_graph_unmodified_on_success(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[(3, 1) if False else (1, 3), (0, 1)])
        before_edges = {tuple(e): dict(d) for *e, d in G.edges(data=True)}
        before_nodes = list(G.nodes(data=True))
        temporal_connected_components(G, [0, 2])
        assert {tuple(e): dict(d) for *e, d in G.edges(data=True)} == before_edges
        assert list(G.nodes(data=True)) == before_nodes


class TestValidation:
    def test_directed_rejected(self):
        G = nx.DiGraph([(1, 2)])
        with pytest.raises(nx.NetworkXNotImplemented):
            temporal_connected_components(G, [0])

    @pytest.mark.parametrize(
        "intervals",
        [
            [(True, 4)],  # bool endpoint
            [(0, False)],  # bool endpoint
            [(1.5, 4)],  # non-integer endpoint
            [(4, 4)],  # empty interval
            [(5, 2)],  # reversed interval
            [(0,)],  # not a pair
            [3],  # not a pair
            "not-intervals",  # not an iterable of pairs
        ],
    )
    def test_invalid_intervals_rejected(self, intervals):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=intervals)
        before = dict(G["a"]["b"])
        with pytest.raises((TypeError, ValueError)):
            temporal_connected_components(G, [0])
        assert dict(G["a"]["b"]) == before  # graph untouched on failure

    @pytest.mark.parametrize("bad_time", [True, 1.5, "3", None])
    def test_invalid_query_time_rejected(self, bad_time):
        G = nx.Graph([(1, 2)])
        with pytest.raises((TypeError, ValueError)):
            temporal_connected_components(G, [0, bad_time])

    def test_graph_unmodified_on_failure(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[(2, 1)])
        snapshot = nx.to_dict_of_dicts(G)
        with pytest.raises(ValueError):
            temporal_connected_components(G, [0])
        assert nx.to_dict_of_dicts(G) == snapshot


class TestAgainstBruteForce:
    @pytest.mark.parametrize("seed", range(20))
    def test_random_graph(self, seed):
        rng = random.Random(seed)
        G = nx.Graph()
        G.add_nodes_from(range(6))
        for u in range(6):
            for v in range(u + 1, 6):
                if rng.random() < 0.4:
                    ivs = []
                    for _ in range(rng.randint(0, 3)):
                        s = rng.randint(-8, 8)
                        ivs.append((s, s + rng.randint(1, 4)))
                    if ivs or rng.random() < 0.5:
                        G.add_edge(u, v, intervals=ivs)
                    else:
                        G.add_edge(u, v)
        times = [rng.randint(-10, 10) for _ in range(12)]
        result = temporal_connected_components(G, times)
        assert same_components(result, brute_force(G, times))

    @pytest.mark.parametrize("seed", range(10))
    def test_random_multigraph(self, seed):
        rng = random.Random(1000 + seed)
        G = nx.MultiGraph()
        G.add_nodes_from(range(5))
        for _ in range(rng.randint(0, 12)):
            u, v = rng.randrange(5), rng.randrange(5)
            ivs = []
            for _ in range(rng.randint(0, 2)):
                s = rng.randint(-6, 6)
                ivs.append((s, s + rng.randint(1, 3)))
            G.add_edge(u, v, intervals=ivs)
        times = [rng.randint(-8, 8) for _ in range(10)]
        result = temporal_connected_components(G, times)
        assert same_components(result, brute_force(G, times))
