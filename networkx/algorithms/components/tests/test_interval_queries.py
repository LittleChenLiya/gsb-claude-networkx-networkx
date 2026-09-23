"""Tests for offline interval connected-component queries."""

import copy
import random
from itertools import pairwise

import pytest

import networkx as nx
from networkx.utils import graphs_equal


def brute_force_components(G, times, intervals_attr="intervals"):
    """Reference answer: rebuild the active graph at every query time."""
    answers = []
    for t in times:
        H = nx.Graph()
        H.add_nodes_from(G)
        for u, v, data in G.edges(data=True):
            raw = data.get(intervals_attr)
            if raw is None or any(start <= t < end for start, end in raw):
                H.add_edge(u, v)
        answers.append([set(c) for c in nx.connected_components(H)])
    return answers


def as_frozensets(answer):
    return {frozenset(c) for c in answer}


def assert_insertion_order(G, answer):
    """Components and their nodes must follow node insertion order."""
    rank = {node: i for i, node in enumerate(G)}
    for components in answer:
        positions = []
        for component in components:
            ranks = [rank[node] for node in component]
            assert ranks == sorted(ranks)
            positions.append(ranks[0])
        assert positions == sorted(positions)


class TestAcceptanceExample:
    def test_three_node_chain(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[(0, 4)])
        G.add_edge("b", "c", intervals=[(2, 6)])
        result = nx.interval_connected_components(G, [4, 1, 2, 2])
        assert result == [
            [["a"], ["b", "c"]],
            [["a", "b"], ["c"]],
            [["a", "b", "c"]],
            [["a", "b", "c"]],
        ]

    def test_duplicate_results_are_independent_lists(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[(0, 4)])
        result = nx.interval_connected_components(G, [1, 1])
        assert result[0] == result[1]
        assert result[0] is not result[1]
        assert result[0][0] is not result[1][0]


class TestIntervalSemantics:
    def test_missing_attribute_is_always_active(self):
        G = nx.Graph()
        G.add_edge("a", "b")
        result = nx.interval_connected_components(G, [-10, 0, 10**9])
        assert result == [[["a", "b"]]] * 3

    def test_empty_interval_list_is_never_active(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[])
        result = nx.interval_connected_components(G, [0, 100])
        assert result == [[["a"], ["b"]], [["a"], ["b"]]]

    def test_overlapping_intervals_are_merged(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[(0, 3), (2, 5), (9, 10), (5, 6)])
        result = nx.interval_connected_components(G, [0, 4, 5, 6, 9, 10])
        expected = [
            [["a", "b"]],
            [["a", "b"]],
            [["a", "b"]],
            [["a"], ["b"]],
            [["a", "b"]],
            [["a"], ["b"]],
        ]
        assert result == expected

    def test_negative_and_far_times(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[(-(10**9), -(10**9) + 2)])
        result = nx.interval_connected_components(G, [-(10**9), -(10**9) + 1, 0])
        assert result == [[["a", "b"]], [["a", "b"]], [["a"], ["b"]]]

    def test_self_loop_does_not_change_components(self):
        G = nx.Graph()
        G.add_edge("a", "a", intervals=[(0, 10)])
        G.add_node("b")
        result = nx.interval_connected_components(G, [0, 5])
        assert result == [[["a"], ["b"]], [["a"], ["b"]]]

    def test_isolated_node_in_every_answer(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[(0, 2)])
        G.add_node("z")
        result = nx.interval_connected_components(G, [1, 3])
        assert result == [[["a", "b"], ["z"]], [["a"], ["b"], ["z"]]]

    def test_empty_times(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[(0, 2)])
        assert nx.interval_connected_components(G, []) == []

    def test_null_graph(self):
        G = nx.Graph()
        assert nx.interval_connected_components(G, [0, 1]) == [[], []]

    def test_custom_attribute_name(self):
        G = nx.Graph()
        G.add_edge("a", "b", active=[(0, 2)])
        result = nx.interval_connected_components(G, [1, 2], intervals_attr="active")
        assert result == [[["a", "b"]], [["a"], ["b"]]]


class TestMultiGraph:
    def test_parallel_edges_fill_gaps(self):
        G = nx.MultiGraph()
        G.add_edge("a", "b", intervals=[(0, 2)])
        G.add_edge("a", "b", intervals=[(2, 4)])
        result = nx.interval_connected_components(G, [1, 2, 3, 4])
        assert result == [
            [["a", "b"]],
            [["a", "b"]],
            [["a", "b"]],
            [["a"], ["b"]],
        ]

    def test_parallel_edges_with_disjoint_gaps(self):
        G = nx.MultiGraph()
        G.add_edge("a", "b", intervals=[(0, 1)])
        G.add_edge("a", "b", intervals=[(3, 4)])
        result = nx.interval_connected_components(G, [0, 2, 3])
        assert result == [[["a", "b"]], [["a"], ["b"]], [["a", "b"]]]

    def test_parallel_edge_without_attribute_wins(self):
        G = nx.MultiGraph()
        G.add_edge("a", "b", intervals=[])
        G.add_edge("a", "b")
        result = nx.interval_connected_components(G, [0, 10**6])
        assert result == [[["a", "b"]], [["a", "b"]]]


class TestMixedNodeTypes:
    def test_incomparable_nodes(self):
        G = nx.Graph()
        G.add_node(("tuple", 1))
        G.add_node(7)
        G.add_node("seven")
        G.add_node(frozenset({1}))
        G.add_edge(("tuple", 1), 7, intervals=[(0, 3)])
        G.add_edge(7, "seven", intervals=[(1, 2)])
        result = nx.interval_connected_components(G, [0, 1, 3])
        assert result == [
            [[("tuple", 1), 7], ["seven"], [frozenset({1})]],
            [[("tuple", 1), 7, "seven"], [frozenset({1})]],
            [[("tuple", 1)], [7], ["seven"], [frozenset({1})]],
        ]
        assert_insertion_order(G, result)


class TestValidation:
    def test_directed_graph_rejected(self):
        for cls in (nx.DiGraph, nx.MultiDiGraph):
            G = cls()
            G.add_edge("a", "b", intervals=[(0, 1)])
            with pytest.raises(nx.NetworkXNotImplemented):
                nx.interval_connected_components(G, [0])

    @pytest.mark.parametrize(
        "intervals",
        [
            [(True, 4)],  # bool start
            [(0, False)],  # bool end
            [(0.5, 4)],  # float endpoint
            [("0", 4)],  # string endpoint
            [(4, 4)],  # empty half-open interval
            [(5, 4)],  # start after end
            [(0, 1, 2)],  # not a pair
            [0],  # not a pair at all
        ],
    )
    def test_invalid_intervals_rejected(self, intervals):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=intervals)
        with pytest.raises((TypeError, ValueError)):
            nx.interval_connected_components(G, [0])

    def test_intervals_attribute_not_a_list(self):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=7)
        with pytest.raises(TypeError):
            nx.interval_connected_components(G, [0])

    @pytest.mark.parametrize("t", [True, 1.5, "1", None])
    def test_invalid_query_time_rejected(self, t):
        G = nx.Graph()
        G.add_edge("a", "b", intervals=[(0, 4)])
        with pytest.raises(TypeError):
            nx.interval_connected_components(G, [0, t])

    def test_failed_query_does_not_mutate_graph(self):
        G = nx.MultiGraph()
        G.add_edge("a", "b", intervals=[(4, 4)])
        G.add_edge("b", "c", intervals=[(0, 1)])
        before = copy.deepcopy(G)
        with pytest.raises((TypeError, ValueError)):
            nx.interval_connected_components(G, [0])
        assert graphs_equal(G, before)

    def test_successful_query_does_not_mutate_graph(self):
        G = nx.MultiGraph()
        G.add_edge("a", "b", intervals=[(0, 4), (2, 6)])
        G.add_edge("b", "c")
        G.add_edge("c", "c", intervals=[(0, 10)])
        before = copy.deepcopy(G)
        nx.interval_connected_components(G, [0, 3, 5, 7])
        assert graphs_equal(G, before)


class TestAgainstBruteForce:
    @pytest.mark.parametrize("seed", range(10))
    def test_random_multigraph(self, seed):
        rng = random.Random(seed)
        G = nx.MultiGraph()
        nodes = list(range(6))
        G.add_nodes_from(nodes)
        for _ in range(10):
            u, v = rng.choice(nodes), rng.choice(nodes)
            kind = rng.randrange(4)
            if kind == 0:
                G.add_edge(u, v)  # no attribute: always active
            elif kind == 1:
                G.add_edge(u, v, intervals=[])  # never active
            else:
                ivs = []
                for _ in range(rng.randrange(1, 4)):
                    s = rng.randrange(-8, 8)
                    ivs.append((s, s + rng.randrange(1, 5)))
                G.add_edge(u, v, intervals=ivs)
        times = [rng.randrange(-10, 10) for _ in range(12)]
        result = nx.interval_connected_components(G, times)
        expected = brute_force_components(G, times)
        assert len(result) == len(times)
        for got, want in zip(result, expected):
            assert as_frozensets(got) == as_frozensets(want)
        assert_insertion_order(G, result)

    def test_random_graph_with_mixed_nodes(self):
        rng = random.Random(42)
        G = nx.Graph()
        nodes = ["x", 3, (1, 2), frozenset({9}), "y", 2.5]
        G.add_nodes_from(nodes)
        for u, v in pairwise(nodes):
            if rng.random() < 0.8:
                s = rng.randrange(-5, 5)
                G.add_edge(u, v, intervals=[(s, s + 2)])
        times = list(range(-6, 7)) + [0, 0]
        result = nx.interval_connected_components(G, times)
        expected = brute_force_components(G, times)
        for got, want in zip(result, expected):
            assert as_frozensets(got) == as_frozensets(want)
        assert_insertion_order(G, result)
