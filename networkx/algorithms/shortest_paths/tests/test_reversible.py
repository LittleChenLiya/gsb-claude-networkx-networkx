import pytest

import networkx as nx
from networkx.algorithms.shortest_paths.reversible import (
    TraversedEdge,
    minimum_reversals_path,
)


def edge(u, v, reversed=False, key=None):
    return TraversedEdge(u, v, key, reversed)


class TestMinimumReversalsPath:
    def test_acceptance_basic_scenario(self):
        # edges a->b and c->b, only c->b reversible: a to c needs one reversal
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("c", "b", reversible=True)

        n, edges = minimum_reversals_path(G, "a", "c")
        assert n == 1
        assert edges == [edge("a", "b"), edge("b", "c", reversed=True)]
        assert sum(e.reversed for e in edges) == 1
        path_nodes = ["a"] + [e.v for e in edges]
        assert path_nodes == ["a", "b", "c"]

    def test_reversibility_off_means_no_path(self):
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("c", "b", reversible=True)
        G["c"]["b"]["reversible"] = False

        with pytest.raises(nx.NetworkXNoPath):
            minimum_reversals_path(G, "a", "c")

    def test_missing_attribute_means_not_reversible(self):
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("c", "b")
        with pytest.raises(nx.NetworkXNoPath):
            minimum_reversals_path(G, "a", "c")

    def test_zero_cost_forward_edge_preferred_over_reversal(self):
        # Cost is compared first, before any insertion-order tie-break.
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("c", "b", reversible=True)
        G.add_edge("a", "c")  # 0-cost route added after the 1-reversal route

        n, edges = minimum_reversals_path(G, "a", "c")
        assert n == 0
        assert edges == [edge("a", "c")]

    def test_cost_zero_beats_one_regardless_of_order(self):
        G = nx.DiGraph()
        G.add_edge("early", "s", reversible=True)
        G.add_edge("s", "late")
        G.add_edge("early", "t")
        G.add_edge("late", "t")

        n, edges = minimum_reversals_path(G, "s", "t")
        assert n == 0
        assert [e.u for e in edges] == ["s", "late"]

    def test_source_equals_target(self):
        G = nx.DiGraph()
        G.add_edge("a", "b")
        n, edges = minimum_reversals_path(G, "a", "a")
        assert n == 0
        assert edges == []

    def test_multiple_reversals_and_cycle(self):
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        G.add_edge("a", "d", reversible=True)
        G.add_edge("d", "c", reversible=True)

        n, edges = minimum_reversals_path(G, "c", "b")
        assert n == 2
        assert edges == [
            edge("c", "d", reversed=True),
            edge("d", "a", reversed=True),
            edge("a", "b"),
        ]

    def test_tie_break_by_node_edge_insertion_order(self):
        G = nx.DiGraph()
        G.add_edge("b1", "a", reversible=True)
        G.add_edge("b2", "a", reversible=True)
        G.add_edge("b1", "c")
        G.add_edge("b2", "c")
        n, edges = minimum_reversals_path(G, "a", "c")
        assert n == 1
        assert [e.v for e in edges] == ["b1", "c"]

        H = nx.DiGraph()
        H.add_edge("b2", "a", reversible=True)
        H.add_edge("b1", "a", reversible=True)
        H.add_edge("b2", "c")
        H.add_edge("b1", "c")
        n, edges = minimum_reversals_path(H, "a", "c")
        assert n == 1
        assert [e.v for e in edges] == ["b2", "c"]

    def test_parallel_edges_return_exact_key(self):
        G = nx.MultiDiGraph()
        k0 = G.add_edge("a", "b")
        k1 = G.add_edge("a", "b", reversible=True)
        G.add_edge("b", "c")
        assert (k0, k1) == (0, 1)

        n, edges = minimum_reversals_path(G, "a", "c")
        assert n == 0
        assert edges[0] == edge("a", "b", key=0)
        assert edges[1] == edge("b", "c", key=0)

        H = nx.MultiDiGraph()
        H.add_edge("b", "a", reversible=True)  # key 0
        H.add_edge("b", "a", reversible=True)  # key 1
        n, edges = minimum_reversals_path(H, "a", "b")
        assert n == 1
        assert edges == [edge("a", "b", reversed=True, key=0)]

    def test_parallel_edges_one_reversible_one_not(self):
        G = nx.MultiDiGraph()
        G.add_edge("b", "a")  # key 0, not reversible
        G.add_edge("b", "a", reversible=True)  # key 1
        n, edges = minimum_reversals_path(G, "a", "b")
        assert n == 1
        assert edges == [edge("a", "b", reversed=True, key=1)]

    def test_self_loop_does_not_move_node(self):
        G = nx.DiGraph()
        G.add_edge("a", "a")
        G.add_edge("a", "a", reversible=True)
        G.add_node("b")
        with pytest.raises(nx.NetworkXNoPath):
            minimum_reversals_path(G, "a", "b")

        G.add_edge("a", "b")
        _n, edges = minimum_reversals_path(G, "a", "b")
        assert edges == [edge("a", "b")]
        assert all(e.u != e.v for e in edges)

    def test_self_loop_multigraph(self):
        G = nx.MultiDiGraph()
        G.add_edge("a", "a", key="loop", reversible=True)
        G.add_node("b")
        with pytest.raises(nx.NetworkXNoPath):
            minimum_reversals_path(G, "a", "b")

    def test_query_does_not_modify_graph(self):
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("c", "b", reversible=True)
        before = list(G.edges(data=True))
        minimum_reversals_path(G, "a", "c")
        assert list(G.edges(data=True)) == before
        assert set(G) == {"a", "b", "c"}
        assert G.number_of_edges() == 2

    def test_unreachable_raises_not_empty_success(self):
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_node("c")
        with pytest.raises(nx.NetworkXNoPath):
            minimum_reversals_path(G, "a", "c")

    def test_node_not_found(self):
        G = nx.DiGraph()
        G.add_edge("a", "b")
        with pytest.raises(nx.NodeNotFound):
            minimum_reversals_path(G, "a", "z")
        with pytest.raises(nx.NodeNotFound):
            minimum_reversals_path(G, "z", "a")

    def test_undirected_graph_rejected(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        with pytest.raises(nx.NetworkXNotImplemented):
            minimum_reversals_path(G, 1, 2)

    @pytest.mark.parametrize("value", [True, 1, "yes", 2])
    def test_truthy_reversible_attribute(self, value):
        G = nx.DiGraph()
        G.add_edge("b", "a", reversible=value)
        n, edges = minimum_reversals_path(G, "a", "b")
        assert n == 1
        assert edges == [edge("a", "b", reversed=True)]

    @pytest.mark.parametrize("value", [False, 0, None, ""])
    def test_falsy_reversible_attribute(self, value):
        G = nx.DiGraph()
        G.add_edge("b", "a", reversible=value)
        with pytest.raises(nx.NetworkXNoPath):
            minimum_reversals_path(G, "a", "b")

    def test_custom_attribute_name(self):
        G = nx.DiGraph()
        G.add_edge("b", "a", can_backtrack=True)
        with pytest.raises(nx.NetworkXNoPath):
            minimum_reversals_path(G, "a", "b")
        n, edges = minimum_reversals_path(G, "a", "b", reversible="can_backtrack")
        assert n == 1
        assert edges == [edge("a", "b", reversed=True)]

    def test_traversed_edge_is_tuple_with_fields(self):
        e = TraversedEdge("u", "v", 7, True)
        assert tuple(e) == ("u", "v", 7, True)
        assert e.u == "u"
        assert e.v == "v"
        assert e.key == 7
        assert e.reversed is True
        u, v, key, rev = e
        assert (u, v, key, rev) == ("u", "v", 7, True)

    def test_isolated_source_target(self):
        G = nx.DiGraph()
        G.add_node("x")
        assert minimum_reversals_path(G, "x", "x") == (0, [])

    def test_nx_namespace(self):
        assert nx.minimum_reversals_path is minimum_reversals_path
        assert nx.TraversedEdge is TraversedEdge
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        n, edges = minimum_reversals_path(G, "a", "c")
        assert n == 0
        assert [e.v for e in edges] == ["b", "c"]
