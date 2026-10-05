"""Tests for the bridge contraction view.

Run only these tests with::

    pytest networkx/algorithms/tests/test_bridge_contraction.py
"""

import pytest

import networkx as nx
from networkx.algorithms.bridge_contraction import BridgeContraction


def make_two_triangles():
    """Two triangles joined by a single bridge (2, 3)."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    G.add_edges_from([(3, 4), (4, 5), (5, 3)])
    G.add_edge(2, 3, color="red")
    return G


class TestTwoTriangles:
    def test_two_blocks(self):
        bc = nx.bridge_contraction(make_two_triangles())
        assert len(bc) == 2
        assert bc.blocks() == [frozenset({0, 1, 2}), frozenset({3, 4, 5})]
        assert set(bc.graph.nodes()) == {0, 1}
        assert set(bc.graph.edges()) == {(0, 1)}

    def test_membership_queries(self):
        bc = nx.bridge_contraction(make_two_triangles())
        assert bc.block_of(0) == 0
        assert bc.block_of(4) == 1
        assert bc.members(0) == frozenset({0, 1, 2})
        assert bc.members(1) == frozenset({3, 4, 5})
        assert bc.graph.nodes[1]["members"] == frozenset({3, 4, 5})

    def test_bridge_identity_queries(self):
        bc = nx.bridge_contraction(make_two_triangles())
        assert bc.bridges() == [(2, 3)]
        assert bc.is_bridge(2, 3)
        assert bc.is_bridge(3, 2)  # undirected: either orientation
        assert not bc.is_bridge(0, 1)
        assert bc.bridge_between(0, 1) == (2, 3)
        assert bc.bridge_between(0, 0) is None
        assert bc.graph.edges[0, 1]["bridge"] == (2, 3)

    def test_contracted_edge_keeps_independent_attr_copy(self):
        G = make_two_triangles()
        bc = nx.bridge_contraction(G)
        assert bc.graph.edges[0, 1]["color"] == "red"
        # Mutating the copy must not touch the source graph.
        bc.graph.edges[0, 1]["color"] = "blue"
        bc.graph.graph["name"] = "mutated"
        assert G[2][3]["color"] == "red"
        assert "name" not in G.graph

    def test_source_graph_unmodified(self):
        G = make_two_triangles()
        G.graph["meta"] = {"tags": ["demo"]}
        before_edges = sorted(G.edges())
        nx.bridge_contraction(G)
        assert sorted(G.edges()) == before_edges
        assert G[2][3] == {"color": "red"}
        assert G.graph["meta"] == {"tags": ["demo"]}


class TestMultiGraph:
    def test_parallel_edges_not_a_bridge(self):
        M = nx.MultiGraph()
        M.add_edges_from([(0, 1), (0, 1)])
        bc = nx.bridge_contraction(M)
        assert len(bc) == 1
        assert bc.blocks() == [frozenset({0, 1})]
        assert bc.bridges() == []
        assert bc.graph.number_of_edges() == 0
        assert not bc.is_bridge(0, 1, key=0)
        assert not bc.is_bridge(0, 1, key=1)

    def test_parallel_edges_with_real_bridge(self):
        # Parallel pair on one side, single bridge to a triangle.
        M = nx.MultiGraph()
        M.add_edge(0, 1, key="a")
        M.add_edge(0, 1, key="b")
        M.add_edge(1, 2, key="link")
        M.add_edges_from([(2, 3), (3, 4), (4, 2)])
        bc = nx.bridge_contraction(M)
        assert bc.blocks() == [frozenset({0, 1}), frozenset({2, 3, 4})]
        assert bc.bridges() == [(1, 2, "link")]
        assert bc.is_bridge(1, 2, key="link")
        assert not bc.is_bridge(0, 1, key="a")
        assert bc.bridge_between(0, 1) == (1, 2, "link")
        assert isinstance(bc.graph, nx.MultiGraph)
        assert bc.graph.has_edge(0, 1, key="link")

    def test_multigraph_bridge_attrs_are_independent_copies(self):
        M = nx.MultiGraph()
        M.add_edge(0, 1, weight=[1, 2])
        M.add_edge(1, 2)
        M.add_edge(2, 0)
        M.add_edge(2, 3, payload={"x": 1})  # the bridge
        bc = nx.bridge_contraction(M)
        bc.graph.edges[0, 1, 0]["payload"]["x"] = 99
        assert M[2][3][0]["payload"] == {"x": 1}


class TestSelfLoopsAndIsolates:
    def test_self_loop_stays_in_block(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 0)])
        G.add_edge(2, 3)
        bc = nx.bridge_contraction(G)
        assert bc.blocks() == [frozenset({0, 1, 2}), frozenset({3})]
        assert not bc.is_bridge(0, 0)
        # The self-loop does not appear in the contracted graph.
        assert bc.graph.number_of_edges() == 1
        assert nx.number_of_selfloops(bc.graph) == 0

    def test_isolated_node_gets_own_block(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        G.add_node("lonely")
        bc = nx.bridge_contraction(G)
        assert bc.blocks() == [frozenset({0, 1, 2}), frozenset({"lonely"})]
        assert bc.block_of("lonely") == 1
        assert bc.members(1) == frozenset({"lonely"})
        assert bc.graph.number_of_edges() == 0

    def test_empty_graph(self):
        bc = nx.bridge_contraction(nx.Graph())
        assert len(bc) == 0
        assert bc.blocks() == []
        assert bc.bridges() == []


class TestBlockOrder:
    def test_block_order_follows_input_node_order(self):
        # Node "z" appears first, so its block gets id 0 even though the
        # bridge is listed first among edges; isolated "c" comes last.
        G = nx.Graph()
        G.add_node("z")
        G.add_edge("a", "b")  # bridge between the "a" block and the "b" block
        G.add_node("c")
        bc = nx.bridge_contraction(G)
        assert bc.blocks() == [
            frozenset({"z"}),
            frozenset({"a"}),
            frozenset({"b"}),
            frozenset({"c"}),
        ]
        assert bc.block_of("z") == 0
        # Deterministic across repeated calls.
        again = nx.bridge_contraction(G)
        assert again.blocks() == bc.blocks()

    def test_iteration_yields_block_ids_in_order(self):
        bc = nx.bridge_contraction(make_two_triangles())
        assert list(bc) == [0, 1]


class TestErrors:
    def test_directed_graph_rejected(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.bridge_contraction(nx.DiGraph([(0, 1)]))

    def test_unknown_node_raises(self):
        bc = nx.bridge_contraction(make_two_triangles())
        with pytest.raises(nx.NetworkXError):
            bc.block_of("missing")

    def test_return_type(self):
        assert isinstance(nx.bridge_contraction(make_two_triangles()), BridgeContraction)

