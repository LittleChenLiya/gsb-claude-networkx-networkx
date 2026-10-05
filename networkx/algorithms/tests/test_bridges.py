"""Unit tests for bridge-finding algorithms."""

import pytest

import networkx as nx


class TestBridges:
    """Unit tests for the bridge-finding function."""

    def test_single_bridge(self):
        edges = [
            # DFS tree edges.
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (5, 9),
            (9, 10),
            # Nontree edges.
            (1, 3),
            (1, 4),
            (2, 5),
            (5, 10),
            (6, 8),
        ]
        G = nx.Graph(edges)
        source = 1
        bridges = list(nx.bridges(G, source))
        assert bridges == [(5, 6)]

    def test_barbell_graph(self):
        # The (3, 0) barbell graph has two triangles joined by a single edge.
        G = nx.barbell_graph(3, 0)
        source = 0
        bridges = list(nx.bridges(G, source))
        assert bridges == [(2, 3)]

    def test_multiedge_bridge(self):
        edges = [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 4),
        ]
        G = nx.MultiGraph(edges)
        assert list(nx.bridges(G)) == [(2, 3)]


class TestHasBridges:
    """Unit tests for the has bridges function."""

    def test_single_bridge(self):
        edges = [
            # DFS tree edges.
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (5, 6),  # The only bridge edge
            (6, 7),
            (7, 8),
            (5, 9),
            (9, 10),
            # Nontree edges.
            (1, 3),
            (1, 4),
            (2, 5),
            (5, 10),
            (6, 8),
        ]
        G = nx.Graph(edges)
        assert nx.has_bridges(G)  # Default root
        assert nx.has_bridges(G, root=1)  # arbitrary root in G

    def test_has_bridges_raises_root_not_in_G(self):
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        with pytest.raises(nx.NodeNotFound):
            nx.has_bridges(G, root=6)

    def test_multiedge_bridge(self):
        edges = [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 4),
        ]
        G = nx.MultiGraph(edges)
        assert nx.has_bridges(G)
        # Make every edge a multiedge
        G.add_edges_from([(0, 1), (0, 2), (2, 3)])
        assert not nx.has_bridges(G)

    def test_bridges_multiple_components(self):
        G = nx.Graph()
        nx.add_path(G, [0, 1, 2])  # One connected component
        nx.add_path(G, [4, 5, 6])  # Another connected component
        assert list(nx.bridges(G, root=4)) == [(4, 5), (5, 6)]


class TestLocalBridges:
    """Unit tests for the local_bridge function."""

    @classmethod
    def setup_class(cls):
        cls.BB = nx.barbell_graph(4, 0)
        cls.square = nx.cycle_graph(4)
        cls.tri = nx.cycle_graph(3)

    def test_nospan(self):
        expected = {(3, 4), (4, 3)}
        assert next(nx.local_bridges(self.BB, with_span=False)) in expected
        assert set(nx.local_bridges(self.square, with_span=False)) == self.square.edges
        assert list(nx.local_bridges(self.tri, with_span=False)) == []

    def test_no_weight(self):
        inf = float("inf")
        expected = {(3, 4, inf), (4, 3, inf)}
        assert next(nx.local_bridges(self.BB)) in expected
        expected = {(u, v, 3) for u, v in self.square.edges}
        assert set(nx.local_bridges(self.square)) == expected
        assert list(nx.local_bridges(self.tri)) == []

    def test_weight(self):
        inf = float("inf")
        G = self.square.copy()

        G.edges[1, 2]["weight"] = 2
        expected = {(u, v, 5 - wt) for u, v, wt in G.edges(data="weight", default=1)}
        assert set(nx.local_bridges(G, weight="weight")) == expected

        expected = {(u, v, 6) for u, v in G.edges}
        lb = nx.local_bridges(G, weight=lambda u, v, d: 2)
        assert set(lb) == expected


class TestBridgeContraction:
    """Unit tests for the bridge_contraction function."""

    def test_barbell_two_blocks(self):
        # Two triangles joined by a single bridge form two blocks.
        G = nx.barbell_graph(3, 0)
        H = nx.bridge_contraction(G)
        assert list(H.nodes) == [0, 1]
        assert list(H.edges) == [(0, 1)]
        assert H.nodes[0]["sources"] == (0, 1, 2)
        assert H.nodes[1]["sources"] == (3, 4, 5)
        assert H.edges[0, 1]["source"] == (2, 3)
        assert H.graph["mapping"] == {
            0: 0,
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 1,
        }

    def test_parallel_edges_same_block(self):
        # A pair of parallel edges is not a bridge, so both stay in the
        # same block.
        G = nx.MultiGraph([(0, 1), (0, 1), (1, 2)])
        H = nx.bridge_contraction(G)
        assert list(H.nodes) == [0, 1]
        assert list(H.edges) == [(0, 1)]
        assert H.nodes[0]["sources"] == (0, 1)
        assert H.nodes[0]["edges"] == ((0, 1, 0), (0, 1, 1))
        assert H.edges[0, 1]["source"] == (1, 2, 0)
        assert H.graph["mapping"] == {0: 0, 1: 0, 2: 1}

    def test_isolated_node_own_block(self):
        # A triangle stays in one block; an isolated node forms its own.
        G = nx.Graph([(0, 1), (1, 2), (2, 0)])
        G.add_node(3)
        H = nx.bridge_contraction(G)
        assert list(H.nodes) == [0, 1]
        assert H.nodes[0]["sources"] == (0, 1, 2)
        assert H.nodes[1]["sources"] == (3,)
        assert H.nodes[1]["edges"] == ()
        assert H.graph["mapping"] == {0: 0, 1: 0, 2: 0, 3: 1}

    def test_self_loop_stays_in_block(self):
        G = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 0)])
        H = nx.bridge_contraction(G)
        assert list(H.nodes) == [0]
        assert list(H.edges) == []
        assert H.nodes[0]["sources"] == (0, 1, 2)
        assert (0, 0) in H.nodes[0]["edges"]

    def test_block_order_by_first_appearance(self):
        G = nx.MultiGraph()
        G.add_nodes_from([5, 3, 1, 2, 4])
        G.add_edges_from([(5, 3), (3, 1), (1, 5)])  # triangle {5,3,1}
        G.add_edges_from([(2, 4), (2, 4)])  # parallel edges -> not a bridge
        G.add_edge(1, 2)  # bridge between the two blocks
        H = nx.bridge_contraction(G)
        assert H.nodes[0]["sources"] == (5, 3, 1)
        assert H.nodes[1]["sources"] == (2, 4)
        assert H.edges[0, 1]["source"] == (1, 2, 0)

    def test_cycle_single_block(self):
        # A cycle has no bridges, so it contracts to a single block.
        G = nx.cycle_graph(5)
        H = nx.bridge_contraction(G)
        assert list(H.nodes) == [0]
        assert list(H.edges) == []
        assert H.nodes[0]["sources"] == tuple(range(5))
        assert len(H.nodes[0]["edges"]) == 5

    def test_path_each_node_own_block(self):
        # Every edge in a path is a bridge, so each node is its own block.
        G = nx.path_graph(4)
        H = nx.bridge_contraction(G)
        assert list(H.nodes) == [0, 1, 2, 3]
        assert list(H.edges) == [(0, 1), (1, 2), (2, 3)]
        assert H.graph["mapping"] == {0: 0, 1: 1, 2: 2, 3: 3}
        for i in range(4):
            assert H.nodes[i]["sources"] == (i,)

    def test_empty_graph(self):
        G = nx.Graph()
        H = nx.bridge_contraction(G)
        assert list(H.nodes) == []
        assert list(H.edges) == []
        assert H.graph["mapping"] == {}

    def test_independent_edge_attributes(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        G.add_edge(0, 3, weight=42, label="bridge")
        H = nx.bridge_contraction(G)
        assert H.edges[0, 1]["weight"] == 42
        assert H.edges[0, 1]["label"] == "bridge"
        # Mutating H must not affect G.
        H.edges[0, 1]["weight"] = 999
        H.edges[0, 1]["new_attr"] = "x"
        assert G[0][3]["weight"] == 42
        assert "new_attr" not in G[0][3]

    def test_independent_graph_attributes(self):
        G = nx.Graph([(0, 1)])
        G.graph["info"] = {"a": [1, 2, 3]}
        H = nx.bridge_contraction(G)
        # Mutating H must not affect G.
        H.graph["info"]["a"].append(4)
        assert G.graph["info"]["a"] == [1, 2, 3]

    def test_independent_node_sources(self):
        G = nx.barbell_graph(3, 0)
        H = nx.bridge_contraction(G)
        # sources is a tuple; mutating it must not affect G.
        sources = H.nodes[0]["sources"]
        assert sources == (0, 1, 2)
        assert list(G.nodes) == [0, 1, 2, 3, 4, 5]

    def test_create_using(self):
        G = nx.MultiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        G.add_edge(0, 3)
        H = nx.bridge_contraction(G, create_using=nx.MultiGraph)
        assert isinstance(H, nx.MultiGraph)
        assert list(H.edges(keys=True)) == [(0, 1, 0)]
        assert H.edges[0, 1, 0]["source"] == (0, 3, 0)

    def test_directed_raises(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.bridge_contraction(nx.DiGraph([(0, 1)]))

    def test_disconnected_forest(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # triangle
        G.add_edge(3, 4)  # edge
        G.add_node(5)  # isolated
        H = nx.bridge_contraction(G)
        assert list(H.nodes) == [0, 1, 2, 3]
        assert list(H.edges) == [(1, 2)]
        assert H.nodes[0]["sources"] == (0, 1, 2)
        assert H.nodes[1]["sources"] == (3,)
        assert H.nodes[2]["sources"] == (4,)
        assert H.nodes[3]["sources"] == (5,)
        assert H.edges[1, 2]["source"] == (3, 4)
