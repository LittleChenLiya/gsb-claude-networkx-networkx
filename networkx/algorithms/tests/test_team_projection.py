"""Tests for networkx.algorithms.team_projection."""

import pytest

import networkx as nx
from networkx.algorithms.team_projection import TeamProjection


def sum_weights(attrs):
    return {"weight": sum(a["weight"] for a in attrs)}


def make_xy_graph():
    """Two X nodes each point to a Y node; X also has an internal edge."""
    G = nx.MultiDiGraph()
    G.add_node("x1", team="X")
    G.add_node("x2", team="X")
    G.add_node("y1", team="Y")
    G.add_edge("x1", "y1", weight=2)
    G.add_edge("x2", "y1", weight=3)
    G.add_edge("x1", "x2", weight=10)  # internal to team X
    return G


class TestTeamProjection:
    def test_external_edges_aggregated(self):
        G = make_xy_graph()
        result = nx.team_projection(G, sum_weights)
        assert isinstance(result, TeamProjection)
        H = result.graph
        assert set(H.nodes) == {"X", "Y"}
        edge = H["X"]["Y"][0]
        assert edge["weight"] == 5
        # Both source parallel edges are recorded, in original order.
        assert [(u, v, k) for u, v, k, _ in edge["sources"]] == [
            ("x1", "y1", 0),
            ("x2", "y1", 0),
        ]
        # Attribute snapshots of the sources are kept.
        assert [a["weight"] for _, _, _, a in edge["sources"]] == [2, 3]

    def test_internal_edges_separate(self):
        G = make_xy_graph()
        result = nx.team_projection(G, sum_weights)
        assert [(u, v, k) for u, v, k, _ in result.internal_edges] == [
            ("x1", "x2", 0)
        ]
        assert result.internal_edges[0][3]["weight"] == 10
        # The internal edge never reaches the aggregator / projected graph.
        assert "X" not in result.graph["X"].get("X", {})
        assert result.graph["X"]["Y"][0]["weight"] == 5

    def test_missing_team_produces_no_result(self):
        G = nx.MultiDiGraph()
        G.add_node("a", team="X")
        G.add_node("b")  # no team attribute
        G.add_node("c", team="Y")
        G.add_edge("a", "b", weight=1)  # target lacks team
        G.add_edge("b", "c", weight=2)  # source lacks team
        G.add_edge("a", "c", weight=4)
        result = nx.team_projection(G, sum_weights)
        assert list(result.graph.edges(data=True)) == [
            ("X", "Y", {"weight": 4, "sources": [("a", "c", 0, {"weight": 4})]})
        ]
        assert result.internal_edges == []

    def test_parallel_edge_keys_preserved(self):
        G = nx.MultiDiGraph()
        G.add_node("a", team="X")
        G.add_node("b", team="Y")
        G.add_edge("a", "b", weight=1)
        G.add_edge("a", "b", weight=2)
        G.add_edge("a", "b", weight=3)
        result = nx.team_projection(G, sum_weights)
        edge = result.graph["X"]["Y"][0]
        assert edge["weight"] == 6
        assert [(u, v, k) for u, v, k, _ in edge["sources"]] == [
            ("a", "b", 0),
            ("a", "b", 1),
            ("a", "b", 2),
        ]
        assert [a["weight"] for _, _, _, a in edge["sources"]] == [1, 2, 3]

    def test_stable_ordering(self):
        G = nx.MultiDiGraph()
        for node, team in [("n0", "B"), ("n1", "A"), ("n2", "C"), ("n3", "A")]:
            G.add_node(node, team=team)
        edges = [
            ("n1", "n0", 1),  # A -> B
            ("n1", "n2", 2),  # A -> C
            ("n3", "n0", 3),  # A -> B (second parallel source)
            ("n2", "n0", 4),  # C -> B
        ]
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        result = nx.team_projection(G, sum_weights)
        # Group order follows first occurrence: A->B, A->C, C->B.
        assert list(result.graph.edges(keys=True)) == [
            ("A", "B", 0),
            ("A", "C", 0),
            ("C", "B", 0),
        ]
        sources = result.graph["A"]["B"][0]["sources"]
        assert [(u, v, k) for u, v, k, _ in sources] == [("n1", "n0", 0), ("n3", "n0", 0)]
        assert [a["weight"] for _, _, _, a in sources] == [1, 3]

    def test_aggregator_error_leaves_graph_untouched(self):
        G = nx.MultiDiGraph()
        G.add_node("a", team="X")
        G.add_node("b", team="Y")
        G.add_edge("a", "b", weight=2, meta={"tags": ["p", "q"]})
        G.add_edge("a", "b", weight=3, meta={"tags": ["r"]})
        before = nx.to_dict_of_dicts(G)  # structure snapshot
        attrs_before = [dict(d, meta={"tags": list(d["meta"]["tags"])})
                        for _, _, _, d in G.edges(keys=True, data=True)]

        def bad_aggregator(attrs):
            # Mutate everything it can reach, then blow up.
            for a in attrs:
                a["weight"] = 999
                a["meta"]["tags"].append("CORRUPTED")
                a["meta"]["evil"] = True
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            nx.team_projection(G, bad_aggregator)

        # Structure unchanged.
        assert nx.to_dict_of_dicts(G) == before
        # Attributes, including nested ones, unchanged.
        attrs_after = [dict(d, meta={"tags": list(d["meta"]["tags"])})
                       for _, _, _, d in G.edges(keys=True, data=True)]
        assert attrs_after == attrs_before
        for _, _, _, d in G.edges(keys=True, data=True):
            assert "evil" not in d["meta"]
            assert "CORRUPTED" not in d["meta"]["tags"]

    def test_result_snapshots_independent_of_input(self):
        G = make_xy_graph()
        result = nx.team_projection(G, sum_weights)
        # Mutating the input afterwards must not corrupt the result.
        G["x1"]["y1"][0]["weight"] = -100
        assert result.graph["X"]["Y"][0]["sources"][0][3]["weight"] == 2

    def test_custom_team_attribute(self):
        G = nx.MultiDiGraph()
        G.add_node("a", group="X")
        G.add_node("b", group="Y")
        G.add_edge("a", "b", weight=7)
        result = nx.team_projection(G, sum_weights, team_attribute="group")
        assert result.graph["X"]["Y"][0]["weight"] == 7

    def test_requires_directed_multigraph(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.team_projection(nx.Graph(), sum_weights)
        with pytest.raises(nx.NetworkXError):
            nx.team_projection(nx.DiGraph(), sum_weights)
