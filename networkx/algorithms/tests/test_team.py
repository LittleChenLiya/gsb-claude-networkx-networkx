"""Unit tests for team projection of multigraphs."""

import copy

import pytest

import networkx as nx


def _build_basic():
    """Two X nodes connect to Y with weights 2 and 3; X has an internal edge."""
    G = nx.MultiDiGraph()
    G.add_node("x1", team="X")
    G.add_node("x2", team="X")
    G.add_node("y1", team="Y")
    G.add_edge("x1", "y1", key=0, weight=2)
    G.add_edge("x2", "y1", key=0, weight=3)
    G.add_edge("x1", "x2", key=0, weight=10)  # internal edge
    return G


def test_basic_aggregation_and_sources():
    G = _build_basic()
    H, internal = nx.team_projection(G)
    # Weights 2 and 3 aggregate to 5.
    assert H["X"]["Y"]["weight"] == 5
    # Both source edges are recorded, in stable (u, v, key) order.
    assert H["X"]["Y"]["sources"] == [("x1", "y1", 0), ("x2", "y1", 0)]
    # The internal edge is listed separately and not part of the total.
    assert len(internal) == 1
    assert internal[0][0] == "x1"
    assert internal[0][1] == "x2"
    assert internal[0][2] == 0
    assert internal[0][3]["weight"] == 10
    # The internal edge did not contribute to the external total.
    assert H["X"]["Y"]["weight"] != 2 + 3 + 10


def test_internal_edges_not_aggregated():
    G = _build_basic()
    H, internal = nx.team_projection(G)
    # No self-loop on X in the projected graph (internal edges stay out).
    assert not H.has_edge("X", "X")
    assert not H.has_edge("Y", "Y")
    # The internal edge is returned as a (u, v, key, data) tuple.
    u, v, key, data = internal[0]
    assert (u, v, key) == ("x1", "x2", 0)


def test_missing_team_drops_edge():
    G = nx.MultiDiGraph()
    G.add_node("x1", team="X")
    G.add_node("y1", team="Y")
    G.add_node("z1")  # no team attribute
    G.add_edge("x1", "y1", key=0, weight=2)
    G.add_edge("x1", "z1", key=0, weight=9)  # edge to unmapped node
    H, internal = nx.team_projection(G)
    assert H.has_edge("X", "Y")
    assert not H.has_node("z1")
    assert H["X"]["Y"]["weight"] == 2
    assert internal == []


def test_missing_team_value_none_drops_edge():
    G = nx.MultiDiGraph()
    G.add_node("x1", team="X")
    G.add_node("y1", team=None)  # explicitly None
    G.add_edge("x1", "y1", key=0, weight=2)
    H, internal = nx.team_projection(G)
    assert H.number_of_edges() == 0
    assert internal == []


def test_parallel_edge_keys_preserved():
    G = nx.MultiDiGraph()
    G.add_node("x1", team="X")
    G.add_node("y1", team="Y")
    G.add_edge("x1", "y1", key="k1", weight=1)
    G.add_edge("x1", "y1", key="k2", weight=2)
    G.add_edge("x1", "y1", key=5, weight=3)
    H, _ = nx.team_projection(G)
    sources = H["X"]["Y"]["sources"]
    # All three parallel edges are kept, with their original keys.
    assert sources == [("x1", "y1", 5), ("x1", "y1", "k1"), ("x1", "y1", "k2")]
    assert H["X"]["Y"]["weight"] == 1 + 2 + 3


def test_stable_order_regardless_of_insertion():
    G = nx.MultiDiGraph()
    G.add_node("z1", team="Z")
    G.add_node("a1", team="A")
    G.add_edge("z1", "a1", key=2, weight=1)
    G.add_edge("z1", "a1", key=0, weight=1)
    G.add_edge("a1", "z1", key=1, weight=1)
    captured = []

    def recording_aggregator(records):
        captured.append([(u, v, key) for u, v, key, _ in records])
        return {"n": len(records)}

    H, _ = nx.team_projection(G, aggregator=recording_aggregator)
    # Records arrive sorted by (u, v, key), independent of insertion order.
    assert captured[0] == [("a1", "z1", 1)]
    assert captured[1] == [("z1", "a1", 0), ("z1", "a1", 2)]
    # Internal edges are also returned in stable order.
    G.add_edge("z1", "z1", key=1, weight=1)
    G.add_edge("z1", "z1", key=0, weight=1)
    _, internal = nx.team_projection(G, aggregator=recording_aggregator)
    assert [(u, v, key) for u, v, key, _ in internal] == [
        ("z1", "z1", 0),
        ("z1", "z1", 1),
    ]


def test_aggregator_error_skips_group():
    G = nx.MultiDiGraph()
    G.add_node("x1", team="X")
    G.add_node("x2", team="X")
    G.add_node("y1", team="Y")
    G.add_node("z1", team="Z")
    G.add_edge("x1", "y1", key=0, weight=2)
    G.add_edge("x2", "y1", key=0, weight=3)
    G.add_edge("x1", "z1", key=0, weight=4)

    def failing_aggregator(records):
        if any(v == "y1" for _, v, _, _ in records):
            raise RuntimeError("cannot aggregate Y")
        return {"weight": sum(d.get("weight", 1) for _, _, _, d in records)}

    H, _ = nx.team_projection(G, aggregator=failing_aggregator)
    # The failing group produces no edge; the other group is unaffected.
    assert not H.has_edge("X", "Y")
    assert H.has_edge("X", "Z")
    assert H["X"]["Z"]["weight"] == 4


def test_aggregator_cannot_modify_input_graph():
    G = nx.MultiDiGraph()
    G.add_node("x1", team="X")
    G.add_node("x2", team="X")
    G.add_node("y1", team="Y")
    G.add_edge("x1", "y1", key=0, weight=2, meta={"nested": [1, 2]})
    G.add_edge("x2", "y1", key=0, weight=3, meta={"nested": [3, 4]})
    G.add_edge("x1", "x2", key=0, weight=10, meta={"nested": [5]})

    snapshot = copy.deepcopy(G)

    def mutating_aggregator(records):
        for _, _, _, data in records:
            data["weight"] = 999
            data["meta"]["nested"].append(999)
        raise RuntimeError("boom after mutation")

    H, internal = nx.team_projection(G, aggregator=mutating_aggregator)
    # The failing group produces no edge.
    assert H.number_of_edges() == 0
    # The original graph and all of its (nested) attributes are unchanged.
    assert G.nodes(data=True) == snapshot.nodes(data=True)
    for u, v, key, data in G.edges(keys=True, data=True):
        assert data == snapshot[u][v][key]
        assert data["meta"]["nested"] == snapshot[u][v][key]["meta"]["nested"]
    # The internal-edge snapshot is also a copy, not the original.
    assert internal[0][3]["meta"]["nested"] == [5]


def test_source_graph_unchanged_after_projection():
    G = _build_basic()
    snapshot = copy.deepcopy(G)
    nx.team_projection(G)
    assert G.nodes(data=True) == snapshot.nodes(data=True)
    assert list(G.edges(keys=True, data=True)) == list(
        snapshot.edges(keys=True, data=True)
    )


def test_undirected_multigraph():
    G = nx.MultiGraph()
    G.add_node("x1", team="X")
    G.add_node("y1", team="Y")
    G.add_edge("x1", "y1", key=0, weight=2)
    G.add_edge("y1", "x1", key=1, weight=3)
    H, _ = nx.team_projection(G)
    assert isinstance(H, nx.Graph)
    assert H["X"]["Y"]["weight"] == 5
    # Undirected edges are canonicalized by NetworkX to (lower, higher).
    assert H["X"]["Y"]["sources"] == [("x1", "y1", 0), ("x1", "y1", 1)]


def test_custom_team_attribute():
    G = nx.MultiDiGraph()
    G.add_node("x1", group="X")
    G.add_node("y1", group="Y")
    G.add_edge("x1", "y1", key=0, weight=2)
    H, _ = nx.team_projection(G, team_attr="group")
    assert H.has_node("X")
    assert H["X"]["Y"]["weight"] == 2
