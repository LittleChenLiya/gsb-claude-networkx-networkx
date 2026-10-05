"""Tests for the independent shortest-path certificate checker."""

import pytest

import networkx as nx
from networkx.algorithms.shortest_paths.certificate import (
    CertificateError,
    NegativeEdgeError,
    Violation,
    is_valid_shortest_path_certificate,
    verify_shortest_path_certificate,
)


def three_node_path():
    """The three-node graph shared by the acceptance scenarios: 0-1-2."""
    G = nx.path_graph(3)
    dist = {0: 0, 1: 1, 2: 2}
    pred = {1: 0, 2: 1}
    return G, dist, pred


# ---------------------------------------------------------------------------
# Acceptance scenarios
# ---------------------------------------------------------------------------


def test_three_node_certificate_passes():
    G, dist, pred = three_node_path()
    assert verify_shortest_path_certificate(G, 0, dist, pred) is None
    assert is_valid_shortest_path_certificate(G, 0, dist, pred) is True


@pytest.mark.parametrize(
    "dist",
    [
        {0: 0, 1: 1, 2: 0},  # terminal distance too small
        {0: 0, 1: 1, 2: 9},  # terminal distance too large
    ],
)
def test_bad_terminal_distance_rejected_by_edge_check(dist):
    G, _, pred = three_node_path()
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, pred)
    kinds = {v.kind for v in exc_info.value.violations}
    # Too small -> the predecessor edge is not tight; too large -> the edge
    # constraint is violated.  Either way the reported defect names edge (1,2).
    assert kinds & {"loose_predecessor", "edge_constraint"}
    assert any(v.edge in ((1, 2), (2, 1)) for v in exc_info.value.violations)
    assert is_valid_shortest_path_certificate(G, 0, dist, pred) is False


def test_shrunk_terminal_distance_reports_specific_edge():
    G, _, pred = three_node_path()
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, {0: 0, 1: 1, 2: 0}, pred)
    (violation,) = exc_info.value.violations
    assert violation.kind == "loose_predecessor"
    assert violation.edge == (1, 2)
    assert "dist[2]=0" in str(violation)
    assert "dist[1]+1=2" in str(violation)


def test_distance_constraint_violation_names_edge():
    # A claimed distance that is too large breaks dist[v] <= dist[u] + w.
    G, _, pred = three_node_path()
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, {0: 0, 1: 1, 2: 9}, pred)
    kinds = {v.kind for v in exc_info.value.violations}
    assert "edge_constraint" in kinds
    edge_violation = next(
        v for v in exc_info.value.violations if v.kind == "edge_constraint"
    )
    assert edge_violation.edge == (1, 2)
    assert "dist[2]=9 > dist[1]+1=2" in str(edge_violation)


def test_predecessor_self_loop_gets_chain_diagnosis():
    G, dist, _ = three_node_path()
    pred = {1: 0, 2: 2}  # node 2 claims itself as predecessor
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, pred)
    cycle_violations = [
        v for v in exc_info.value.violations if v.kind == "predecessor_cycle"
    ]
    assert len(cycle_violations) == 1
    cycle = cycle_violations[0]
    assert cycle.chain == (2, 2)
    assert "2 -> 2" in str(cycle)


def test_predecessor_cycle_chain_diagnosis():
    G, dist, _ = three_node_path()
    pred = {1: 2, 2: 1}  # 1 <-> 2 cycle detached from the source
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, pred)
    cycle_violations = [
        v for v in exc_info.value.violations if v.kind == "predecessor_cycle"
    ]
    assert len(cycle_violations) == 1  # the cycle is reported exactly once
    assert cycle_violations[0].chain in ((1, 2, 1), (2, 1, 2))


def test_three_node_cycle_with_selfloop_chain_reaches_source():
    # A self-loop on the source is a chain that never leaves the source.
    G = nx.cycle_graph(3)  # edges 0-1, 1-2, 2-0, unit weights
    dist = {0: 0, 1: 1, 2: 1}
    pred = {1: 0, 2: 0}
    verify_shortest_path_certificate(G, 0, dist, pred)


# ---------------------------------------------------------------------------
# Predecessor edge existence and tightness
# ---------------------------------------------------------------------------


def test_nonexistent_predecessor_edge_rejected():
    G, dist, _ = three_node_path()
    pred = {1: 0, 2: 0}  # edge (0, 2) does not exist
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, pred)
    kinds = {v.kind for v in exc_info.value.violations}
    assert "nonexistent_edge" in kinds
    bad = next(v for v in exc_info.value.violations if v.kind == "nonexistent_edge")
    assert bad.edge == (0, 2)
    assert bad.node == 2


def test_loose_predecessor_edge_rejected():
    # A real, existing predecessor edge whose weight does not match the
    # claimed distance is not tight.
    G = nx.Graph()
    G.add_edge(0, 1, weight=2)
    G.add_edge(1, 2, weight=4)
    G.add_edge(0, 2, weight=6)
    dist = {0: 0, 1: 2, 2: 6}
    # Both the direct edge (weight 6) and the via-1 route (2 + 4 = 6) are
    # tight predecessor choices for node 2.
    verify_shortest_path_certificate(G, 0, dist, {1: 0, 2: 0})
    verify_shortest_path_certificate(G, 0, dist, {1: 0, 2: 1})
    # Claiming the weight-4 edge with dist[2] = 5 fails tightness.
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, {0: 0, 1: 2, 2: 5}, {1: 0, 2: 1})
    (violation,) = exc_info.value.violations
    assert violation.kind == "loose_predecessor"
    assert violation.edge == (1, 2)


def test_directed_reverse_arc_predecessor_rejected():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=2)
    G.add_edge(1, 2, weight=3)
    dist = {0: 0, 1: 2, 2: 5}
    pred = {1: 2, 2: 1}  # arc 2 -> 1 does not exist
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, pred)
    kinds = {v.kind for v in exc_info.value.violations}
    assert "nonexistent_edge" in kinds


def test_missing_predecessor_for_reachable_node_rejected():
    G, dist, _ = three_node_path()
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, {1: 0})
    kinds = {v.kind for v in exc_info.value.violations}
    assert "missing_predecessor" in kinds
    assert any(v.node == 2 for v in exc_info.value.violations)


def test_predecessor_on_source_rejected():
    G, dist, _ = three_node_path()
    pred = {0: 1, 1: 0, 2: 1}
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, pred)
    assert any(
        v.kind == "loose_predecessor" and v.node == 0
        for v in exc_info.value.violations
    )


# ---------------------------------------------------------------------------
# Parallel edges: edge identity must be explicit
# ---------------------------------------------------------------------------


def _parallel_graph():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, key="a", weight=5)
    G.add_edge(0, 1, key="b", weight=1)
    G.add_edge(1, 2, key="c", weight=2)
    return G


def test_parallel_edges_require_edge_identity():
    G = _parallel_graph()
    dist = {0: 0, 1: 1, 2: 3}
    # Bare predecessor node is ambiguous between keys "a" and "b".
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, {1: 0, 2: 1})
    (violation,) = exc_info.value.violations
    assert violation.kind == "ambiguous_parallel_edge"
    assert violation.node == 1
    assert "'a'" in str(violation) and "'b'" in str(violation)


def test_parallel_edges_explicit_key_selects_identity():
    G = _parallel_graph()
    dist = {0: 0, 1: 1, 2: 3}
    # Key "b" is the cheap parallel edge.
    verify_shortest_path_certificate(G, 0, dist, {1: (0, "b"), 2: 1})


def test_parallel_edges_wrong_key_tightness_rejected():
    G = _parallel_graph()
    # Picking key "a" (weight 5) as predecessor with dist[1]=1 is not tight;
    # additionally the cheap key "b" forces the edge constraint dist[1] <= 1.
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(
            G, 0, {0: 0, 1: 5, 2: 7}, {1: (0, "a"), 2: 1}
        )
    kinds = {v.kind for v in exc_info.value.violations}
    assert "edge_constraint" in kinds
    edge_violation = next(
        v
        for v in exc_info.value.violations
        if v.kind == "edge_constraint" and v.edge == (0, 1, "b")
    )
    assert edge_violation is not None


def test_parallel_edges_unknown_key_rejected():
    G = _parallel_graph()
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(
            G, 0, {0: 0, 1: 1, 2: 3}, {1: (0, "zzz"), 2: 1}
        )
    assert any(v.kind == "nonexistent_edge" for v in exc_info.value.violations)
    assert any(v.edge == (0, 1, "zzz") for v in exc_info.value.violations)


def test_single_parallel_bare_node_allowed():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key="only", weight=3)
    dist = {0: 0, 1: 3}
    verify_shortest_path_certificate(G, 0, dist, {1: 0})


# ---------------------------------------------------------------------------
# Unreachable nodes, sources, malformed distance tables
# ---------------------------------------------------------------------------


def test_unreachable_nodes_omitted_accepted():
    G = nx.path_graph(3)
    G.add_edge(8, 9)  # separate component
    dist = {0: 0, 1: 1, 2: 2}
    pred = {1: 0, 2: 1}
    verify_shortest_path_certificate(G, 0, dist, pred)


def test_unreachable_node_with_zero_distance_rejected():
    # The reachable component and the isolated node 9: claiming dist[9] = 0
    # must not silently treat an unreachable node as if it were the source.
    G = nx.path_graph(3)
    G.add_edge(8, 9)
    dist = {0: 0, 1: 1, 2: 2, 9: 0}
    pred = {1: 0, 2: 1}
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, pred)
    assert any(v.node == 9 for v in exc_info.value.violations)


def test_unreachable_node_with_predecessor_rejected():
    G = nx.path_graph(3)
    G.add_edge(8, 9)
    dist = {0: 0, 1: 1, 2: 2}
    pred = {1: 0, 2: 1, 8: 9}  # 8 absent from dist
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, pred)
    violation = next(
        v for v in exc_info.value.violations if v.kind == "unreachable_has_predecessor"
    )
    assert violation.node == 8


def test_source_must_claim_distance_zero():
    G, _, pred = three_node_path()
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, {1: 1, 2: 2}, pred)
    assert any(
        v.node == 0 and v.kind == "invalid_distance"
        for v in exc_info.value.violations
    )
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, {0: 3, 1: 1, 2: 2}, pred)
    assert any(
        v.node == 0 and v.kind == "invalid_distance"
        for v in exc_info.value.violations
    )


def test_multi_source_certificate_passes():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (3, 2)])
    dist = {0: 0, 3: 0, 1: 1, 2: 1}
    pred = {1: 0, 2: 3}
    verify_shortest_path_certificate(G, [0, 3], dist, pred)


def test_unknown_nodes_in_tables_rejected():
    G, dist, pred = three_node_path()
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, {**dist, 7: 1}, pred)
    assert any(
        v.kind == "unknown_node" and v.node == 7
        for v in exc_info.value.violations
    )
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, {**pred, 7: 0})
    assert any(
        v.kind == "unknown_node" and v.node == 7
        for v in exc_info.value.violations
    )


def test_invalid_distance_values_rejected():
    G, _, pred = three_node_path()
    for bad in (-1, float("inf"), float("nan"), "x", True):
        with pytest.raises(CertificateError) as exc_info:
            verify_shortest_path_certificate(G, 0, {0: 0, 1: 1, 2: bad}, pred)
        assert any(
            v.kind in {"invalid_distance", "unreachable_distance"}
            for v in exc_info.value.violations
        )


def test_empty_and_unknown_sources_rejected():
    G, _, _ = three_node_path()
    with pytest.raises(nx.NetworkXError):
        verify_shortest_path_certificate(G, [], {0: 0})
    with pytest.raises(nx.NetworkXError):
        verify_shortest_path_certificate(G, 99, {99: 0})


# ---------------------------------------------------------------------------
# Negative weights: explicitly unsupported
# ---------------------------------------------------------------------------


def test_negative_edge_weight_raises():
    G = nx.Graph()
    G.add_edge(0, 1, weight=-1)
    with pytest.raises(NegativeEdgeError) as exc_info:
        verify_shortest_path_certificate(G, 0, {0: 0})
    assert isinstance(exc_info.value, nx.NetworkXError)
    assert isinstance(exc_info.value, ValueError)
    assert "(0, 1)" in str(exc_info.value)


def test_negative_parallel_edge_weight_raises_with_identity():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, key="ok", weight=2)
    G.add_edge(0, 1, key="bad", weight=-4)
    with pytest.raises(NegativeEdgeError) as exc_info:
        verify_shortest_path_certificate(
            G, 0, {0: 0, 1: 2}, {1: (0, "ok")}
        )
    assert "'bad'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Graph is never mutated; dist-only mode; weighted / callable weights
# ---------------------------------------------------------------------------


def test_graph_is_not_modified():
    G, dist, pred = three_node_path()
    edges_before = sorted(G.edges())
    nodes_before = sorted(G.nodes)
    verify_shortest_path_certificate(G, 0, dist, pred)
    assert sorted(G.edges()) == edges_before
    assert sorted(G.nodes) == nodes_before
    # Failure path must not mutate either.
    with pytest.raises(CertificateError):
        verify_shortest_path_certificate(G, 0, {0: 0, 1: 1, 2: 0}, pred)
    assert sorted(G.edges()) == edges_before
    assert sorted(G.nodes) == nodes_before


def test_dist_only_mode_checks_constraints():
    G, dist, _ = three_node_path()
    verify_shortest_path_certificate(G, 0, dist)
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, {0: 0, 1: 1, 2: 99})
    assert any(v.kind == "edge_constraint" for v in exc_info.value.violations)


def test_weighted_graph_certificate():
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.5)
    G.add_edge(1, 2, weight=0.5)
    dist = {0: 0, 1: 2.5, 2: 3.0}
    pred = {1: 0, 2: 1}
    verify_shortest_path_certificate(G, 0, dist, pred)


def test_callable_weight_certificate():
    G = nx.path_graph(3)
    dist = {0: 0, 1: 2, 2: 4}
    pred = {1: 0, 2: 1}
    verify_shortest_path_certificate(G, 0, dist, pred, weight=lambda u, v, d: 2)


def test_zero_weight_edges():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=0)
    G.add_edge(1, 2, weight=0)
    dist = {0: 0, 1: 0, 2: 0}
    pred = {1: 0, 2: 1}
    verify_shortest_path_certificate(G, 0, dist, pred)


def test_violations_are_all_reported_together():
    G, _, _ = three_node_path()
    # Two independent defects at once: bad distance and a self-loop chain.
    dist = {0: 0, 1: 1, 2: 0}
    pred = {1: 0, 2: 2}
    with pytest.raises(CertificateError) as exc_info:
        verify_shortest_path_certificate(G, 0, dist, pred)
    kinds = {v.kind for v in exc_info.value.violations}
    assert "predecessor_cycle" in kinds
    assert all(isinstance(v, Violation) for v in exc_info.value.violations)

