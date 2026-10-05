"""Local tests for the independent shortest-path certificate checker.

Run from this directory with::

    python -m pytest test_certificate_checker.py -q -p no:asyncio -p no:aiohttp

(The extra ``-p`` flags disable unrelated third-party pytest plugins that
are installed in this environment and crash during startup.)

These tests use tiny deterministic fixtures and do not re-run any
shortest-path algorithm: the checker under test is the verifier itself.
"""

import math

import networkx as nx
import pytest

from certificate_checker import (
    CertificateError,
    check_shortest_path_certificate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_node_digraph():
    """s -> a (1), s -> b (100), a -> b (1): true distances 0, 1, 2."""
    G = nx.DiGraph()
    G.add_weighted_edges_from([("s", "a", 1), ("s", "b", 100), ("a", "b", 1)])
    return G


@pytest.fixture
def good_certificate(three_node_digraph):
    return {
        "sources": ["s"],
        "dist": {"s": 0, "a": 1, "b": 2},
        "pred": {"s": None, "a": "s", "b": "a"},
    }


# ---------------------------------------------------------------------------
# Acceptance: a correct 3-node certificate passes
# ---------------------------------------------------------------------------

def test_good_three_node_certificate_passes(three_node_digraph, good_certificate):
    assert check_shortest_path_certificate(
        three_node_digraph,
        good_certificate["sources"],
        good_certificate["dist"],
        good_certificate["pred"],
    ) is True


def test_undirected_certificate_passes():
    G = nx.Graph()
    G.add_weighted_edges_from([("a", "b", 1), ("b", "c", 2)])
    assert check_shortest_path_certificate(
        G, ["a"], {"a": 0, "b": 1, "c": 3}, {"a": None, "b": "a", "c": "b"}
    ) is True


def test_multi_source_certificate_passes():
    G = nx.DiGraph()
    G.add_weighted_edges_from([("s1", "a", 1), ("s2", "a", 2), ("a", "b", 1)])
    assert check_shortest_path_certificate(
        G,
        ["s1", "s2"],
        {"s1": 0, "s2": 0, "a": 1, "b": 2},
        {"s1": None, "s2": None, "a": "s1", "b": "a"},
    ) is True


def test_unreachable_nodes_may_be_inf_or_none():
    G = nx.DiGraph()
    G.add_weighted_edges_from([("s", "a", 1), ("x", "y", 1)])
    assert check_shortest_path_certificate(
        G,
        ["s"],
        {"s": 0, "a": 1, "x": math.inf, "y": None},
        {"s": None, "a": "s", "x": None, "y": None},
    ) is True


def test_unit_weights_when_weight_is_none():
    G = nx.DiGraph()
    G.add_edges_from([("s", "a"), ("a", "b")])
    assert check_shortest_path_certificate(
        G,
        ["s"],
        {"s": 0, "a": 1, "b": 2},
        {"s": None, "a": "s", "b": "a"},
        weight=None,
    ) is True

# ---------------------------------------------------------------------------
# Acceptance: making the end distance smaller is rejected by an edge
# constraint (the predecessor edge is no longer tight)
# ---------------------------------------------------------------------------

def test_end_distance_too_small_rejected_by_edge_constraint(
    three_node_digraph, good_certificate
):
    bad_dist = dict(good_certificate["dist"])
    bad_dist["b"] = 1  # true distance is 2; predecessor edge a->b has weight 1
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            three_node_digraph,
            good_certificate["sources"],
            bad_dist,
            good_certificate["pred"],
        )
    message = str(exc_info.value)
    # the report names the specific edge and the violated constraint
    assert "('a', 'b')" in message
    assert "not tight" in message


def test_end_distance_too_large_rejected_by_edge_constraint(
    three_node_digraph, good_certificate
):
    bad_dist = dict(good_certificate["dist"])
    bad_dist["b"] = 3  # certificate is not minimal: edge a->b gives 2
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            three_node_digraph,
            good_certificate["sources"],
            bad_dist,
            good_certificate["pred"],
        )
    message = str(exc_info.value)
    assert "('a', 'b')" in message
    assert "relaxation" in message


# ---------------------------------------------------------------------------
# Acceptance: predecessor self-loop gives a chain diagnosis
# ---------------------------------------------------------------------------

def test_predecessor_self_loop_gives_chain_diagnosis():
    G = nx.DiGraph()
    G.add_weighted_edges_from([("s", "a", 1), ("a", "a", 0)])
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            G, ["s"], {"s": 0, "a": 1}, {"s": None, "a": "a"}
        )
    message = str(exc_info.value)
    assert "chain" in message
    assert "self-loop" in message
    assert "'a' -> 'a'" in message


def test_predecessor_cycle_gives_chain_diagnosis():
    G = nx.DiGraph()
    G.add_weighted_edges_from([("s", "a", 1), ("a", "b", 0), ("b", "a", 0)])
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            G,
            ["s"],
            {"s": 0, "a": 1, "b": 1},
            {"s": None, "a": "b", "b": "a"},
        )
    message = str(exc_info.value)
    assert "chain" in message
    assert "cycle" in message
    assert "'a'" in message and "'b'" in message


# ---------------------------------------------------------------------------
# Unreachable nodes must not be mixed up as distance 0
# ---------------------------------------------------------------------------

def test_unreachable_node_with_finite_distance_is_rejected():
    G = nx.DiGraph()
    G.add_weighted_edges_from([("s", "a", 1), ("x", "y", 1)])
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            G,
            ["s"],
            {"s": 0, "a": 1, "x": 0, "y": 0},
            {"s": None, "a": "s", "x": None, "y": None},
        )
    message = str(exc_info.value)
    assert "'x'" in message
    assert "unreachable" in message


def test_reachable_node_without_distance_is_rejected():
    G = nx.DiGraph()
    G.add_weighted_edges_from([("s", "a", 1)])
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            G, ["s"], {"s": 0, "a": None}, {"s": None, "a": "s"}
        )
    assert "reachable" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Negative weights are explicitly unsupported
# ---------------------------------------------------------------------------

def test_negative_weight_is_rejected():
    G = nx.DiGraph()
    G.add_weighted_edges_from([("s", "a", 1), ("a", "b", -5)])
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            G,
            ["s"],
            {"s": 0, "a": 1, "b": -4},
            {"s": None, "a": "s", "b": "a"},
        )
    message = str(exc_info.value)
    assert "('a', 'b')" in message
    assert "negative" in message

# ---------------------------------------------------------------------------
# Parallel edges: edge identity must be explicit
# ---------------------------------------------------------------------------

@pytest.fixture
def parallel_digraph():
    G = nx.MultiDiGraph()
    G.add_edge("s", "a", key=0, weight=1)
    G.add_edge("s", "a", key=1, weight=5)
    return G


def test_parallel_edge_correct_key_passes(parallel_digraph):
    assert check_shortest_path_certificate(
        parallel_digraph,
        ["s"],
        {"s": 0, "a": 1},
        {"s": None, "a": "s"},
        edge_keys={"a": 0},
    ) is True


def test_parallel_edge_missing_key_is_rejected(parallel_digraph):
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            parallel_digraph,
            ["s"],
            {"s": 0, "a": 1},
            {"s": None, "a": "s"},
        )
    message = str(exc_info.value)
    assert "edge_keys" in message
    assert "multigraph" in message


def test_parallel_edge_wrong_key_is_rejected(parallel_digraph):
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            parallel_digraph,
            ["s"],
            {"s": 0, "a": 1},
            {"s": None, "a": "s"},
            edge_keys={"a": 1},
        )
    message = str(exc_info.value)
    assert "key=1" in message
    assert "not tight" in message


def test_parallel_edge_nonexistent_key_is_rejected(parallel_digraph):
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            parallel_digraph,
            ["s"],
            {"s": 0, "a": 1},
            {"s": None, "a": "s"},
            edge_keys={"a": 99},
        )
    assert "does not exist" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Predecessor validity
# ---------------------------------------------------------------------------

def test_predecessor_unknown_node_is_rejected():
    G = nx.DiGraph()
    G.add_edge("s", "a")
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            G, ["s"], {"s": 0, "a": 1}, {"s": None, "a": "z"}
        )
    assert "'z'" in str(exc_info.value)


def test_predecessor_edge_missing_is_rejected():
    G = nx.DiGraph()
    G.add_edge("s", "a")
    G.add_edge("z", "a")  # z unreachable; predecessor edge must be real
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            G, ["s"], {"s": 0, "a": 1}, {"s": None, "a": "z"}
        )
    message = str(exc_info.value)
    assert "reachable" in message


def test_source_must_have_distance_zero_and_no_pred():
    G = nx.DiGraph()
    G.add_edge("s", "a")
    with pytest.raises(CertificateError) as exc_info:
        check_shortest_path_certificate(
            G, ["s"], {"s": 1, "a": 2}, {"s": None, "a": "s"}
        )
    assert "source" in str(exc_info.value)


# ---------------------------------------------------------------------------
# The checker must not modify the input graph
# ---------------------------------------------------------------------------

def test_graph_is_not_mutated(three_node_digraph, good_certificate):
    nodes_before = list(three_node_digraph.nodes)
    edges_before = list(three_node_digraph.edges(data=True))
    check_shortest_path_certificate(
        three_node_digraph,
        good_certificate["sources"],
        good_certificate["dist"],
        good_certificate["pred"],
    )
    assert list(three_node_digraph.nodes) == nodes_before
    assert list(three_node_digraph.edges(data=True)) == edges_before
