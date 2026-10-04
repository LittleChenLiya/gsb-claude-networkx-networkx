"""Tests for the DAG scheduling :class:`CompletionFrontier`."""

import json

import pytest

import networkx as nx
from networkx.algorithms.completion_frontier import (
    CheckpointMismatchError,
    CompletionFrontier,
    DuplicateNodeIdentityError,
    GraphStructureError,
    IllegalCompletionError,
)

EDGES = [("a", "c"), ("b", "c"), ("c", "d")]
NODES = ["a", "b", "c", "d"]


def make_frontier():
    return CompletionFrontier(NODES, EDGES)


def make_graph():
    G = nx.DiGraph()
    G.add_nodes_from(NODES)
    G.add_edges_from(EDGES)
    return G


def test_acceptance_release_order():
    # a->c, b->c, c->d: initially only a and b are executable.
    frontier = make_frontier()
    assert frontier.ready_nodes() == ["a", "b"]
    assert frontier.completed_nodes() == []

    # Completing a alone does not release c (b is still pending).
    assert frontier.complete("a") == []
    assert frontier.ready_nodes() == ["b"]
    assert not frontier.is_ready("c")
    with pytest.raises(IllegalCompletionError):
        frontier.complete("c")  # illegal early completion

    # Completing b releases c; completing c then releases d.
    assert frontier.complete("b") == ["c"]
    assert frontier.ready_nodes() == ["c"]
    assert frontier.complete("c") == ["d"]
    assert frontier.complete("d") == []
    assert frontier.is_finished()
    assert frontier.ready_nodes() == []


def test_illegal_early_completion_unknown_and_repeat():
    frontier = make_frontier()

    # Unknown node.
    with pytest.raises(IllegalCompletionError):
        frontier.complete("nope")

    # c needs both a and b; neither is done yet.
    with pytest.raises(IllegalCompletionError):
        frontier.complete("c")

    frontier.complete("a")
    with pytest.raises(IllegalCompletionError):
        frontier.complete("d")  # d is not even close


def test_duplicate_confirmation_is_idempotent():
    frontier = make_frontier()
    frontier.complete("a")
    first = frontier.checkpoint()

    assert frontier.complete("a") == []  # repeat releases nothing
    assert frontier.is_completed("a")
    assert frontier.ready_nodes() == ["b"]
    assert frontier.checkpoint() == first  # state unchanged
    assert frontier.complete("b") == ["c"]  # and c is still released once


def test_from_graph_matches_direct_construction():
    g_frontier = CompletionFrontier.from_graph(make_graph())
    assert g_frontier.nodes == NODES
    assert g_frontier.ready_nodes() == ["a", "b"]

    direct = make_frontier()
    assert g_frontier.fingerprint == direct.fingerprint


def test_snapshot_does_not_share_mutable_edges():
    G = make_graph()
    frontier = CompletionFrontier.from_graph(G)

    # Mutating the caller's graph after construction must not touch the
    # frontier (no shared mutable edges/nodes).
    G.add_edge("a", "d")
    G.add_node("e")
    G.remove_edge("b", "c")
    assert frontier.nodes == NODES
    assert frontier.ready_nodes() == ["a", "b"]
    assert frontier.fingerprint == make_frontier().fingerprint

    # Returned lists are independent copies too.
    ready = frontier.ready_nodes()
    ready.append("c")
    assert frontier.ready_nodes() == ["a", "b"]


def test_init_rejects_cycles_and_duplicate_identities():
    with pytest.raises(GraphStructureError):
        CompletionFrontier([1, 2], [(1, 2), (2, 1)])

    with pytest.raises(GraphStructureError):
        CompletionFrontier.from_graph(nx.cycle_graph(3, create_using=nx.DiGraph))

    with pytest.raises(DuplicateNodeIdentityError):
        CompletionFrontier(["a", "b", "a"], [])

    with pytest.raises(GraphStructureError):
        CompletionFrontier.from_graph(nx.Graph([(1, 2)]))  # undirected

    with pytest.raises(ValueError):
        CompletionFrontier(["a", "b"], [("a", "z")])  # unknown endpoint


def test_checkpoint_roundtrip_is_same_frontier():
    frontier = make_frontier()
    frontier.complete("a")
    frontier.complete("b")
    checkpoint = frontier.checkpoint()

    # Plain, JSON-serializable data.
    assert json.loads(json.dumps(checkpoint)) == checkpoint
    assert checkpoint["completed"] == [NODES.index("a"), NODES.index("b")]
    assert len(checkpoint["fingerprint"]["hexdigest"]) == 64

    restored = CompletionFrontier.restore(checkpoint, NODES, EDGES)
    assert restored == frontier
    assert restored.ready_nodes() == ["c"]
    assert restored.completed_nodes() == ["a", "b"]
    assert restored.fingerprint == frontier.fingerprint

    restored_g = CompletionFrontier.restore(checkpoint, graph=make_graph())
    assert restored_g == frontier
    assert restored_g.ready_nodes() == ["c"]


def test_checkpoint_rejects_wrong_graph():
    frontier = make_frontier()
    frontier.complete("a")
    checkpoint = frontier.checkpoint()

    # Different structure (missing edge b->c).
    with pytest.raises(CheckpointMismatchError):
        CompletionFrontier.restore(
            checkpoint, NODES, [("a", "c"), ("c", "d")]
        )

    # Same shape, different node ordering -> different fingerprint.
    with pytest.raises(CheckpointMismatchError):
        CompletionFrontier.restore(
            checkpoint, ["d", "c", "b", "a"], EDGES
        )

    # A genuinely different graph must never accept the old state.
    other = nx.path_graph(4, create_using=nx.DiGraph)
    with pytest.raises(CheckpointMismatchError):
        CompletionFrontier.restore(checkpoint, graph=other)


def test_checkpoint_rejects_malformed_and_nonclosed_state():
    good = make_frontier()
    good.complete("a")
    good.complete("b")
    checkpoint = good.checkpoint()

    bad_version = {**checkpoint, "version": 99}
    with pytest.raises(CheckpointMismatchError):
        CompletionFrontier.restore(bad_version, NODES, EDGES)

    bad_format = {**checkpoint, "format": "something-else"}
    with pytest.raises(CheckpointMismatchError):
        CompletionFrontier.restore(bad_format, NODES, EDGES)

    # d completed without c (its predecessor) is not predecessor-closed.
    not_closed = {**checkpoint, "completed": [0, 3]}
    with pytest.raises(CheckpointMismatchError):
        CompletionFrontier.restore(not_closed, NODES, EDGES)

    out_of_range = {**checkpoint, "completed": [0, 99]}
    with pytest.raises(CheckpointMismatchError):
        CompletionFrontier.restore(out_of_range, NODES, EDGES)

    duplicates = {**checkpoint, "completed": [0, 0]}
    with pytest.raises(CheckpointMismatchError):
        CompletionFrontier.restore(duplicates, NODES, EDGES)

    with pytest.raises(CheckpointMismatchError):
        CompletionFrontier.restore(["not", "a", "dict"], NODES, EDGES)


def test_stable_output_order_follows_node_order():
    nodes = ["x", "a", "m", "q"]
    edges = [("x", "q"), ("a", "q"), ("m", "q")]
    frontier = CompletionFrontier(nodes, edges)
    assert frontier.ready_nodes() == ["x", "a", "m"]

    # q is released only after the last predecessor; stable order preserved.
    assert frontier.complete("m") == []
    frontier.complete("x")
    assert frontier.complete("a") == ["q"]
    assert frontier.completed_nodes() == ["x", "a", "m"]


def test_checkpoint_does_not_alias_frontier_state():
    frontier = make_frontier()
    frontier.complete("a")
    checkpoint = frontier.checkpoint()
    checkpoint["completed"].append(3)
    checkpoint["fingerprint"]["hexdigest"] = "tampered"
    assert frontier.checkpoint()["completed"] == [0]
    assert len(frontier.checkpoint()["fingerprint"]["hexdigest"]) == 64


def test_copy_is_independent():
    frontier = make_frontier()
    clone = frontier.copy()
    frontier.complete("a")
    assert clone.ready_nodes() == ["a", "b"]
    assert clone != frontier


def test_int_and_tuple_labels_round_trip():
    nodes = [1, 2, (3, 4)]
    edges = [(1, (3, 4)), (2, (3, 4))]
    frontier = CompletionFrontier(nodes, edges)
    frontier.complete(1)
    assert frontier.ready_nodes() == [2]
    restored = CompletionFrontier.restore(frontier.checkpoint(), nodes, edges)
    assert restored == frontier
    assert restored.completed_nodes() == [1]

