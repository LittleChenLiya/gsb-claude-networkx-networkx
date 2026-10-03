"""Tests for the gate index (necessary-node queries with versioned snapshots)."""

import pytest

import networkx as nx
from networkx.algorithms.gate import GateIndex, gate_query


class TestGateIndex:
    def test_diamond_counterexample(self):
        # Diamond: s->a->t and s->b->t. a is not a gate.
        G = nx.DiGraph([("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")])
        index = GateIndex(G, "s")
        result = index.query("t", "a")
        assert result.reachable is True
        assert result.necessary is False
        # The counterexample is a real s->t path that avoids a.
        assert result.path == ["s", "b", "t"]
        assert result.version == 0
        # Every consecutive pair is a real edge of the snapshot.
        assert all(G.has_edge(u, v) for u, v in zip(result.path, result.path[1:]))

    def test_update_publishes_new_version_old_snapshot_keeps_old_graph(self):
        G = nx.DiGraph([("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")])
        index = GateIndex(G, "s")
        v0 = index.snapshot(0)

        v1 = index.update(remove_edges=[("b", "t")])
        assert v1 == 1
        assert index.current_version == 1

        # New snapshot: a is now a necessary gate.
        result = index.query("t", "a")
        assert result.reachable is True
        assert result.necessary is True
        assert result.path is None
        assert result.version == 1

        # Old snapshot is retained and still queries the old graph:
        # a is not necessary and the counterexample goes through b.
        old = index.query("t", "a", version=0)
        assert old.necessary is False
        assert old.path == ["s", "b", "t"]
        assert old.version == 0
        assert index.snapshot(0) is v0

        # The original graph is untouched.
        assert G.has_edge("b", "t") is True

    def test_failed_batch_is_rejected_and_keeps_version(self):
        G = nx.DiGraph([("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")])
        index = GateIndex(G, "s")
        before = index.current_version
        with pytest.raises(nx.NetworkXError, match=r"rejected batch"):
            index.update(
                add_edges=[("a", "b")],  # new edge
                remove_edges=[("b", "t"), ("x", "y")],  # x->y does not exist
            )
        # No version change, and the new edge was not added.
        assert index.current_version == before
        assert index.versions() == [0]
        assert index.snapshot(0).has_edge("a", "b") is False
        assert index.query("t", "a").necessary is False

    def test_unreachable_is_not_a_gate(self):
        # t is unreachable from s; it must not be reported as a gate.
        G = nx.DiGraph([("s", "a"), ("c", "t")])
        index = GateIndex(G, "s")
        result = index.query("t", "a")
        assert result.reachable is False
        assert result.necessary is False
        assert result.path is None

    def test_missing_nodes_raise(self):
        G = nx.DiGraph([("s", "a"), ("a", "t")])
        with pytest.raises(nx.NodeNotFound, match=r"start node"):
            GateIndex(G, "x")
        index = GateIndex(G, "s")
        with pytest.raises(nx.NodeNotFound, match=r"end node"):
            index.query("nope", "a")
        with pytest.raises(nx.NodeNotFound, match=r"gate node"):
            index.query("t", "nope")
        with pytest.raises(nx.NetworkXError, match=r"unknown snapshot version"):
            index.query("t", "a", version=5)

    def test_undirected_raises(self):
        with pytest.raises(nx.NetworkXNotImplemented, match=r"directed"):
            GateIndex(nx.Graph([("s", "a")]), "s")

    @pytest.mark.parametrize("gate", ["s", "t"])
    def test_gate_equal_to_endpoint(self, gate):
        G = nx.DiGraph([("s", "a"), ("a", "t")])
        result = GateIndex(G, "s").query("t", gate)
        assert result.reachable is True
        assert result.necessary is True
        assert result.path is None

    def test_gate_equal_to_start_with_cycle(self):
        # Every path starts at s, so s is always a necessary gate.
        G = nx.DiGraph([("s", "s"), ("s", "a"), ("a", "t")])
        result = GateIndex(G, "s").query("t", "s")
        assert result.reachable is True
        assert result.necessary is True

    def test_loops_do_not_cause_false_counterexample(self):
        # s->s loop plus s->a->t: a is a gate despite the loop.
        G = nx.DiGraph([("s", "s"), ("s", "a"), ("a", "t")])
        index = GateIndex(G, "s")
        result = index.query("t", "a")
        assert result.necessary is True
        # A loop on the gate itself also does not help avoid it.
        G2 = nx.DiGraph([("s", "a"), ("a", "a"), ("a", "t")])
        assert GateIndex(G2, "s").query("t", "a").necessary is True

    def test_loop_provides_real_counterexample(self):
        # s->s->b->t avoids a; the loop must not trap the BFS.
        G = nx.DiGraph([("s", "s"), ("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")])
        result = GateIndex(G, "s").query("t", "a")
        assert result.necessary is False
        assert result.path == ["s", "b", "t"]

    def test_same_query_uses_one_version(self):
        # Reachability and counterexample of one query come from one snapshot.
        G = nx.DiGraph([("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")])
        index = GateIndex(G, "s")
        index.update(remove_edges=[("b", "t")])
        result = index.query("t", "a", version=0)
        assert result.version == 0
        assert result.path == ["s", "b", "t"]
        # The version-1 snapshot has no path through b.
        assert index.query("t", "a", version=1).path is None

    def test_snapshot_is_read_only(self):
        G = nx.DiGraph([("s", "a"), ("a", "t")])
        snap = GateIndex(G, "s").snapshot()
        # Frozen adjacency: no mutation method may change the snapshot.
        with pytest.raises((TypeError, AttributeError)):
            snap._adj["s"].add("t")
        with pytest.raises(TypeError):
            snap._adj["s"] = frozenset()
        # MultiDiGraph snapshots are frozen too.
        M = nx.MultiDiGraph([("s", "a", 0)])
        msnap = GateIndex(M, "s").snapshot()
        with pytest.raises((TypeError, AttributeError)):
            msnap._adj["s"]["a"].add(1)

    def test_gate_query_helper(self):
        G = nx.DiGraph([("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")])
        result = gate_query(G, "s", "t", "a")
        assert result.necessary is False
        assert result.path == ["s", "b", "t"]


class TestMultiDiGraph:
    def test_parallel_edge_keys_are_tracked(self):
        # Parallel edges s->a with keys 0 and 1; removing one key must not
        # remove the whole edge.
        M = nx.MultiDiGraph()
        M.add_edge("s", "a", key=0)
        M.add_edge("s", "a", key=1)
        M.add_edge("a", "t", key=0)
        M.add_edge("s", "b", key=0)
        M.add_edge("b", "t", key=0)
        index = GateIndex(M, "s")
        assert index.query("t", "a").necessary is False
        assert index.query("t", "a").path == ["s", "b", "t"]

        index.update(remove_edges=[("s", "a", 0)])
        snap = index.snapshot(1)
        assert snap.has_edge("s", "a", key=0) is False
        assert snap.has_edge("s", "a", key=1) is True
        # a is still not a gate: the parallel key-1 edge plus b route work.
        assert index.query("t", "a").necessary is False

    def test_remove_all_keys(self):
        M = nx.MultiDiGraph()
        M.add_edge("s", "a", key=0)
        M.add_edge("s", "a", key=1)
        M.add_edge("a", "t", key=0)
        index = GateIndex(M, "s")
        index.update(remove_edges=[("s", "a")])  # removes both parallel edges
        assert index.snapshot(1).has_edge("s", "a") is False
        assert index.query("t", "a").reachable is False

    def test_batch_with_nonexistent_key_is_rejected(self):
        M = nx.MultiDiGraph()
        M.add_edge("s", "a", key=0)
        M.add_edge("a", "t", key=0)
        M.add_edge("s", "b", key=0)
        M.add_edge("b", "t", key=0)
        index = GateIndex(M, "s")
        before = index.current_version
        with pytest.raises(nx.NetworkXError, match=r"rejected batch"):
            # key 5 of b->t does not exist; the whole batch is rejected.
            index.update(
                add_edges=[("b", "t", 1)],
                remove_edges=[("b", "t", 5)],
            )
        assert index.current_version == before
        assert index.snapshot(0).has_edge("b", "t", key=0) is True
        assert index.snapshot(0).has_edge("b", "t", key=1) is False
        assert index.query("t", "a").necessary is False

    def test_add_without_key_auto_generates(self):
        M = nx.MultiDiGraph()
        M.add_edge("s", "a", key=0)
        M.add_edge("a", "t", key=0)
        index = GateIndex(M, "s")
        index.update(add_edges=[("s", "a")])  # auto key
        snap = index.snapshot(1)
        assert snap.has_edge("s", "a", key=0) is True
        assert snap.has_edge("s", "a", key=1) is True

    def test_original_multigraph_unchanged(self):
        M = nx.MultiDiGraph()
        M.add_edge("s", "a", key=0)
        M.add_edge("a", "t", key=0)
        index = GateIndex(M, "s")
        index.update(remove_edges=[("a", "t", 0)])
        assert M.has_edge("a", "t", key=0) is True


def test_demo_print_two_versions():
    """Small demo: print the gate answer for both snapshot versions.

    Run with ``pytest -s`` to see the printed output.
    """
    G = nx.DiGraph([("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")])
    index = GateIndex(G, "s")
    index.update(remove_edges=[("b", "t")])

    printed = []
    for version in index.versions():
        result = index.query("t", "a", version=version)
        line = (
            f"snapshot v{version}: gate a necessary={result.necessary}, "
            f"counterexample={result.path}"
        )
        printed.append(line)
        print(line)

    assert printed[0] == (
        "snapshot v0: gate a necessary=False, counterexample=['s', 'b', 't']"
    )
    assert printed[1] == "snapshot v1: gate a necessary=True, counterexample=None"
