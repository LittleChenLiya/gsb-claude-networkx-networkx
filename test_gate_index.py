"""Small pytest demo for gate_index: run with ``python -m pytest -s``.

Covers the acceptance scenario (diamond graph, two snapshot versions),
atomic batch rejection, cycles, gate==endpoint, unreachable targets,
missing-node errors, and MultiDiGraph parallel-edge keys.
"""

import networkx as nx
import pytest

from gate_index import (
    BatchUpdateError,
    GateIndex,
    GateIndexError,
    MissingNodeError,
)


def diamond():
    G = nx.DiGraph()
    G.add_edges_from([("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")])
    return G


def test_diamond_two_versions_print_demo():
    """Acceptance: a is not unavoidable; after removing b->t it becomes so,
    while the old snapshot still serves the old counterexample."""
    G = diamond()
    v0 = GateIndex(G, "s")

    r0 = v0.query("t", "a")
    assert not r0.unavoidable
    assert r0.reachable
    assert r0.counterexample is not None
    assert "b" in r0.counterexample.nodes
    assert "a" not in r0.counterexample.nodes
    # counterexample is a real path in the version-0 graph
    assert r0.counterexample.nodes[0] == "s"
    assert r0.counterexample.nodes[-1] == "t"
    for u, v, _ in r0.counterexample.edges:
        assert G.has_edge(u, v)

    v1 = v0.update(remove_edges=[("b", "t")])
    assert v1.version == v0.version + 1
    r1 = v1.query("t", "a")
    assert r1.unavoidable
    assert r1.counterexample is None

    # old snapshot still answers against the old graph
    r0_again = v0.query("t", "a")
    assert not r0_again.unavoidable
    assert "b" in r0_again.counterexample.nodes
    assert r0_again.version == r0.version == 0

    # the source graph was never mutated
    assert G.has_edge("b", "t")

    print(f"\nversion {r0.version}: unavoidable(a)={r0.unavoidable}, "
          f"counterexample={r0.counterexample.nodes}")
    print(f"version {r1.version}: unavoidable(a)={r1.unavoidable}, "
          f"counterexample={r1.counterexample}")
    print(f"version {r0_again.version} (re-queried): "
          f"unavoidable(a)={r0_again.unavoidable}, "
          f"counterexample={r0_again.counterexample.nodes}")


def test_batch_remove_missing_edge_is_rejected_atomically():
    """A batch mixing a bogus removal with a valid addition is rejected as a
    whole: no new version, current snapshot unchanged."""
    v0 = GateIndex(diamond(), "s")
    with pytest.raises(BatchUpdateError, match="no such edge"):
        v0.update(remove_edges=[("s", "t")], add_edges=[("a", "b")])
    # failed update did not change the current version
    assert v0.version == 0
    assert not v0.has_edge("a", "b")
    assert not v0.has_edge("s", "t")
    # and the old query result is untouched
    assert not v0.query("t", "a").unavoidable


def test_batch_add_and_remove_together():
    v0 = GateIndex(diamond(), "s")
    v1 = v0.update(add_edges=[("s", "t")], remove_edges=[("s", "b")])
    assert v1.has_edge("s", "t")
    assert not v1.has_edge("s", "b")
    # direct edge s->t avoids both a and b
    r = v1.query("t", "a")
    assert not r.unavoidable
    assert r.counterexample.nodes == ("s", "t")


def test_unreachable_end_is_not_unavoidable():
    G = nx.DiGraph([("s", "a"), ("x", "t")])
    G.add_node("g")
    idx = GateIndex(G, "s")
    r = idx.query("t", "g")
    assert not r.reachable
    assert not r.unavoidable
    assert r.counterexample is None


def test_cycles_do_not_break_queries():
    G = nx.DiGraph([("s", "a"), ("a", "b"), ("b", "a"), ("b", "t"), ("a", "t")])
    idx = GateIndex(G, "s")
    assert idx.query("t", "b").unavoidable is False  # s->a->t avoids b
    assert idx.query("t", "a").unavoidable is True   # cycle always re-enters a
    # removing a->t makes b unavoidable too
    v1 = idx.update(remove_edges=[("a", "t")])
    assert v1.query("t", "b").unavoidable is True


def test_gate_equal_to_endpoints():
    idx = GateIndex(diamond(), "s")
    assert idx.query("t", "s").unavoidable is True
    assert idx.query("t", "t").unavoidable is True
    r = idx.query("s", "s")  # trivial empty path start==end==gate
    assert r.reachable and r.unavoidable


def test_missing_nodes_raise_clear_errors():
    idx = GateIndex(diamond(), "s")
    with pytest.raises(MissingNodeError, match="zzz"):
        idx.query("zzz", "a")
    with pytest.raises(MissingNodeError, match="zzz"):
        idx.query("t", "zzz")
    with pytest.raises(MissingNodeError):
        GateIndex(diamond(), "zzz")
    with pytest.raises(BatchUpdateError, match="absent"):
        idx.update(remove_edges=[("zzz", "t")])


def test_multidigraph_parallel_edges_and_keys():
    G = nx.MultiDiGraph()
    k0 = G.add_edge("s", "a")          # key 0
    k1 = G.add_edge("s", "a")          # key 1 (parallel)
    G.add_edge("a", "t")
    G.add_edge("s", "b")
    G.add_edge("b", "t")
    idx = GateIndex(G, "s")
    assert idx.is_multigraph

    r = idx.query("t", "a")
    assert not r.unavoidable
    assert "b" in r.counterexample.nodes
    # counterexample edges carry real keys
    assert all(len(e) == 3 for e in r.counterexample.edges)

    # removing one parallel edge by key leaves the other
    v1 = idx.update(remove_edges=[("s", "a", k0)])
    assert v1.has_edge("s", "a", k1)
    assert not v1.has_edge("s", "a", k0)
    assert idx.has_edge("s", "a", k0)  # old snapshot unchanged

    # removing a key that does not exist rejects the whole batch
    with pytest.raises(BatchUpdateError, match="no such edge"):
        idx.update(remove_edges=[("s", "a", 999)], add_edges=[("a", "b")])
    assert not idx.has_edge("a", "b")

    # removing all parallel s->a edges (no key) makes b unavoidable
    v2 = idx.update(remove_edges=[("s", "a")])
    assert v2.query("t", "b").unavoidable is True

    # additions without keys get fresh keys; explicit duplicate keys rejected
    v3 = idx.update(add_edges=[("s", "a"), ("s", "a", 7)])
    assert v3.has_edge("s", "a", 7)
    with pytest.raises(BatchUpdateError, match="key already exists"):
        v3.update(add_edges=[("s", "a", 7)])


def test_undirected_graph_rejected():
    with pytest.raises(GateIndexError, match="directed"):
        GateIndex(nx.Graph([(1, 2)]), 1)


def test_same_version_consistency():
    """Reachability verdict and counterexample of one query share a version."""
    v0 = GateIndex(diamond(), "s")
    v1 = v0.update(remove_edges=[("b", "t")])
    r0, r1 = v0.query("t", "a"), v1.query("t", "a")
    assert r0.version == 0 and r1.version == 1
    assert (r0.counterexample is not None) != (r1.counterexample is not None)
