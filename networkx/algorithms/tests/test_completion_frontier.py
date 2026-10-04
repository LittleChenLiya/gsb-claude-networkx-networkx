"""Tests for :class:`networkx.algorithms.dag.CompletionFrontier`."""

import pytest

import networkx as nx
from networkx.algorithms.dag import CompletionFrontier


class TestCompletionFrontier:
    def test_initial_ready_are_root_nodes(self):
        G = nx.DiGraph([("a", "c"), ("b", "c"), ("c", "d")])
        frontier = CompletionFrontier(G)
        assert frontier.ready == ("a", "b")
        assert frontier.completed == ()

    def test_acceptance_diamond(self):
        # a -> c, b -> c, c -> d: only a and b are executable initially;
        # completing a must not release c; completing b must release c.
        # Nodes are declared in topological order so the stable output
        # order reads a, b, c, d.
        G = nx.DiGraph()
        G.add_nodes_from(["a", "b", "c", "d"])
        G.add_edges_from([("a", "c"), ("b", "c"), ("c", "d")])
        frontier = CompletionFrontier(G)
        assert frontier.ready == ("a", "b")
        frontier.complete("a")
        assert frontier.ready == ("b",)  # c still blocked by b
        frontier.complete("b")
        assert frontier.ready == ("c",)
        frontier.complete("c")
        assert frontier.ready == ("d",)
        frontier.complete("d")
        assert frontier.ready == ()
        assert frontier.completed == ("a", "b", "c", "d")

    def test_illegal_early_completion(self):
        frontier = CompletionFrontier(nx.DiGraph([("a", "c"), ("b", "c"), ("c", "d")]))
        with pytest.raises(ValueError, match="not executable"):
            frontier.complete("c")
        # the failed completion must not change the state
        assert frontier.ready == ("a", "b")
        assert frontier.completed == ()

    def test_illegal_completion_unknown_node(self):
        frontier = CompletionFrontier(nx.DiGraph([("a", "b")]))
        with pytest.raises(ValueError, match="not in the graph"):
            frontier.complete("zzz")

    def test_duplicate_confirmation_is_idempotent(self):
        frontier = CompletionFrontier(nx.DiGraph([("a", "c"), ("b", "c"), ("c", "d")]))
        frontier.complete("a")
        frontier.complete("a")
        frontier.complete("a")
        assert frontier.completed == ("a",)
        assert frontier.ready == ("b",)

    def test_complete_returns_self(self):
        frontier = CompletionFrontier(nx.DiGraph([("a", "b")]))
        assert frontier.complete("a") is frontier

    def test_cycle_rejected(self):
        with pytest.raises(ValueError, match="cycle"):
            CompletionFrontier(nx.DiGraph([("a", "b"), ("b", "a")]))

    def test_self_loop_rejected(self):
        G = nx.DiGraph()
        G.add_edge("a", "a")
        with pytest.raises(ValueError, match="cycle"):
            CompletionFrontier(G)

    def test_undirected_graph_rejected(self):
        with pytest.raises(ValueError, match="directed"):
            CompletionFrontier(nx.Graph([("a", "b")]))

    def test_duplicate_edges_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            CompletionFrontier([("a", "b"), ("a", "b")])

    def test_snapshot_is_independent_of_caller_graph(self):
        G = nx.DiGraph([("a", "b")])
        frontier = CompletionFrontier(G)
        # mutate the caller's graph after the frontier was created
        G.add_edge("a", "c")
        G.remove_edge("a", "b")
        assert frontier.ready == ("a",)
        frontier.complete("a")
        assert frontier.ready == ("b",)  # c is not part of the snapshot

    def test_output_order_follows_graph_node_order(self):
        # two roots, inserted in reverse topological order
        G = nx.DiGraph()
        G.add_nodes_from(["b", "a"])
        assert CompletionFrontier(G).ready == ("b", "a")
        # the single root comes first regardless of insertion order
        G2 = nx.DiGraph()
        G2.add_node("b")
        G2.add_node("a")
        G2.add_edge("a", "b")
        assert CompletionFrontier(G2).ready == ("a",)

    def test_checkpoint_roundtrip(self):
        G = nx.DiGraph([("a", "c"), ("b", "c"), ("c", "d")])
        frontier = CompletionFrontier(G)
        frontier.complete("a")
        frontier.complete("b")
        checkpoint = frontier.checkpoint()
        assert checkpoint["format"] == "networkx.CompletionFrontier/v1"
        assert isinstance(checkpoint["fingerprint"], str)
        assert len(checkpoint["fingerprint"]) == 64
        assert checkpoint["completed"] == ["a", "b"]

        restored = CompletionFrontier(G).restore(checkpoint)
        assert restored.ready == ("c",)
        assert restored.completed == ("a", "b")
        # the checkpoint is a plain dict: no shared mutable state
        checkpoint["completed"].append("d")
        assert restored.completed == ("a", "b")

    def test_from_checkpoint_classmethod(self):
        G = nx.DiGraph([("a", "c"), ("b", "c"), ("c", "d")])
        checkpoint = CompletionFrontier(G).complete("a").checkpoint()
        restored = CompletionFrontier.from_checkpoint(G, checkpoint)
        assert restored.ready == ("b",)
        assert restored.completed == ("a",)

    def test_wrong_graph_restore_rejected(self):
        G1 = nx.DiGraph([("a", "c"), ("b", "c"), ("c", "d")])
        G2 = nx.DiGraph([("x", "y")])
        checkpoint = CompletionFrontier(G1).complete("a").checkpoint()

        frontier2 = CompletionFrontier(G2)
        with pytest.raises(ValueError, match="fingerprint"):
            frontier2.restore(checkpoint)
        # the rejected restore must leave the frontier untouched
        assert frontier2.completed == ()
        assert frontier2.ready == ("x",)

    def test_restore_rejects_unknown_node(self):
        G = nx.DiGraph([("a", "b")])
        frontier = CompletionFrontier(G)
        checkpoint = frontier.checkpoint()
        checkpoint["completed"] = ["zzz"]
        with pytest.raises(ValueError, match="not in the graph"):
            frontier.restore(checkpoint)

    def test_restore_rejects_illegal_early_completion(self):
        G = nx.DiGraph([("a", "c"), ("b", "c"), ("c", "d")])
        frontier = CompletionFrontier(G)
        checkpoint = frontier.checkpoint()
        # tamper: mark c completed while its prerequisites are not
        checkpoint["completed"] = ["c"]
        with pytest.raises(ValueError, match="prerequisite"):
            frontier.restore(checkpoint)

    def test_restore_returns_self(self):
        frontier = CompletionFrontier(nx.DiGraph([("a", "b")]))
        assert frontier.restore(frontier.checkpoint()) is frontier

    def test_fingerprint_stable_and_structure_sensitive(self):
        G = nx.DiGraph([("a", "b")])
        assert (
            CompletionFrontier(G).fingerprint
            == CompletionFrontier(nx.DiGraph([("a", "b")])).fingerprint
        )
        # different edge -> different fingerprint
        assert (
            CompletionFrontier(G).fingerprint
            != CompletionFrontier(nx.DiGraph([("a", "c")])).fingerprint
        )
        # same structure built via edge iterable -> same fingerprint
        assert (
            CompletionFrontier(G).fingerprint
            == CompletionFrontier([("a", "b")]).fingerprint
        )

    def test_empty_graph(self):
        frontier = CompletionFrontier(nx.DiGraph())
        assert frontier.ready == ()
        assert frontier.completed == ()
        checkpoint = frontier.checkpoint()
        assert checkpoint["completed"] == []
        restored = CompletionFrontier(nx.DiGraph()).restore(checkpoint)
        assert restored.ready == ()

    def test_repr(self):
        frontier = CompletionFrontier(nx.DiGraph([("a", "b")]))
        assert "CompletionFrontier" in repr(frontier)
