"""Tests for plan-based batch relabeling (plan_relabel_nodes/apply_relabel_plan)."""

import pytest

import networkx as nx
from networkx.relabel import (
    RelabelConflict,
    RelabelPlan,
    apply_relabel_plan,
    plan_relabel_nodes,
)


class TestRelabelPlan:
    def test_chain_relabel_and_original_unchanged(self):
        # a->b, b->c chain: renamed edges are returned, original is untouched
        G = nx.path_graph(3)
        G.nodes[0]["color"] = "red"
        plan = plan_relabel_nodes(G, {0: "a", 1: "b", 2: "c"})
        H = apply_relabel_plan(G, plan)
        assert sorted(H.nodes) == ["a", "b", "c"]
        assert sorted(H.edges) == [("a", "b"), ("b", "c")]
        # the original graph is unchanged
        assert sorted(G.nodes) == [0, 1, 2]
        assert sorted(G.edges) == [(0, 1), (1, 2)]
        assert G.nodes[0]["color"] == "red"

    def test_swap_two_nodes(self):
        # a<->b swap must work and leave the original graph unchanged
        G = nx.Graph([("a", "b"), ("b", "c")])
        plan = plan_relabel_nodes(G, {"a": "b", "b": "a"})
        H = apply_relabel_plan(G, plan)
        assert sorted(H.nodes) == ["a", "b", "c"]
        assert {tuple(sorted(e)) for e in H.edges} == {("a", "b"), ("a", "c")}
        # the original graph is unchanged
        assert sorted(G.nodes) == ["a", "b", "c"]
        assert sorted(G.edges) == [("a", "b"), ("b", "c")]

    def test_multidigraph_preserves_same_endpoint_edges(self):
        # two edges with the same endpoints but different keys must both survive
        G = nx.MultiDiGraph()
        G.add_edge("a", "b", key="x", weight=1)
        G.add_edge("a", "b", key="y", weight=2)
        G.add_edge("b", "c", key="z", weight=3)
        plan = plan_relabel_nodes(G, {"a": "a1", "b": "b1", "c": "c1"})
        H = apply_relabel_plan(G, plan)
        assert sorted(H.edges(keys=True)) == [
            ("a1", "b1", "x"),
            ("a1", "b1", "y"),
            ("b1", "c1", "z"),
        ]
        # the original graph is unchanged
        assert sorted(G.edges(keys=True)) == [
            ("a", "b", "x"),
            ("a", "b", "y"),
            ("b", "c", "z"),
        ]

    def test_conflict_after_edge_added(self):
        # adding an edge after planning invalidates the whole batch
        G = nx.path_graph(3)
        plan = plan_relabel_nodes(G, {0: "a", 1: "b", 2: "c"})
        G.add_edge(0, 2)
        with pytest.raises(RelabelConflict, match=r"edge\(s\) added"):
            apply_relabel_plan(G, plan)
        # no partial result: the original graph is unchanged
        assert sorted(G.edges) == [(0, 1), (0, 2), (1, 2)]
        assert sorted(G.nodes) == [0, 1, 2]

    def test_conflict_reports_all_differences_in_one_batch(self):
        # node and edge differences are reported together, not one by one
        G = nx.path_graph(3)
        plan = plan_relabel_nodes(G, {0: "a", 1: "b", 2: "c"})
        G.add_node(9)
        G.add_edge(0, 2)
        G.remove_edge(0, 1)
        with pytest.raises(RelabelConflict) as exc_info:
            apply_relabel_plan(G, plan)
        msg = str(exc_info.value)
        assert "node(s) added" in msg
        assert "edge(s) added" in msg
        assert "edge(s) missing" in msg

    def test_conflict_after_node_removed(self):
        G = nx.path_graph(3)
        plan = plan_relabel_nodes(G, {0: "a", 1: "b", 2: "c"})
        G.remove_node(2)
        with pytest.raises(RelabelConflict, match=r"node\(s\) missing"):
            apply_relabel_plan(G, plan)

    def test_target_name_collision_rejected(self):
        # two nodes cannot be merged into one target name
        G = nx.Graph([("a", "b"), ("b", "c")])
        with pytest.raises(RelabelConflict, match="merge"):
            plan_relabel_nodes(G, {"a": "x", "c": "x"})
        # renaming onto an existing, unmoved node is also a collision
        with pytest.raises(RelabelConflict, match="merge"):
            plan_relabel_nodes(G, {"a": "b"})
        # the graph is untouched by failed planning
        assert sorted(G.edges) == [("a", "b"), ("b", "c")]

    def test_unknown_mapping_key_rejected(self):
        # a mapping key that is not a node of G is reported, not silently ignored
        G = nx.path_graph(3)
        with pytest.raises(RelabelConflict, match="not in the graph"):
            plan_relabel_nodes(G, {"nope": "x"})

    def test_result_attributes_do_not_share_mutable_dicts(self):
        # mutating the result must not affect the original graph
        G = nx.Graph()
        G.add_node("a", data={"v": 1})
        G.add_node("b", data={"v": 2})
        G.add_edge("a", "b", data={"w": [1, 2]})
        plan = plan_relabel_nodes(G, {"a": "x", "b": "y"})
        H = apply_relabel_plan(G, plan)
        # the attribute dictionaries are not shared with the original
        assert H.nodes["x"] is not G.nodes["a"]
        assert H.nodes["y"] is not G.nodes["b"]
        assert H.edges["x", "y"] is not G.edges["a", "b"]
        # mutating the result's attributes does not affect the original
        H.nodes["x"]["color"] = 100
        H.edges["x", "y"]["weight"] = 5
        assert "color" not in G.nodes["a"]
        assert "weight" not in G.edges["a", "b"]
        assert G.nodes["a"]["data"] == {"v": 1}
        assert G.nodes["b"]["data"] == {"v": 2}
        assert G.edges["a", "b"]["data"] == {"w": [1, 2]}

    def test_plan_does_not_mutate_and_can_be_reapplied(self):
        G = nx.path_graph(3)
        plan = plan_relabel_nodes(G, {0: "a", 1: "b"})
        assert sorted(G.nodes) == [0, 1, 2]
        H1 = apply_relabel_plan(G, plan)
        H2 = apply_relabel_plan(G, plan)
        assert H1 is not H2
        assert sorted(H1.nodes, key=str) == sorted(H2.nodes, key=str)
        assert sorted(H1.edges, key=str) == sorted(H2.edges, key=str)
        # the original graph is still unchanged
        assert sorted(G.nodes) == [0, 1, 2]

    def test_directed_graph(self):
        G = nx.DiGraph([("a", "b"), ("b", "c")])
        plan = plan_relabel_nodes(G, {"a": "x", "b": "y", "c": "z"})
        H = apply_relabel_plan(G, plan)
        assert sorted(H.edges) == [("x", "y"), ("y", "z")]
        assert sorted(G.edges) == [("a", "b"), ("b", "c")]

    def test_plan_snapshot_is_a_RelabelPlan(self):
        G = nx.path_graph(3)
        plan = plan_relabel_nodes(G, {0: "a"})
        assert isinstance(plan, RelabelPlan)
        assert plan.multigraph is False
        assert plan.directed is False
        # applying a non-plan is rejected
        with pytest.raises(TypeError):
            apply_relabel_plan(G, {0: "a"})
