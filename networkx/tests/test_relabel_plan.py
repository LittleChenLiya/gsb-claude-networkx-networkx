"""Tests for the plan-then-apply batch node relabeling API
(:func:`networkx.plan_relabel` and :func:`networkx.apply_relabel_plan`)."""

import pytest

import networkx as nx


def make_chain():
    # nodes {a, b} with a single edge a -> b, so that the chained mapping
    # a -> b, b -> c (and the swap a <-> b) does not merge existing nodes
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=1)
    return G


def test_chain_mapping_and_swap_leave_original_unchanged():
    G = make_chain()
    original_edges = sorted(G.edges())

    # a -> b, b -> c (chained targets)
    plan = nx.plan_relabel(G, {"a": "b", "b": "c"})
    H = nx.apply_relabel_plan(G, plan)
    print("a->b, b->c relabeled edges:", sorted(H.edges()))
    assert sorted(H.edges()) == [("b", "c")]
    assert H["b"]["c"]["weight"] == 1

    # swap a and b
    plan = nx.plan_relabel(G, {"a": "b", "b": "a"})
    H = nx.apply_relabel_plan(G, plan)
    print("swap a<->b relabeled edges:", sorted(H.edges()))
    assert sorted(H.edges()) == [("b", "a")]

    # the original graph must be untouched
    print("original edges after planning/applying:", sorted(G.edges()))
    assert sorted(G.edges()) == original_edges
    assert set(G.nodes()) == {"a", "b"}


def test_multidigraph_parallel_edges_with_keys_preserved():
    G = nx.MultiDiGraph()
    G.add_edge("u", "v", key="first", weight=1)
    G.add_edge("u", "v", key="second", weight=2)
    G.add_edge("v", "w", key="only", weight=3)

    plan = nx.plan_relabel(G, {"u": "x", "v": "y"})
    H = nx.apply_relabel_plan(G, plan)

    # no edge lost: both parallel edges survive under their original keys
    assert H.number_of_edges() == 3
    assert H["x"]["y"]["first"]["weight"] == 1
    assert H["x"]["y"]["second"]["weight"] == 2
    assert H["y"]["w"]["only"]["weight"] == 3
    assert sorted(H.edges(keys=True)) == [
        ("x", "y", "first"),
        ("x", "y", "second"),
        ("y", "w", "only"),
    ]
    # original untouched
    assert G.number_of_edges() == 3
    assert set(G.nodes()) == {"u", "v", "w"}


def test_apply_after_graph_change_raises_conflict_for_whole_batch():
    G = make_chain()
    plan = nx.plan_relabel(G, {"a": "x"})

    # graph changes after the plan was generated -> conflict
    G.add_edge("b", "c", weight=4)
    with pytest.raises(nx.RelabelConflictError):
        nx.apply_relabel_plan(G, plan)

    # node-set changes are also conflicts
    G = make_chain()
    plan = nx.plan_relabel(G, {"a": "x"})
    G.remove_node("b")
    with pytest.raises(nx.RelabelConflictError):
        nx.apply_relabel_plan(G, plan)

    # multigraph key-level changes are conflicts too
    G = nx.MultiDiGraph()
    G.add_edge("a", "b", key="k1")
    plan = nx.plan_relabel(G, {"a": "x"})
    G.add_edge("a", "b", key="k2")
    with pytest.raises(nx.RelabelConflictError):
        nx.apply_relabel_plan(G, plan)


def test_target_name_collision_and_node_merging_rejected():
    G = make_chain()

    # target label collides with an existing node that is not renamed
    with pytest.raises(nx.NetworkXError, match="collide"):
        nx.plan_relabel(G, {"a": "b"})

    # chained targets are fine when the final target is a fresh label
    plan = nx.plan_relabel(G, {"a": "b", "b": "c"})
    assert plan.mapping == {"a": "b", "b": "c"}

    # two nodes merged into a single target
    with pytest.raises(nx.NetworkXError, match="merge"):
        nx.plan_relabel(G, {"a": "z", "b": "z"})

    # mapping key that is not a node of the graph
    with pytest.raises(nx.NetworkXError, match="not nodes"):
        nx.plan_relabel(G, {"a": "x", "ghost": "y"})

    # ... but swapping two existing labels is fine
    plan = nx.plan_relabel(G, {"a": "b", "b": "a"})
    assert plan.mapping == {"a": "b", "b": "a"}


def test_result_attributes_do_not_share_mutable_dicts_with_original():
    G = nx.MultiDiGraph()
    G.graph["meta"] = {"tags": ["original"]}
    G.add_node("a", props={"count": 1})
    G.add_edge("a", "b", key="k", props={"count": 2})

    plan = nx.plan_relabel(G, {"a": "x", "b": "y"})
    H = nx.apply_relabel_plan(G, plan)

    # containers must not be shared with the original graph
    assert H.graph["meta"] is not G.graph["meta"]
    assert H.nodes["x"]["props"] is not G.nodes["a"]["props"]
    assert H["x"]["y"]["k"]["props"] is not G["a"]["b"]["k"]["props"]

    # mutating the result must not affect the original
    H.graph["meta"]["tags"].append("mutated")
    H.nodes["x"]["props"]["count"] = 99
    H["x"]["y"]["k"]["props"]["count"] = 99
    assert G.graph["meta"] == {"tags": ["original"]}
    assert G.nodes["a"]["props"] == {"count": 1}
    assert G["a"]["b"]["k"]["props"] == {"count": 2}


def test_callable_mapping_and_partial_mapping():
    G = make_chain()
    plan = nx.plan_relabel(G, str.upper)
    H = nx.apply_relabel_plan(G, plan)
    assert sorted(H.edges()) == [("A", "B")]

    # partial mapping: unmapped nodes keep their labels
    plan = nx.plan_relabel(G, {"b": "z"})
    H = nx.apply_relabel_plan(G, plan)
    assert sorted(H.edges()) == [("a", "z")]
