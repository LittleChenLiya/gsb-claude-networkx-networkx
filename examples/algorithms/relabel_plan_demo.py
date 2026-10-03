"""
============
Relabel plan
============

Batch-rename nodes by first generating a plan, then applying it to a copy.

This demo shows how to use :func:`networkx.plan_relabel_nodes` and
:func:`networkx.apply_relabel_plan` to rename nodes in one batch without
ever mutating the original graph. The plan snapshots the nodes and (keyed)
edges of the graph; applying it checks that the graph is unchanged, copies
all attribute dictionaries, and rejects the whole batch on conflict.

Run directly with::

    python examples/algorithms/relabel_plan_demo.py
"""

import networkx as nx


def main():
    G = nx.MultiDiGraph()
    G.add_edge("a", "b", key="x", weight=1)
    G.add_edge("a", "b", key="y", weight=2)
    G.add_edge("b", "c", key="z", weight=3)
    G.nodes["a"]["color"] = "red"

    # 1) chain rename a->b, b->c, c->d (names are freed in order)
    plan = nx.plan_relabel_nodes(G, {"a": "b", "b": "c", "c": "d"})
    H = nx.apply_relabel_plan(G, plan)
    print("after a->b, b->c, c->d:", sorted(H.edges(keys=True)))
    print("original unchanged:     ", sorted(G.edges(keys=True)))

    # 2) swap the two old names a and b
    plan = nx.plan_relabel_nodes(G, {"a": "b", "b": "a"})
    H = nx.apply_relabel_plan(G, plan)
    print("after a<->b swap:       ", sorted(H.edges(keys=True)))
    print("original unchanged:     ", sorted(G.edges(keys=True)))

    # 3) same endpoints, different keys: no edges lost
    print("edges preserved:        ", len(H.edges) == len(G.edges))

    # 4) adding an edge after planning conflicts with the whole batch
    plan = nx.plan_relabel_nodes(G, {"a": "a1", "b": "b1", "c": "c1"})
    G.add_edge("a", "c", key="new")
    try:
        nx.apply_relabel_plan(G, plan)
    except nx.RelabelConflict as err:
        print("conflict reported:      ", err)


if __name__ == "__main__":
    main()
