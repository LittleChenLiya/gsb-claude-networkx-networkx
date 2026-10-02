"""
Short demo of the plan-then-apply batch node relabeling API.

Run from the repo root with:  PYTHONPATH=. python examples/basic/demo_relabel_plan.py
"""

import networkx as nx


def show(title, G, mapping):
    plan = nx.plan_relabel(G, mapping)
    H = nx.apply_relabel_plan(G, plan)
    edges = H.edges(keys=True) if H.is_multigraph() else H.edges()
    print(f"{title}: mapping={mapping}")
    print(f"  relabeled edges: {sorted(edges)}")
    print(f"  original edges : {sorted(G.edges())}  (original unchanged)")
    return H


def main():
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=1)

    show("a->b, b->c", G, {"a": "b", "b": "c"})
    show("swap a<->b", G, {"a": "b", "b": "a"})
    print("original graph still:", sorted(G.edges(data=True)))

    # MultiDiGraph: two parallel edges with different keys, nothing lost
    M = nx.MultiDiGraph()
    M.add_edge("u", "v", key="first", weight=1)
    M.add_edge("u", "v", key="second", weight=2)
    plan = nx.plan_relabel(M, {"u": "x"})
    H = nx.apply_relabel_plan(M, plan)
    print("MultiDiGraph relabeled edges:", sorted(H.edges(keys=True, data=True)))
    assert H.number_of_edges() == 2  # no edge lost

    # graph changed after planning -> whole batch rejected
    plan = nx.plan_relabel(G, {"a": "x"})
    G.add_edge("b", "c")
    try:
        nx.apply_relabel_plan(G, plan)
    except nx.RelabelConflictError as err:
        print("conflict detected, batch rejected:", err)


if __name__ == "__main__":
    main()
