"""Short demo: minimum-reversal path with temporary backward edge use.

Run with::

    python examples/algorithms/minimum_reversals_demo.py
"""

import networkx as nx


def show(title, G, source, target):
    print(f"--- {title} ---")
    try:
        n, edges = nx.minimum_reversals_path(G, source, target)
    except nx.NetworkXNoPath:
        print(f"{source} -> {target}: no route (nothing reversible helps)")
        return
    print(f"{source} -> {target}: {n} reversal(s)")
    for e in edges:
        if e.key is None:
            used = f"{e.u}->{e.v}"
        else:
            used = f"{e.u}->{e.v} (key={e.key})"
        kind = "REVERSED" if e.reversed else "forward "
        print(f"    {kind} {used}")


def main():
    # a->b and c->b; only c->b may be walked backwards (b->c).
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("c", "b", reversible=True)

    show("one backward walk needed", G, "a", "c")

    # Turning reversibility off leaves c unreachable from a.
    G["c"]["b"]["reversible"] = False
    show("reversibility switched off", G, "a", "c")

    # A direct forward edge costs 0 and wins over the 1-reversal route.
    G["c"]["b"]["reversible"] = True
    G.add_edge("a", "c")
    show("zero-cost forward edge added", G, "a", "c")


if __name__ == "__main__":
    main()
