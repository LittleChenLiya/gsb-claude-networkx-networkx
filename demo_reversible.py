"""
Demo: shortest_path_with_reversals

Run with: python demo_reversible.py
"""

import networkx as nx


def main():
    print("=== Demo: shortest_path_with_reversals ===")
    print()

    # 1. Basic reversal: a->b and c->b (reversible), query a->c
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("c", "b", reversible=True)
    n, path = nx.shortest_path_with_reversals(G, "a", "c")
    print(f"1. Basic reversal (a->c): {n} reversal(s), path={path}")
    print()

    # 2. Turn reversible off -> no path
    G["c"]["b"]["reversible"] = False
    try:
        nx.shortest_path_with_reversals(G, "a", "c")
    except nx.NetworkXNoPath:
        print("2. Reversible disabled (a->c): NetworkXNoPath (no path)")
    print()

    # 3. Add a->c forward edge -> cost 0
    G.add_edge("a", "c")
    n, path = nx.shortest_path_with_reversals(G, "a", "c")
    print(f"3. Forward edge added (a->c): {n} reversal(s), path={path}")
    print()

    # 4. Same start/end
    n, path = nx.shortest_path_with_reversals(G, "a", "a")
    print(f"4. Same start/end (a->a): {n} reversal(s), path={path}")
    print()

    # 5. Parallel edges
    G = nx.MultiDiGraph()
    G.add_edge("a", "b")  # key 0
    G.add_edge("a", "b", reversible=True)  # key 1
    G.add_edge("b", "c")  # key 0
    n, path = nx.shortest_path_with_reversals(G, "a", "c")
    print(f"5. Parallel edges (a->c): {n} reversal(s), path={path}")
    n, path = nx.shortest_path_with_reversals(G, "b", "a")
    print(f"   Parallel edges (b->a): {n} reversal(s), path={path}")
    print()

    # 6. Cycle
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    G.add_edge("c", "a")
    n, path = nx.shortest_path_with_reversals(G, "a", "c")
    print(f"6. Cycle (a->c): {n} reversal(s), path={path}")
    n, path = nx.shortest_path_with_reversals(G, "a", "a")
    print(f"   Cycle (a->a): {n} reversal(s), path={path}")
    print()

    print("=== Demo complete ===")


if __name__ == "__main__":
    main()
