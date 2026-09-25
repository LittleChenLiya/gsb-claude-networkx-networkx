"""
Demo: circulation with lower/upper edge bounds and node demands.

Builds a small delivery network, solves for a feasible integer flow with
``nx.lower_bound_circulation`` and prints the flow on every edge. Then the
key corridor is tightened until the instance becomes infeasible, and the
cut witness plus deficit carried by ``nx.InfeasibleCirculationError`` is
printed and verified.
"""

import networkx as nx


def print_flows(G, flow):
    for u, v, k, data in (
        G.edges(data=True, keys=True)
        if G.is_multigraph()
        else ((u, v, None, d) for u, v, d in G.edges(data=True))
    ):
        f = flow[u][v][k] if G.is_multigraph() else flow[u][v]
        lo = data.get("lower", 0)
        hi = data.get("upper", "inf")
        label = f"{u} -> {v}" + (f" (key {k})" if k is not None else "")
        print(f"  {label:18s} flow={f}   [{lo}, {hi}]")


def build_network():
    G = nx.DiGraph()
    # Warehouses (negative demand = supply) and stores (positive demand).
    G.add_edge("factory", "hub", lower=2, upper=6)
    G.add_edge("factory", "depot", lower=0, upper=4)
    G.add_edge("hub", "store1", lower=1, upper=4)
    G.add_edge("hub", "store2", lower=0, upper=3)
    G.add_edge("depot", "store2", lower=1, upper=4)
    G.add_edge("depot", "hub", lower=0, upper=2)
    G.nodes["factory"]["demand"] = -7
    G.nodes["store1"]["demand"] = 3
    G.nodes["store2"]["demand"] = 4
    return G


def main():
    G = build_network()
    print("Feasible instance: flow on every edge")
    flow = nx.lower_bound_circulation(G)
    print_flows(G, flow)

    print()
    print("MultiDiGraph with parallel edges (distinguished by key)")
    M = nx.MultiDiGraph()
    M.add_edge("a", "b", lower=1, upper=2)
    M.add_edge("a", "b", lower=0, upper=3)
    M.nodes["a"]["demand"] = -3
    M.nodes["b"]["demand"] = 3
    print_flows(M, nx.lower_bound_circulation(M))

    print()
    print("Tightening factory -> hub to upper=2 makes it infeasible:")
    G["factory"]["hub"]["upper"] = 2
    try:
        nx.lower_bound_circulation(G)
    except nx.InfeasibleCirculationError as exc:
        S = set(exc.cut)
        T = set(G) - S
        print(f"  failure cut S = {sorted(S)}")
        print(f"  deficit       = {exc.deficit}")
        # Verify the certificate: the net inflow required by T exceeds the
        # capacity crossing the cut from S to T by exactly the deficit.
        upper_ST = sum(
            d.get("upper", 0) for u, v, d in G.edges(data=True) if u in S and v in T
        )
        lower_TS = sum(
            d.get("lower", 0) for u, v, d in G.edges(data=True) if u in T and v in S
        )
        demand_T = sum(G.nodes[n].get("demand", 0) for n in T)
        print(
            f"  demand(T) = {demand_T}, upper(S->T) = {upper_ST}, "
            f"lower(T->S) = {lower_TS}"
        )
        assert demand_T - (upper_ST - lower_TS) == exc.deficit > 0
        print(
            "  certificate verified: demand(T) - (upper(S->T) - lower(T->S)) "
            "== deficit > 0"
        )


if __name__ == "__main__":
    main()
