"""
=======================
Lower Bound Circulation
=======================

Demo of :func:`networkx.lower_bound_circulation`: find a feasible
integer flow on a directed graph with integer lower/upper edge bounds
and node demands (positive demand = net inflow). On infeasible
instances the raised :class:`networkx.InfeasibleCirculation` carries a
verifiable cut witness and the deficit.
"""

import networkx as nx

# ---------------------------------------------------------------------
# Feasible instance: a MultiDiGraph with parallel edges and a self-loop.
# ---------------------------------------------------------------------
G = nx.MultiDiGraph()
G.add_edge("A", "B", key="fast", lower=2, upper=5)
G.add_edge("A", "B", key="slow", lower=0, upper=2)
G.add_edge("B", "C", key="only", lower=1, upper=4)
G.add_edge("C", "A", key="back", lower=0, upper=3)
G.add_edge("C", "C", key="loop", lower=1, upper=1)  # self-loop
G.nodes["A"]["demand"] = -3  # supplies 3 units
G.nodes["B"]["demand"] = 2
G.nodes["C"]["demand"] = 1

flow = nx.lower_bound_circulation(G)
print("Feasible flow:")
for u, v, key, data in G.edges(keys=True, data=True):
    f = flow[u][v][key]
    print(f"  {u} -> {v} [{key}]: flow {f} (bounds [{data['lower']}, {data['upper']}])")

# ---------------------------------------------------------------------
# Infeasible instance: tightening an upper bound breaks feasibility and
# the exception reports a cut witness and the deficit.
# ---------------------------------------------------------------------
H = nx.DiGraph()
H.add_edge("A", "B", lower=2, upper=2)
H.nodes["A"]["demand"] = -3
H.nodes["B"]["demand"] = 3

try:
    nx.lower_bound_circulation(H)
except nx.InfeasibleCirculation as exc:
    S, T = exc.cut
    print("\nNo feasible circulation.")
    print(f"  cut: S = {sorted(S)}, T = {sorted(T)}")
    print(f"  deficit: {exc.deficit} unit(s) of inflow missing for T")
