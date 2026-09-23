"""Terminal demo for nx.temporal_connected_components.

Run with:  python examples/algorithms/temporal_components.py
"""

import networkx as nx

G = nx.Graph()
G.add_edge("a", "b", intervals=[(0, 4)])
G.add_edge("b", "c", intervals=[(2, 6)])

times = [4, 1, 2, 2]

print("Edges and active intervals (half-open [start, end)):")
for u, v, data in G.edges(data=True):
    print(f"  {u!r}-{v!r}: {data['intervals']}")
print(f"Query times: {times}")
print()

for t, components in zip(times, nx.temporal_connected_components(G, times)):
    print(f"  t={t}: {components}")

print()
print("Invalid interval is rejected and the graph is left unchanged:")
bad = nx.Graph()
bad.add_edge("x", "y", intervals=[(3, 1)])  # reversed: start >= end
before = nx.to_dict_of_dicts(bad)
try:
    nx.temporal_connected_components(bad, [0])
except ValueError as err:
    print(f"  ValueError: {err}")
print(f"  graph unchanged: {nx.to_dict_of_dicts(bad) == before}")
