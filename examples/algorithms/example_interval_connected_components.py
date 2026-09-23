"""
Terminal demo: offline connected-component queries on an interval graph.

Run with:  python examples/algorithms/example_interval_connected_components.py
"""

import networkx as nx

G = nx.Graph()
G.add_edge("a", "b", intervals=[(0, 4)])
G.add_edge("b", "c", intervals=[(2, 6)])
G.add_node("isolated")

times = [4, 1, 2, 2]

print("Edges and their active intervals (half-open [start, end)):")
for u, v, data in G.edges(data=True):
    print(f"  {u!r}-{v!r}: {data['intervals']}")
print(f"Isolated node: 'isolated'")
print(f"Query times: {times}\n")

answers = nx.interval_connected_components(G, times)
for t, components in zip(times, answers):
    print(f"t = {t}: {components}")

print("\nNow an invalid interval (start >= end) is rejected...")
G2 = nx.Graph()
G2.add_edge("x", "y", intervals=[(3, 3)])
before = {tuple(e): dict(d) for *e, d in G2.edges(data=True)}
try:
    nx.interval_connected_components(G2, [0])
except ValueError as err:
    print(f"  ValueError: {err}")
after = {tuple(e): dict(d) for *e, d in G2.edges(data=True)}
print(f"  Graph unchanged after failure: {before == after}")
