"""
===============
Team Projection
===============

Project a multidigraph onto a graph of team-to-team relationships.

Each node of the input graph belongs to a team (the ``team`` node
attribute).  Edges between nodes of different teams are *external* and are
aggregated, team pair by team pair, into a single weighted edge.  Edges
within a team are *internal* and are collected separately instead of being
aggregated.

The aggregator receives deep copies of the edge attributes, so it can
never modify the source graph.  This example shows how to:

* aggregate parallel edges between two teams (weights 2 and 3 -> 5),
* keep the provenance of every aggregated edge (the two source edges),
* keep internal edges out of the external total,
* recover gracefully when an aggregator fails, without touching the
  original graph.
"""

import os
import sys

# Make the worktree's networkx importable when running this script directly
# from a checkout (two levels up from examples/algorithms/).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import copy

import networkx as nx

# ---------------------------------------------------------------------------
# Build a small multidigraph with team attributes.
# ---------------------------------------------------------------------------
G = nx.MultiDiGraph()
G.add_node("x1", team="X")
G.add_node("x2", team="X")
G.add_node("y1", team="Y")
G.add_node("y2", team="Y")
G.add_node("z1")  # no team: edges touching this node are dropped

# Two X nodes connect to Y with weights 2 and 3.
G.add_edge("x1", "y1", key=0, weight=2, label="a")
G.add_edge("x2", "y2", key=0, weight=3, label="b")
# An internal edge within X: listed separately, not aggregated.
G.add_edge("x1", "x2", key=0, weight=100, label="internal")
# An edge to a node without a team: dropped entirely.
G.add_edge("x1", "z1", key=0, weight=99, label="dropped")

print("Original graph:")
print(f"  nodes: {sorted(G.nodes(data='team'))}")
print(f"  edges: {sorted(G.edges(keys=True, data='weight'))}")

# ---------------------------------------------------------------------------
# Project onto teams with the default aggregator (sums weights).
# ---------------------------------------------------------------------------
H, internal_edges = nx.team_projection(G)

print("\nProjected team graph:")
for u, v, data in sorted(H.edges(data=True)):
    print(f"  {u} -> {v}: weight={data['weight']}, sources={data['sources']}")

print("\nInternal edges (not aggregated):")
for u, v, key, data in internal_edges:
    print(f"  {u} -> {v} key={key} weight={data['weight']} label={data['label']}")

# The two source edges behind the X -> Y aggregate.
xy = H["X"]["Y"]
print(f"\nX -> Y total weight: {xy['weight']}")
print("Source edges of X -> Y:")
for u, v, key in xy["sources"]:
    print(f"  ({u}, {v}, {key}) weight={G[u][v][key]['weight']} "
          f"label={G[u][v][key]['label']}")

# ---------------------------------------------------------------------------
# A custom aggregator that lists every source edge.
# ---------------------------------------------------------------------------
def listing_aggregator(records):
    return {
        "weight": sum(d["weight"] for _, _, _, d in records),
        "labels": [d["label"] for _, _, _, d in records],
    }

H2, _ = nx.team_projection(G, aggregator=listing_aggregator)
print("\nCustom aggregator labels:", H2["X"]["Y"]["labels"])

# ---------------------------------------------------------------------------
# A failing aggregator must not modify the source graph.
# ---------------------------------------------------------------------------
snapshot = copy.deepcopy(G)

def bad_aggregator(records):
    for _, _, _, data in records:
        data["weight"] = 0  # mutate the copy ...
    raise RuntimeError("aggregator failure after mutation")

H3, _ = nx.team_projection(G, aggregator=bad_aggregator)
assert H3.number_of_edges() == 0  # failing group produces no edge
assert G.nodes(data=True) == snapshot.nodes(data=True)
assert list(G.edges(keys=True, data=True)) == list(
    snapshot.edges(keys=True, data=True)
)
print("\nFailing aggregator left the source graph unchanged.")
print("Done.")
