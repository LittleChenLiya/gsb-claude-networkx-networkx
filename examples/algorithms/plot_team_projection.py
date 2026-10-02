"""
===============
Team Projection
===============

Project a multidigraph onto teams given by a node attribute.

Two members of team X each send a weighted edge to team Y (weights 2 and
3).  The projection merges those external edges into a single X -> Y edge
whose weight is the sum (5), while keeping the original ``(u, v, key)``
sources.  An edge inside team X is reported separately and never enters
the external aggregate.
"""

import networkx as nx

G = nx.MultiDiGraph()
G.add_node("alice", team="X")
G.add_node("bob", team="X")
G.add_node("carol", team="Y")
G.add_edge("alice", "carol", weight=2)
G.add_edge("bob", "carol", weight=3)
G.add_edge("alice", "bob", weight=10)  # internal to team X

result = nx.team_projection(
    G, lambda attrs: {"weight": sum(a["weight"] for a in attrs)}
)

edge = result.graph["X"]["Y"][0]
print(f"Aggregated X -> Y weight: {edge['weight']}")
print("Source edges kept on the aggregate:")
for u, v, key, attrs in edge["sources"]:
    print(f"  ({u!r}, {v!r}, key={key}) attrs={attrs}")

print("Internal edges (not part of the external total):")
for u, v, key, attrs in result.internal_edges:
    print(f"  ({u!r}, {v!r}, key={key}) attrs={attrs}")
