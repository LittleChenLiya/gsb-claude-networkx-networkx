"""
==================
Bridge Contraction
==================

This example demonstrates ``nx.bridge_contraction``: every connected
component that remains after deleting all bridges (a 2-edge-connected
*block*) is contracted to a single node, and the bridges become the
edges between blocks.  The result of contracting any undirected graph
this way is a forest -- the *bridge tree* of the graph.

The view keeps the full mapping back to the original graph: which
source nodes form each block, and which original edge is each
inter-block bridge.
"""

import matplotlib.pyplot as plt

import networkx as nx

# Two triangles joined by a single bridge, plus an isolated node.
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 0)], block="left triangle")
G.add_edges_from([(3, 4), (4, 5), (5, 3)], block="right triangle")
G.add_edge(2, 3, color="red")  # the only bridge
G.add_node("isolated")

bc = nx.bridge_contraction(G)

# Block decomposition: block ids follow the input node order.
for block in bc:
    print(f"block {block}: source nodes {sorted(bc.members(block), key=str)}")

# Bridge identity: which original edge joins the blocks?
print("bridges:", bc.bridges())
print("is (2, 3) a bridge?", bc.is_bridge(2, 3))
print("is (0, 1) a bridge?", bc.is_bridge(0, 1))
print("bridge between blocks 0 and 1:", bc.bridge_between(0, 1))

# The contracted graph is a forest of block nodes; its edges carry an
# independent copy of the original bridge's attributes.
print("contracted edges:", list(bc.graph.edges(data=True)))

# %%
# Draw the original graph (bridges in red) next to its bridge tree.

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

pos = nx.spring_layout(G, seed=42)
colors = ["red" if bc.is_bridge(u, v) else "black" for u, v in G.edges()]
nx.draw(G, pos, ax=axes[0], with_labels=True, edge_color=colors)
axes[0].set_title("Original graph (bridge in red)")

cpos = nx.spring_layout(bc.graph, seed=42)
nx.draw(bc.graph, cpos, ax=axes[1], with_labels=True, node_color="orange")
axes[1].set_title("Bridge contraction (blocks as nodes)")

fig.tight_layout()
plt.show()

