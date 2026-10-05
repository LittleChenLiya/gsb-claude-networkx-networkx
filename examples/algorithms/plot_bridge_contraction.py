"""
==================
Bridge contraction
==================

Example of computing the bridge contraction of a graph: each connected
component that remains after deleting all bridges becomes a node (a
*block*), and each bridge becomes an edge between the blocks it joins.
The contraction is a forest whenever the original graph is connected.

This example draws a graph with several bridges, computes its bridge
contraction, and visualizes the decomposition: nodes are colored by
block and bridges are highlighted.
"""

import matplotlib.pyplot as plt

import networkx as nx

# Build a graph with two triangles joined by a bridge, plus a pendant
# node attached by another bridge and an isolated node.
G = nx.barbell_graph(3, 0)
G.add_edge(5, 6)  # pendant bridge attaching node 6 to node 5
G.add_node(7)  # isolated node

# Compute the bridge contraction.
H = nx.bridge_contraction(G)

print("Bridge contraction decomposition")
print("--------------------------------")
print(f"original nodes: {sorted(G.nodes)}")
print(f"original edges: {sorted(G.edges)}")
print(f"blocks (nodes of the contraction): {list(H.nodes)}")
for b in H.nodes:
    sources = H.nodes[b]["sources"]
    internal = H.nodes[b]["edges"]
    print(f"  block {b}: sources={sources}, internal edges={list(internal)}")
print(f"bridges (edges of the contraction): {list(H.edges)}")
for b, c in H.edges:
    print(f"  block edge ({b}, {c}) <- bridge {H.edges[b, c]['source']}")
print(f"node mapping: {H.graph['mapping']}")

# Visualize the decomposition.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: original graph colored by block.
block_of = H.graph["mapping"]
colors = [block_of[n] for n in G.nodes]
pos = nx.spring_layout(G, seed=42)
nx.draw(
    G,
    pos,
    ax=axes[0],
    node_color=colors,
    cmap=plt.cm.Set3,
    with_labels=True,
    node_size=600,
)
bridge_edges = [H.edges[b, c]["source"] for b, c in H.edges]
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=bridge_edges,
    ax=axes[0],
    edge_color="red",
    width=2.5,
)
axes[0].set_title("Original graph (red = bridges)")

# Right: bridge contraction.
pos_h = nx.spring_layout(H, seed=42)
nx.draw(
    H,
    pos_h,
    ax=axes[1],
    node_color=list(H.nodes),
    cmap=plt.cm.Set3,
    with_labels=True,
    node_size=800,
    edge_color="red",
    width=2.5,
)
axes[1].set_title("Bridge contraction")

fig.tight_layout()
plt.show()
