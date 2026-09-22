"""
=================
DAG Task Schedule
=================

Schedule tasks on a DAG with :func:`~networkx.algorithms.dag.dag_schedule`.

Node attributes ``duration`` (default 1) and ``release`` (default 0) give
the task processing time and earliest start; the edge attribute ``lag``
(default 0) gives a handover waiting time between tasks. The result reports
the earliest start (``es``), latest start (``ls``) and total float (``tf``)
of every task together with the critical path determining the makespan.
A deadline earlier than the natural makespan raises
``NetworkXUnfeasibleDeadline``, which carries the minimum feasible makespan.
"""

import matplotlib.pyplot as plt

import networkx as nx

G = nx.DiGraph()
G.add_node("A", duration=2)
G.add_node("B", duration=4, release=1)
G.add_node("C", duration=3)
G.add_edge("A", "C")
G.add_edge("B", "C", lag=2)

print("Input nodes:", {n: dict(d) for n, d in G.nodes(data=True)})
print("Input edges:", {(u, v): dict(d) for u, v, d in G.edges(data=True)})

result = nx.dag_schedule(G, deadline=12)
for node, item in result["schedule"].items():
    print(node, item)
print("makespan:", result["makespan"])
print("critical path:", result["critical_path"])

try:
    nx.dag_schedule(G, deadline=9)
except nx.NetworkXUnfeasibleDeadline as err:
    print("rejected deadline:", err.deadline, "minimum makespan:", err.makespan)

pos = nx.planar_layout(G)
critical_edges = list(nx.utils.pairwise(result["critical_path"]))
nx.draw_networkx_nodes(G, pos, node_color="lightblue")
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=G.edges, edge_color="gray")
nx.draw_networkx_edges(G, pos, edgelist=critical_edges, edge_color="red", width=2)
nx.draw_networkx_edge_labels(
    G, pos, edge_labels={(u, v): d.get("lag", 0) for u, v, d in G.edges(data=True)}
)
plt.axis("off")
plt.show()
