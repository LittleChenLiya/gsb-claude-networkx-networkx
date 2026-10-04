"""
====================
Completion Frontier
====================

Scheduling tasks on a directed acyclic graph (DAG): a *completion frontier*
tracks which tasks are completed and which tasks are currently executable
(all prerequisites done), without executing any tasks itself.

This demo shows:

* the frontier releases a task only when **all** of its prerequisites are
  completed;
* completing the same task twice is a no-op (idempotent);
* a checkpoint is bound to a fingerprint of the graph structure, so a
  checkpoint exported from one graph cannot be loaded into another;
* the checkpoint format: ``{"format", "fingerprint", "completed"}``.
"""

import networkx as nx
from networkx.algorithms.dag import CompletionFrontier

# %%
# The diamond graph: a -> c, b -> c, c -> d. Nodes are declared in
# topological order so the stable output order reads a, b, c, d.
G = nx.DiGraph()
G.add_nodes_from(["a", "b", "c", "d"])
G.add_edges_from([("a", "c"), ("b", "c"), ("c", "d")])

frontier = CompletionFrontier(G)
print("initial ready:    ", frontier.ready)
print("initial completed:", frontier.completed)

# %%
# Completing a alone must not release c, because b is still pending.
frontier.complete("a")
print("after completing a, ready:", frontier.ready)

# Completing the same node twice is idempotent.
frontier.complete("a")
print("after completing a twice, ready:", frontier.ready)

# Completing b releases c; completing c releases d.
frontier.complete("b")
print("after completing b, ready:", frontier.ready)
frontier.complete("c")
print("after completing c, ready:", frontier.ready)

# %%
# A node cannot be completed before it is executable.
try:
    CompletionFrontier(G).complete("c")
except ValueError as err:
    print("early completion rejected:", err)

# %%
# Export a checkpoint. The fingerprint binds the state to this exact graph
# structure; "completed" lists the completed nodes in the graph's node order.
checkpoint = frontier.checkpoint()
print("checkpoint keys: ", sorted(checkpoint))
print("format:          ", checkpoint["format"])
print("fingerprint:     ", checkpoint["fingerprint"])
print("completed:       ", checkpoint["completed"])

# %%
# Reloading the checkpoint on a fresh frontier of the same graph yields the
# same frontier.
restored = CompletionFrontier(G).restore(checkpoint)
print("restored ready:    ", restored.ready)
print("restored completed:", restored.completed)

# %%
# A checkpoint from a different graph must be rejected: the fingerprint does
# not match, so old state can never be misused on another graph.
other = nx.DiGraph([("x", "y")])
try:
    CompletionFrontier(other).restore(checkpoint)
except ValueError as err:
    print("wrong-graph restore rejected:", err)
