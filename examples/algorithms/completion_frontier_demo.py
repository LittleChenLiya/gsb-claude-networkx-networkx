"""Short demonstration of ``nx.CompletionFrontier`` for DAG task scheduling.

The frontier tracks which tasks are done and which are executable; it never
runs the tasks itself. Run from the repository root with::

    PYTHONPATH=. python examples/algorithms/completion_frontier_demo.py
"""

import json

import networkx as nx
from networkx.algorithms.completion_frontier import (
    CheckpointMismatchError,
    CompletionFrontier,
    IllegalCompletionError,
)


def show(title, frontier):
    print(f"[{title}] ready={frontier.ready_nodes()} "
          f"completed={frontier.completed_nodes()}")


def main():
    # DAG: a -> c, b -> c, c -> d. Nodes are added first so that their
    # insertion order (a, b, c, d) is the stable output order.
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    graph.add_edges_from([("a", "c"), ("b", "c"), ("c", "d")])

    frontier = CompletionFrontier.from_graph(graph)
    print("fingerprint (sha256, binds state to this graph):")
    print("   ", frontier.fingerprint)
    show("start", frontier)

    # Completing a alone does not make c executable: b is still pending.
    print("complete(a) ->", frontier.complete("a"))
    show("after a", frontier)

    # Illegal early completion is rejected.
    try:
        frontier.complete("c")
    except IllegalCompletionError as err:
        print("early complete(c) rejected:", err)

    # Repeated confirmation is idempotent.
    print("complete(a) again ->", frontier.complete("a"), "(idempotent)")

    # Once b finishes, c is released; then c releases d.
    print("complete(b) ->", frontier.complete("b"))
    show("after b", frontier)
    print("complete(c) ->", frontier.complete("c"))
    print("complete(d) ->", frontier.complete("d"))
    show("done", frontier)
    print("finished:", frontier.is_finished())

    # Checkpoint: JSON-serializable, completed nodes are stable node-order
    # indices (a=0, b=1, c=2, d=3), bound to the graph fingerprint.
    mid = CompletionFrontier.from_graph(graph)
    mid.complete("a")
    mid.complete("b")
    checkpoint = mid.checkpoint()
    print("\ncheckpoint after a and b:")
    print(json.dumps(checkpoint, indent=2))

    restored = CompletionFrontier.restore(checkpoint, graph=graph)
    print("restored ready set:", restored.ready_nodes(),
          "(same frontier:", restored == mid, ")")

    # External mutation of the graph cannot affect an existing frontier.
    graph.add_edge("a", "d")
    print("\nafter caller mutates graph, restored frontier still sees ready:",
          restored.ready_nodes())

    # Restoring the checkpoint against a different graph fails loudly instead
    # of silently applying stale state.
    other = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d")])
    try:
        CompletionFrontier.restore(checkpoint, graph=other)
    except CheckpointMismatchError as err:
        print("restore onto different graph rejected:", err)


if __name__ == "__main__":
    main()

