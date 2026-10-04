"""
Shortest paths with edges that may be traversed in reverse.

In a directed graph some edges are allowed to be traversed from their head to
their tail (against their natural direction) at a cost of one reversal, while
traversing an edge in its natural direction costs nothing. This is the 0-1
shortest path problem: forward moves have cost 0, reverse moves have cost 1.
"""

from heapq import heappop, heappush

import networkx as nx

__all__ = ["shortest_path_with_reversals"]


def _dijkstra_lexicographic(G, source, target, reversible):
    """Return ``(dist, path)`` for the lexicographically smallest shortest
    path from `source` to `target`.

    Forward edges cost 0, reversible reverse edges cost 1. Among paths with
    the same cost the one smallest by node insertion order, and then edge
    insertion order, is returned.
    """
    node_order = {node: i for i, node in enumerate(G.nodes)}
    order_to_node = {i: node for node, i in node_order.items()}
    inf = float("inf")
    dist = {source: 0}
    # best[v] is a tuple of (node_order, key, reversed) for each step from
    # source to v. Tuples compare lexicographically, which gives the
    # tie-break by node insertion order then edge insertion order.
    best = {source: ()}
    heap = [(0, node_order[source], source)]
    while heap:
        d, _, u = heappop(heap)
        if d != dist.get(u, inf):
            # A stale entry for u was popped.
            continue
        # Forward traversal: cost 0.
        for v in G.successors(u):
            if v == u:
                # A self-loop cannot move the path to another node.
                continue
            key = next(iter(G[u][v])) if G.is_multigraph() else None
            nd = d
            new_path = best[u] + ((node_order[v], key, False),)
            if nd < dist.get(v, inf):
                dist[v] = nd
                best[v] = new_path
                heappush(heap, (nd, node_order[v], v))
            elif nd == dist[v] and new_path < best[v]:
                best[v] = new_path
                heappush(heap, (nd, node_order[v], v))
        # Reverse traversal: cost 1, only for edges flagged `reversible`.
        for v in G.predecessors(u):
            if v == u:
                # Reversing a self-loop cannot move the path to another node.
                continue
            if G.is_multigraph():
                key = None
                for k, attr in G[v][u].items():
                    if attr.get(reversible, False):
                        key = k
                        break
                if key is None:
                    continue
            else:
                if not G[v][u].get(reversible, False):
                    continue
                key = None
            nd = d + 1
            new_path = best[u] + ((node_order[v], key, True),)
            if nd < dist.get(v, inf):
                dist[v] = nd
                best[v] = new_path
                heappush(heap, (nd, node_order[v], v))
            elif nd == dist[v] and new_path < best[v]:
                best[v] = new_path
                heappush(heap, (nd, node_order[v], v))
    # Reconstruct the actual edge sequence from the stored path.
    if target not in dist:
        return dist, None
    path = []
    u = source
    for no, key, rev in best[target]:
        v = order_to_node[no]
        path.append((u, v, key, rev))
        u = v
    return dist, path


@nx._dispatchable(edge_attrs="reversible")
def shortest_path_with_reversals(G, source, target, reversible="reversible"):
    """Return the minimum number of edge reversals and the edge sequence
    needed to travel from `source` to `target` in the directed graph `G`.

    An edge can always be traversed in its natural direction at cost 0. An
    edge can be traversed in the reverse direction at cost 1 (one reversal)
    only when its `reversible` edge attribute is true. Self-loops never move
    the path to another node, so reversing one is never useful and is not
    considered.

    Among paths with the same number of reversals the one that is smallest
    with respect to node insertion order, and then edge insertion order, is
    returned. The graph is not modified.

    Parameters
    ----------
    G : NetworkX directed graph

    source : node label
        Starting node.

    target : node label
        Ending node.

    reversible : string, optional (default='reversible')
        Name of the edge attribute that marks edges which may be traversed
        in reverse. Edges without the attribute, or with a falsey value,
        may not be reversed.

    Returns
    -------
    reversals : int
        Minimum number of edges that have to be traversed in reverse.

    path : list of tuples
        The edges traversed from `source` to `target` in order. Each tuple
        is ``(u, v, key, reversed)`` where ``u`` and ``v`` are the endpoints
        in traversal direction, ``key`` is the edge key (``None`` for
        graphs without parallel edges) and ``reversed`` is True when the
        edge was traversed against its natural direction.

    Raises
    ------
    NetworkXError
        If `G` is undirected.

    NodeNotFound
        If `source` or `target` is not in `G`.

    NetworkXNoPath
        If `target` is not reachable from `source`, even allowing
        reversible edges. An empty path is never returned for two distinct
    nodes.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge("a", "b")
    >>> G.add_edge("c", "b", reversible=True)
    >>> nx.shortest_path_with_reversals(G, "a", "c")
    (1, [('a', 'b', None, False), ('b', 'c', None, True)])

    With the reversible edge disabled, no path exists:

    >>> G["c"]["b"]["reversible"] = False
    >>> nx.shortest_path_with_reversals(G, "a", "c")
    Traceback (most recent call last):
        ...
    networkx.exception.NetworkXNoPath: Node c is not reachable from a even with reversible edges

    A direct forward edge is preferred over any detour:

    >>> G.add_edge("a", "c")
    >>> nx.shortest_path_with_reversals(G, "a", "c")
    (0, [('a', 'c', None, False)])

    Parallel edges report the exact key used:

    >>> G = nx.MultiDiGraph()
    >>> _ = G.add_edge("a", "b")
    >>> _ = G.add_edge("a", "b", reversible=True)
    >>> _ = G.add_edge("b", "c")
    >>> nx.shortest_path_with_reversals(G, "a", "c")
    (0, [('a', 'b', 0, False), ('b', 'c', 0, False)])
    >>> nx.shortest_path_with_reversals(G, "b", "a")
    (1, [('b', 'a', 1, True)])

    See Also
    --------
    shortest_path
    dijkstra_path
    """
    if not G.is_directed():
        raise nx.NetworkXError(
            "shortest_path_with_reversals() requires a directed graph"
        )
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not found in graph")
    if target not in G:
        raise nx.NodeNotFound(f"Target {target} not found in graph")

    if source == target:
        return 0, []

    dist, path = _dijkstra_lexicographic(G, source, target, reversible)
    if target not in dist:
        raise nx.NetworkXNoPath(
            f"Node {target} is not reachable from {source} "
            "even with reversible edges"
        )
    return dist[target], path
