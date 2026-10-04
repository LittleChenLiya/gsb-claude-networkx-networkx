"""
Shortest paths where directed edges may temporarily be traversed backwards.

Forward traversal of an edge is free (cost 0). Traversing an edge against its
direction is only allowed when that edge carries a truthy ``reversible`` edge
attribute and costs 1. The resulting edge weights are all 0 or 1, so a
layered (dial) breadth-first search finds the path using the fewest
reversals in :math:`O(V + E)` time.
"""

from collections import deque

import networkx as nx

__all__ = ["TraversedEdge", "minimum_reversals_path"]


class TraversedEdge(tuple):
    """A directed edge as actually traversed in a minimum-reversal path.

    A 4-tuple ``(u, v, key, reversed)`` describing movement from node ``u``
    to node ``v``:

    - ``u`` and ``v`` are the tail and head in traversal direction.
    - ``key`` is the edge key used to identify parallel edges, or ``None``
      for non-multigraphs.
    - ``reversed`` is ``True`` when the underlying graph edge runs in the
      opposite direction (``v -> u``) and was walked backwards.
    """

    __slots__ = ()

    def __new__(cls, u, v, key, reversed):
        return tuple.__new__(cls, (u, v, key, reversed))

    def __repr__(self):
        return (
            f"TraversedEdge(u={self[0]!r}, v={self[1]!r}, key={self[2]!r}, "
            f"reversed={self[3]!r})"
        )

    u = property(lambda self: self[0])
    v = property(lambda self: self[1])
    key = property(lambda self: self[2])
    reversed = property(lambda self: self[3])


def _outgoing(G, u):
    """Yield ``(v, key, data)`` for every arc leaving ``u`` in insertion order."""
    if G.is_multigraph():
        for v, keydict in G.succ[u].items():
            for key, data in keydict.items():
                yield v, key, data
    else:
        for v, data in G.succ[u].items():
            yield v, None, data


def _incoming(G, u):
    """Yield ``(v, key, data)`` for every arc ``v -> u`` in insertion order."""
    if G.is_multigraph():
        for v, keydict in G.pred[u].items():
            for key, data in keydict.items():
                yield v, key, data
    else:
        for v, data in G.pred[u].items():
            yield v, None, data


@nx._dispatchable
def minimum_reversals_path(G, source, target, reversible="reversible"):
    r"""Shortest directed path allowing reversible edges to be walked backwards.

    Every edge can be followed in its defined direction at cost 0. An edge
    may additionally be traversed against its direction, at cost 1, only when
    it has a truthy ``reversible`` attribute. The returned path minimizes the
    number of such backward traversals using the 0-1 shortest-path idea.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph (``DiGraph`` or ``MultiDiGraph``).

    source : node
        Starting node for the path.

    target : node
        Ending node for the path.

    reversible : string
        Name of the edge attribute that marks an edge as walkable backwards.
        An edge is reversible only when this attribute is truthy.
        Default is ``"reversible"``.

    Returns
    -------
    num_reversals : int
        Minimum number of edges traversed against their direction.

    edges : list of :class:`TraversedEdge`
        The edges of the path in traversal order, each as a
        ``(u, v, key, reversed)`` tuple. ``u`` and ``v`` are the endpoints
        in the direction walked; ``key`` identifies parallel edges (it is
        ``None`` for non-multigraphs); ``reversed`` is ``True`` for edges
        walked backwards. For ``source == target`` the list is empty.

    Raises
    ------
    NodeNotFound
        If `source` or `target` is not in `G`.

    NetworkXNoPath
        If `target` cannot be reached from `source`.

    NetworkXNotImplemented
        If `G` is not a directed graph.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge("a", "b")
    >>> G.add_edge("c", "b", reversible=True)
    >>> n, edges = nx.minimum_reversals_path(G, "a", "c")
    >>> n
    1
    >>> for e in edges:
    ...     print(e)
    TraversedEdge(u='a', v='b', key=None, reversed=False)
    TraversedEdge(u='b', v='c', key=None, reversed=True)

    Turning reversibility off makes ``c`` unreachable:

    >>> G["c"]["b"]["reversible"] = False
    >>> nx.minimum_reversals_path(G, "a", "c")
    Traceback (most recent call last):
    ...
    networkx.exception.NetworkXNoPath: No path to c.

    Notes
    -----
    Paths with the same minimum cost are chosen deterministically: nodes are
    expanded in the order in which adjacency insertion first reaches them,
    forward arcs are considered before backward arcs, and within each kind
    edges are considered in their insertion order.

    The graph is never modified.

    See Also
    --------
    dijkstra_path
    """
    if not G.is_directed():
        raise nx.NetworkXNotImplemented(
            "minimum_reversals_path is only defined for directed graphs; "
            "in an undirected graph every edge can already be walked both ways."
        )
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} not found in graph")
    if target not in G:
        raise nx.NodeNotFound(f"Node {target} not found in graph")
    if source == target:
        return 0, []

    # dist[v] = minimum number of reversed edges on a path source -> v.
    dist = {source: 0}
    # pred[v] = (u, key, reversed) for the chosen arc u -> v.
    pred = {}

    # Layered 0-1 BFS: `level` holds nodes at distance d in discovery order,
    # `next_level` collects newly reached nodes at distance d + 1.
    level = deque([source])
    next_level = deque()
    d = 0

    def relax(u, v, key, reversed, cost):
        nd = d + cost
        if v not in dist or nd < dist[v]:
            dist[v] = nd
            pred[v] = (u, key, reversed)
            (level if cost == 0 else next_level).append(v)

    while level:
        u = level.popleft()
        if d > dist[u]:
            # Stale entry from a later distance improvement.
            continue
        # Cost-0 forward arcs first, then cost-1 backward arcs.
        for v, key, data in _outgoing(G, u):
            relax(u, v, key, False, 0)
        for v, key, data in _incoming(G, u):
            if data.get(reversible, False):
                relax(u, v, key, True, 1)
        if not level:
            level, next_level = next_level, level
            d += 1

    if target not in pred:
        raise nx.NetworkXNoPath(f"No path to {target}.")

    # Reconstruct the directed edge sequence from the predecessor chain.
    chain = []
    v = target
    while v != source:
        u, key, was_reversed = pred[v]
        chain.append(TraversedEdge(u, v, key, was_reversed))
        v = u
    chain.reverse()
    return dist[target], chain
