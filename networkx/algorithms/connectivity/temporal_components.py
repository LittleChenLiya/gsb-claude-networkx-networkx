"""Offline connected-component queries on graphs with time-limited edges."""

from bisect import bisect_left
from collections.abc import Iterable
from numbers import Integral

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["temporal_connected_components"]


def _validate_time(value, what):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    return int(value)


def _parse_intervals(data, attr, edge_desc):
    """Return a merged list of half-open intervals for one edge (read-only)."""
    if attr not in data:
        return None  # edge is always active
    raw = data[attr]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        raise TypeError(
            f"intervals for edge {edge_desc} must be an iterable of (start, end) pairs"
        )
    intervals = []
    for item in raw:
        try:
            start, end = item
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"interval {item!r} for edge {edge_desc} is not a (start, end) pair"
            ) from err
        start = _validate_time(start, f"interval start for edge {edge_desc}")
        end = _validate_time(end, f"interval end for edge {edge_desc}")
        if start >= end:
            raise ValueError(
                f"interval ({start}, {end}) for edge {edge_desc} is empty or reversed"
            )
        intervals.append((start, end))
    if not intervals:
        return []
    # Merge overlapping/adjacent intervals on a local copy; the graph is untouched.
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            if end > last_end:
                merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


class _RollbackDSU:
    """Union-find with union-by-size and undo, but no path compression."""

    __slots__ = ("history", "parent", "size")

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.history = []

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            x = parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        self.history.append(ry)

    def checkpoint(self):
        return len(self.history)

    def rollback(self, mark):
        while len(self.history) > mark:
            ry = self.history.pop()
            rx = self.parent[ry]
            self.size[rx] -= self.size[ry]
            self.parent[ry] = ry


@not_implemented_for("directed")
@nx._dispatchable
def temporal_connected_components(G, times, *, intervals_attr="intervals"):
    """Connected components of an undirected graph at multiple query times.

    Edges of `G` may carry an attribute (named `intervals` by default)
    holding a list of integer half-open intervals `(start, end)` during
    which the edge is active. Edges without the attribute are always
    active. All query times are answered in a single offline sweep: the
    edge intervals are inserted into a segment tree built over the sorted
    distinct query times, and a depth-first walk of that tree maintains
    the connected components with a rollback union-find. The graph is
    never copied per query and time values are never scanned point by
    point.

    Parameters
    ----------
    G : NetworkX Graph or MultiGraph
        Undirected graph. Directed graphs raise `NetworkXNotImplemented`.
    times : iterable of int
        Query times. May contain duplicates and negative values.
    intervals_attr : str, optional (default="intervals")
        Edge attribute holding the list of active intervals.

    Returns
    -------
    list of list of list
        One entry per query time, in the same order as `times` (duplicates
        included). Each entry is a list of components; every component is
        a list of nodes. Nodes inside a component and the components
        themselves follow the node insertion order of `G`. Isolated
        nodes appear as singleton components. Node objects are never
        compared, so they need not be orderable.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed.
    TypeError
        If a query time or interval endpoint is not an integer (`bool`
        counts as invalid), or if the intervals attribute is not an
        iterable of pairs.
    ValueError
        If an interval is empty or reversed.

    Notes
    -----
    Overlapping intervals of the same edge are merged first; parallel
    edges connect two nodes whenever at least one of them is active;
    self-loops never change the components. The input graph and its
    attributes are left unmodified whether the call succeeds or fails.

    Let `n` be the number of nodes, `m` the total number of edge
    intervals and `q` the number of distinct query times. Excluding the
    output itself, the running time is
    `O((n + m log q) alpha(n))` and the extra space is
    `O(n + m log q)`; nothing depends on the magnitude of the time
    values, so times are never scanned point by point.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge("a", "b", intervals=[(0, 4)])
    >>> G.add_edge("b", "c", intervals=[(2, 6)])
    >>> temporal_connected_components(G, [4, 1, 2, 2])
    [[['a'], ['b', 'c']], [['a', 'b'], ['c']], [['a', 'b', 'c']], [['a', 'b', 'c']]]
    """
    node_index = {}
    for node in G.nodes:
        if node not in node_index:
            node_index[node] = len(node_index)
    nodes = list(node_index)
    n = len(nodes)

    query_times = [_validate_time(t, "query time") for t in times]
    if not query_times:
        return []
    distinct = sorted(set(query_times))
    q = len(distinct)

    # Segment tree over the distinct query times: each edge interval is
    # stored in the O(log q) canonical nodes covering the query times it
    # spans. Buckets are built lazily to keep memory at O(m log q).
    tree = {}

    def add_interval(lo, hi, edge, node=1, seg_lo=0, seg_hi=None):
        if seg_hi is None:
            seg_hi = q
        if lo <= seg_lo and seg_hi <= hi:
            tree.setdefault(node, []).append(edge)
            return
        mid = (seg_lo + seg_hi) // 2
        if lo < mid:
            add_interval(lo, hi, edge, 2 * node, seg_lo, mid)
        if hi > mid:
            add_interval(lo, hi, edge, 2 * node + 1, mid, seg_hi)

    if G.is_multigraph():
        edge_iter = ((u, v, k, data) for u, v, k, data in G.edges(keys=True, data=True))
    else:
        edge_iter = ((u, v, None, data) for u, v, data in G.edges(data=True))

    for u, v, key, data in edge_iter:
        desc = (u, v, key) if key is not None else (u, v)
        intervals = _parse_intervals(data, intervals_attr, desc)
        iu, iv = node_index[u], node_index[v]
        if intervals is None:  # no attribute: always active
            add_interval(0, q, (iu, iv))
        else:
            for start, end in intervals:
                lo = bisect_left(distinct, start)
                hi = bisect_left(distinct, end)
                if lo < hi:
                    add_interval(lo, hi, (iu, iv))

    dsu = _RollbackDSU(n)
    answers = [None] * q

    def walk(node, seg_lo, seg_hi):
        mark = dsu.checkpoint()
        for iu, iv in tree.get(node, ()):
            dsu.union(iu, iv)
        if seg_hi - seg_lo == 1:
            groups = {}
            for idx, obj in enumerate(nodes):
                groups.setdefault(dsu.find(idx), []).append(obj)
            answers[seg_lo] = list(groups.values())
        else:
            mid = (seg_lo + seg_hi) // 2
            walk(2 * node, seg_lo, mid)
            walk(2 * node + 1, mid, seg_hi)
        dsu.rollback(mark)

    walk(1, 0, q)

    position = {t: i for i, t in enumerate(distinct)}
    return [answers[position[t]] for t in query_times]
