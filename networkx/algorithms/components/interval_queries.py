"""Offline connected-component queries on interval-annotated graphs."""

from bisect import bisect_left

import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = ["interval_connected_components"]


class _RollbackDisjointSet:
    """Union-find with union-by-size and rollback (no path compression)."""

    __slots__ = ("_history", "parent", "size")

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self._history = []

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            x = parent[x]
        return x

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        self._history.append(b)

    def snapshot(self):
        return len(self._history)

    def rollback(self, snap):
        while len(self._history) > snap:
            b = self._history.pop()
            a = self.parent[b]
            self.size[a] -= self.size[b]
            self.parent[b] = b


def _check_integer(value, what):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an integer, got {value!r}")


def _collect_pair_intervals(G, index, intervals_attr):
    """Validate interval data and return merged active intervals per node pair.

    Returns a pair ``(always, merged)`` where ``always`` is a set of index
    pairs connected at every time (edges without the interval attribute) and
    ``merged`` maps an index pair to its merged, sorted, non-overlapping
    active intervals. Only reads ``G``; never mutates it.
    """
    always = set()
    timed = {}
    for u, v, data in G.edges(data=True):
        if u == v:
            continue  # self-loops cannot change connectivity
        iu, iv = index[u], index[v]
        key = (iu, iv) if iu < iv else (iv, iu)
        if key in always:
            continue
        raw = data.get(intervals_attr)
        if raw is None:
            # An edge without the attribute is always active, so the pair is
            # connected at all times regardless of any parallel edge data.
            always.add(key)
            timed.pop(key, None)
            continue
        if not isinstance(raw, (list, tuple)):
            raise TypeError(
                f"edge ({u!r}, {v!r}) attribute {intervals_attr!r} must be a "
                f"list of (start, end) intervals, got {raw!r}"
            )
        pair_intervals = timed.setdefault(key, [])
        for interval in raw:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(
                    f"edge ({u!r}, {v!r}) has invalid interval {interval!r}; "
                    "intervals must be (start, end) pairs"
                )
            start, end = interval
            _check_integer(start, f"interval start of edge ({u!r}, {v!r})")
            _check_integer(end, f"interval end of edge ({u!r}, {v!r})")
            if start >= end:
                raise ValueError(
                    f"edge ({u!r}, {v!r}) has invalid interval "
                    f"({start}, {end}); start must be strictly less than end"
                )
            pair_intervals.append((start, end))
    merged = {}
    for key, ivs in timed.items():
        if not ivs:  # empty interval list: the pair is never connected
            continue
        ivs.sort()
        merged_ivs = []
        cur_start, cur_end = ivs[0]
        for start, end in ivs[1:]:
            if start <= cur_end:  # overlapping or adjacent half-open intervals
                cur_end = max(cur_end, end)
            else:
                merged_ivs.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged_ivs.append((cur_start, cur_end))
        merged[key] = merged_ivs
    return always, merged


@not_implemented_for("directed")
@nx._dispatchable
def interval_connected_components(G, times, intervals_attr="intervals"):
    """Connected components of an interval graph at multiple query times.

    Each edge of ``G`` may carry an attribute (named by ``intervals_attr``)
    holding a list of integer half-open intervals ``(start, end)`` during
    which the edge is active. Edges without the attribute are always active,
    and an empty list means the edge is never active. This function answers
    all queries offline with a single sweep instead of rebuilding the graph
    and re-running a search for every query time.

    Parameters
    ----------
    G : NetworkX Graph or MultiGraph
        An undirected graph. Self-loops are ignored (they cannot change
        connectivity). For parallel edges of a MultiGraph, two nodes are
        connected at time ``t`` if at least one edge between them is active.
    times : iterable of int
        Query times. Negative times are allowed. The result preserves the
        order of ``times``, including duplicates.
    intervals_attr : str, optional (default="intervals")
        Name of the edge attribute holding the interval list.

    Returns
    -------
    list of list of list
        ``result[i]`` is the partition of the nodes into connected
        components at time ``times[i]``. Each component is a list of nodes
        in node insertion order, and components are ordered by the insertion
        order of their earliest node. Nodes are never compared with each
        other, so nodes of arbitrary, mutually incomparable types are
        supported. Isolated nodes appear as singleton components in every
        answer.

    Raises
    ------
    NetworkXNotImplemented
        If ``G`` is directed.
    TypeError
        If a query time or an interval endpoint is not an integer (``bool``
        counts as invalid), or if the interval attribute is not a list of
        pairs.
    ValueError
        If an interval is not a ``(start, end)`` pair with
        ``start < end``.

    Notes
    -----
    Overlapping or adjacent intervals on the same pair of nodes are merged
    first. The merged intervals are inserted into a segment tree (an
    interval event index) over the sorted unique query times, and the tree
    is traversed with a rollbackable disjoint-set union data structure that
    undoes unions when backtracking. Let ``n`` be the number of nodes,
    ``m`` the number of edges, ``k`` the total number of merged intervals,
    ``q`` the number of query times and ``u`` the number of distinct query
    times. The running time is

    ``O(m + k log k + k log u * alpha(n) + u * n log n)``

    plus the ``O(n * q)`` output itself, and the extra space usage is
    ``O(n + k log u)`` besides the output. Time values are only used as
    sort keys; the algorithm never scans the integer time axis point by
    point, so arbitrarily large or negative times cost nothing extra.
    The input graph and its attributes are never modified.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge("a", "b", intervals=[(0, 4)])
    >>> G.add_edge("b", "c", intervals=[(2, 6)])
    >>> nx.interval_connected_components(G, [4, 1, 2, 2])
    [[['a'], ['b', 'c']], [['a', 'b'], ['c']], [['a', 'b', 'c']], [['a', 'b', 'c']]]

    See Also
    --------
    connected_components
    """
    times = list(times)
    for t in times:
        _check_integer(t, "query time")
    if not times:
        return []

    nodes = list(G)
    index = {node: i for i, node in enumerate(nodes)}
    always, merged = _collect_pair_intervals(G, index, intervals_attr)

    n = len(nodes)
    utimes = sorted(set(times))
    u = len(utimes)

    # Segment tree over the distinct query times; each merged interval is
    # stored at the O(log u) canonical nodes covering its query range.
    # The tree is padded to a power of two so that node indices 2*node and
    # 2*node+1 always stay inside the array; padded leaves hold no queries.
    size = 1 << (u - 1).bit_length()
    tree = [[] for _ in range(2 * size)]

    def add_interval(lo, hi, key):
        lo += size
        hi += size
        while lo < hi:
            if lo & 1:
                tree[lo].append(key)
                lo += 1
            if hi & 1:
                hi -= 1
                tree[hi].append(key)
            lo >>= 1
            hi >>= 1

    for key, ivs in merged.items():
        for start, end in ivs:
            lo = bisect_left(utimes, start)
            hi = bisect_left(utimes, end)
            if lo < hi:
                add_interval(lo, hi, key)

    dsu = _RollbackDisjointSet(n)
    for a, b in always:
        dsu.union(a, b)

    answers = [None] * u

    def sweep(node, lo, hi):
        if lo >= u:  # padded range without query times
            return
        snap = dsu.snapshot()
        for a, b in tree[node]:
            dsu.union(a, b)
        if hi - lo == 1:
            groups = {}
            for x in range(n):
                groups.setdefault(dsu.find(x), []).append(x)
            answers[lo] = [[nodes[x] for x in group] for group in groups.values()]
        else:
            mid = (lo + hi) // 2
            sweep(2 * node, lo, mid)
            sweep(2 * node + 1, mid, hi)
        dsu.rollback(snap)

    sweep(1, 0, size)

    position = {t: i for i, t in enumerate(utimes)}
    return [[list(component) for component in answers[position[t]]] for t in times]
