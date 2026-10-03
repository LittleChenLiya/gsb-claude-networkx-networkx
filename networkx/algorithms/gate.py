"""
Gate index: necessary-node queries over directed graphs.

A *gate* (or necessary node) for an ordered pair of nodes ``(start, end)``
is a node that every path from ``start`` to ``end`` must pass through.
:class:`GateIndex` builds a read-only :class:`GateSnapshot` of the graph
rooted at ``start`` and answers gate queries against a chosen snapshot
version.  Explicit batch updates (add/remove edges) are validated as a
whole before a new snapshot version is published; the original graph is
never modified and earlier snapshots remain queryable.
"""

from collections import deque
from types import MappingProxyType

from networkx.exception import NetworkXError, NetworkXNotImplemented, NodeNotFound

__all__ = ["GateIndex", "GateResult", "GateSnapshot", "gate_query"]


class GateResult:
    """Result of a gate query against a snapshot.

    Attributes
    ----------
    reachable : bool
        True if ``end`` is reachable from ``start`` in the queried
        snapshot version.
    necessary : bool
        True if every path from ``start`` to ``end`` passes through
        ``gate``.  Always False when ``reachable`` is False.
    path : list or None
        A real counterexample path (node list) from ``start`` to ``end``
        that avoids ``gate``, when ``necessary`` is False and
        ``reachable`` is True; otherwise None.  The path is taken from
        the same snapshot version as the reachability result.
    version : int
        The snapshot version the query was answered against.
    """

    __slots__ = ("necessary", "path", "reachable", "version")

    def __init__(self, reachable, necessary, path, version):
        self.reachable = reachable
        self.necessary = necessary
        self.path = path
        self.version = version

    def __bool__(self):
        return self.necessary

    def __repr__(self):
        return (
            f"GateResult(reachable={self.reachable}, necessary={self.necessary}, "
            f"path={self.path!r}, version={self.version})"
        )


class GateSnapshot:
    """Immutable, read-only adjacency snapshot of a directed graph.

    The snapshot stores a frozen copy of the adjacency structure, so it
    can safely be retained and queried after the index has moved on to a
    newer version.  It never modifies the graph it was built from.
    """

    __slots__ = ("_adj", "multigraph", "version")

    def __init__(self, adj, version, multigraph):
        # adj: {u: frozenset(neighbors)} for graphs,
        #      {u: MappingProxyType({v: frozenset(keys)})} for multigraphs.
        self._adj = adj
        self.version = version
        self.multigraph = multigraph

    def __contains__(self, node):
        return node in self._adj

    def __iter__(self):
        return iter(self._adj)

    def __len__(self):
        return len(self._adj)

    def __repr__(self):
        kind = "MultiDiGraph" if self.multigraph else "DiGraph"
        return f"GateSnapshot(version={self.version}, {kind}, nodes={len(self)})"

    def neighbors(self, node):
        """Return the frozen neighbors of ``node`` in this snapshot."""
        return self._adj[node]

    def has_edge(self, u, v, key=None):
        """Return True if edge ``u->v`` (optionally with ``key``) exists."""
        nbrs = self._adj.get(u)
        if nbrs is None or v not in nbrs:
            return False
        if not self.multigraph or key is None:
            return True
        return key in nbrs[v]


def _freeze(G):
    """Return a frozen adjacency copy of a directed graph."""
    if G.is_multigraph():
        return MappingProxyType(
            {
                u: MappingProxyType({v: frozenset(keys) for v, keys in nbrs.items()})
                for u, nbrs in G.adj.items()
            }
        )
    return MappingProxyType({u: frozenset(nbrs) for u, nbrs in G.adj.items()})


def _freeze_adj(adj, multigraph):
    """Freeze a mutable adjacency dict (see :func:`_thaw`)."""
    if multigraph:
        return MappingProxyType(
            {
                u: MappingProxyType({v: frozenset(keys) for v, keys in nbrs.items()})
                for u, nbrs in adj.items()
            }
        )
    return MappingProxyType({u: frozenset(nbrs) for u, nbrs in adj.items()})


def _thaw(snapshot):
    """Return a mutable adjacency copy of a frozen snapshot."""
    if snapshot.multigraph:
        return {
            u: {v: set(keys) for v, keys in nbrs.items()}
            for u, nbrs in snapshot._adj.items()
        }
    return {u: set(nbrs) for u, nbrs in snapshot._adj.items()}


def _unpack_edge(edge, multigraph):
    """Normalize an edge tuple to ``(u, v, key)`` (key may be None)."""
    if not isinstance(edge, (tuple, list)):
        raise NetworkXError(f"edge must be a tuple (u, v[, key]), got {edge!r}")
    if len(edge) == 2:
        u, v = edge
        key = None
    elif len(edge) == 3:
        u, v, key = edge
    else:
        raise NetworkXError(f"edge must be a 2- or 3-tuple, got {edge!r}")
    if not multigraph and len(edge) == 3:
        raise NetworkXError(
            f"DiGraph edges are (u, v); got {edge!r}. "
            "Use a MultiDiGraph for keyed parallel edges."
        )
    return u, v, key


def _new_edge_key(nbrs):
    """Lowest non-negative integer key not already used (networkx convention)."""
    key = 0
    while key in nbrs:
        key += 1
    return key


def _bfs_avoiding(snapshot, start, end, gate):
    """BFS from ``start`` to ``end`` without touching ``gate``.

    Returns the path as a list of nodes, or None if no such path exists.
    Loops are harmless: the visited set prevents revisiting nodes.
    """
    if start == gate:
        # Every path from start starts at start, which is the gate.
        return None
    parent = {start: None}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if u == end:
            break
        for v in snapshot._adj[u]:
            if v == gate or v in parent:
                continue
            parent[v] = u
            queue.append(v)
            if v == end:
                queue.clear()
                break
    if end not in parent:
        return None
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def _reachable(snapshot, start, end):
    """True if ``end`` is reachable from ``start`` in the snapshot."""
    seen = {start}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if u == end:
            return True
        for v in snapshot._adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return False


class GateIndex:
    """Necessary-node (gate) index with versioned read-only snapshots.

    The index is rooted at a ``start`` node and can answer, for any
    ``end`` and ``gate`` node, whether every ``start`` -> ``end`` path
    passes through ``gate``.  When it does not, a real counterexample
    path is returned.

    Parameters
    ----------
    G : DiGraph or MultiDiGraph
        The directed graph to index.  It is copied into a read-only
        snapshot; the graph itself is never modified.
    start : node
        The start node of all gate queries.

    Raises
    ------
    NetworkXNotImplemented
        If ``G`` is undirected.
    NodeNotFound
        If ``start`` is not in ``G``.

    Examples
    --------
    >>> G = nx.DiGraph([("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")])
    >>> index = GateIndex(G, "s")
    >>> index.query("t", "a").necessary
    False
    >>> index.query("t", "a").path
    ['s', 'b', 't']
    >>> v1 = index.update(remove_edges=[("b", "t")])
    >>> index.query("t", "a").necessary
    True
    >>> index.query("t", "a", version=0).path
    ['s', 'b', 't']
    """

    def __init__(self, G, start):
        if not G.is_directed():
            raise NetworkXNotImplemented(
                "GateIndex is implemented for directed graphs; "
                "use a DiGraph or MultiDiGraph."
            )
        if start not in G:
            raise NodeNotFound(f"start node {start!r} is not in G")
        self._multigraph = G.is_multigraph()
        self._start = start
        snapshot = GateSnapshot(_freeze(G), version=0, multigraph=self._multigraph)
        self._snapshots = [snapshot]
        self._current = 0

    @property
    def start(self):
        """The start node all queries are rooted at."""
        return self._start

    @property
    def current_version(self):
        """Version number of the current snapshot."""
        return self._current

    @property
    def multigraph(self):
        """True if the indexed graph is a MultiDiGraph."""
        return self._multigraph

    def versions(self):
        """Return the list of available snapshot version numbers."""
        return list(range(len(self._snapshots)))

    def snapshot(self, version=None):
        """Return the read-only :class:`GateSnapshot` for ``version``.

        If ``version`` is None (the default), the current snapshot is
        returned.  The returned snapshot is immutable and stays valid
        even after newer versions are published.
        """
        if version is None:
            version = self._current
        if not isinstance(version, int) or not 0 <= version < len(self._snapshots):
            raise NetworkXError(
                f"unknown snapshot version {version!r}; "
                f"available versions: {self.versions()}"
            )
        return self._snapshots[version]

    def query(self, end, gate, version=None):
        """Answer whether ``gate`` is on every path from ``start`` to ``end``.

        Parameters
        ----------
        end : node
        gate : node
        version : int, optional
            Snapshot version to query (default: the current version).

        Returns
        -------
        GateResult
            With ``reachable``, ``necessary``, ``path`` and ``version``
            attributes.  If ``end`` is unreachable from ``start``,
            ``reachable`` is False and ``necessary`` is False (an
            unreachable node is never reported as a gate).

        Raises
        ------
        NodeNotFound
            If ``end`` or ``gate`` is not a node of the queried snapshot.
        """
        snapshot = self.snapshot(version)
        if end not in snapshot:
            raise NodeNotFound(
                f"end node {end!r} is not in snapshot version {snapshot.version}"
            )
        if gate not in snapshot:
            raise NodeNotFound(
                f"gate node {gate!r} is not in snapshot version {snapshot.version}"
            )
        start = self._start

        # First try to reach end while avoiding gate.
        path = _bfs_avoiding(snapshot, start, end, gate)
        if path is not None:
            return GateResult(
                reachable=True, necessary=False, path=path, version=snapshot.version
            )
        # No path avoids gate: either gate is necessary, or end itself is
        # unreachable from start.
        if _reachable(snapshot, start, end):
            return GateResult(
                reachable=True, necessary=True, path=None, version=snapshot.version
            )
        return GateResult(
            reachable=False, necessary=False, path=None, version=snapshot.version
        )

    def update(self, add_edges=None, remove_edges=None):
        """Validate and publish a batch of edge updates.

        The whole batch is validated against the current snapshot first;
        if any edge is invalid (e.g. a removed edge that does not exist),
        the batch is rejected, the current version is left unchanged and
        a :class:`NetworkXError` is raised.  Only after the entire batch
        validates is a new snapshot version published.

        Parameters
        ----------
        add_edges : iterable of (u, v[, key]) tuples, optional
            Edges to add.  Nodes are created as needed.  On a
            MultiDiGraph a missing key is auto-generated.
        remove_edges : iterable of (u, v[, key]) tuples, optional
            Edges to remove.  On a MultiDiGraph, ``(u, v)`` removes all
            parallel edges between ``u`` and ``v`` while ``(u, v, key)``
            removes only that edge.

        Returns
        -------
        version : int
            The new current snapshot version.

        Raises
        ------
        NetworkXError
            If the batch is invalid; no new version is published.
        """
        add_edges = [] if add_edges is None else list(add_edges)
        remove_edges = [] if remove_edges is None else list(remove_edges)
        snapshot = self._snapshots[self._current]

        # ---- validate the whole batch against the current snapshot ----
        errors = []
        norm_remove = []
        for edge in remove_edges:
            try:
                u, v, key = _unpack_edge(edge, self._multigraph)
            except NetworkXError as err:
                errors.append(str(err))
                continue
            if u not in snapshot or not snapshot.has_edge(u, v):
                errors.append(
                    f"cannot remove {edge!r}: edge {u!r}->{v!r} does not exist "
                    f"in snapshot version {snapshot.version}"
                )
            elif self._multigraph and key is not None and not snapshot.has_edge(
                u, v, key
            ):
                errors.append(
                    f"cannot remove {edge!r}: key {key!r} does not exist on edge "
                    f"{u!r}->{v!r} in snapshot version {snapshot.version}"
                )
            norm_remove.append((u, v, key))

        norm_add = []
        for edge in add_edges:
            try:
                u, v, key = _unpack_edge(edge, self._multigraph)
            except NetworkXError as err:
                errors.append(str(err))
                continue
            norm_add.append((u, v, key))

        if errors:
            raise NetworkXError(
                f"rejected batch update for snapshot version {snapshot.version} "
                f"(no version change):\n  " + "\n  ".join(errors)
            )

        # ---- apply to a mutable copy and publish a new snapshot ----
        adj = _thaw(snapshot)
        for u, v, key in norm_remove:
            if self._multigraph:
                if key is None:
                    adj[u].pop(v, None)
                else:
                    keys = adj.get(u, {}).get(v)
                    if keys is not None:
                        keys.discard(key)
                        if not keys:
                            del adj[u][v]
            else:
                adj.get(u, set()).discard(v)
        for u, v, key in norm_add:
            if self._multigraph:
                keys = adj.setdefault(u, {}).setdefault(v, set())
                if key is None:
                    key = _new_edge_key(keys)
                keys.add(key)
            else:
                adj.setdefault(u, set()).add(v)

        new_snapshot = GateSnapshot(
            _freeze_adj(adj, self._multigraph),
            version=len(self._snapshots),
            multigraph=self._multigraph,
        )
        self._snapshots.append(new_snapshot)
        self._current = new_snapshot.version
        return self._current


def gate_query(G, start, end, gate):
    """Answer whether ``gate`` is on every ``start`` -> ``end`` path.

    Convenience wrapper that builds a :class:`GateIndex` for ``G`` and
    immediately queries it.  Retain the index if you need versioned
    snapshots or repeated queries.

    Returns
    -------
    GateResult
    """
    return GateIndex(G, start).query(end, gate)
