"""Versioned read-only snapshots answering "gate" (unavoidable node) queries.

A *gate* query asks: is node ``gate`` on **every** directed path from a fixed
``start`` node to an ``end`` node?  If not, the index returns a real
counterexample path (one that reaches ``end`` while avoiding ``gate``).

Design notes
------------
* :class:`GateIndex` is built from a ``networkx.DiGraph`` or
  ``networkx.MultiDiGraph`` and captures an immutable snapshot of the graph.
  The source graph is never mutated, and snapshots already handed out keep
  answering queries against the version they were built from.
* Edits are applied explicitly and in batches via :meth:`GateIndex.update`.
  The whole batch is validated first; if any removal targets a missing edge
  (or any other item is invalid) the batch is rejected as a whole, no new
  version is published, and the current snapshot is left untouched.  On
  success a *new* ``GateIndex`` is returned (the index is rebuilt from
  scratch -- small graphs do not need an incremental algorithm).
* Cycles and parallel edges (``MultiDiGraph`` with edge keys) are supported.
* If ``end`` is not reachable from ``start`` the gate is reported as *not*
  unavoidable -- an unreachable target is never misreported as "must pass".
* Queries naming nodes that do not exist in the snapshot raise
  :class:`MissingNodeError`; they never silently succeed.
* A single query's reachability verdict and counterexample path always come
  from the same snapshot version (the one the query is issued against).

Example
-------
>>> import networkx as nx
>>> from gate_index import GateIndex
>>> G = nx.DiGraph([("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")])
>>> v0 = GateIndex(G, "s")
>>> r = v0.query("t", "a")
>>> r.unavoidable
False
>>> r.counterexample.nodes
('s', 'b', 't')
>>> v1 = v0.update(remove_edges=[("b", "t")])
>>> v1.query("t", "a").unavoidable
True
>>> v0.query("t", "a").counterexample.nodes  # old snapshot unchanged
('s', 'b', 't')
"""

from collections import deque
from dataclasses import dataclass
from typing import Hashable, Optional

__all__ = [
    "GateIndex",
    "GateIndexError",
    "MissingNodeError",
    "BatchUpdateError",
    "QueryResult",
    "Path",
]


class GateIndexError(Exception):
    """Base class for all errors raised by :class:`GateIndex`."""


class MissingNodeError(GateIndexError, KeyError):
    """A node referenced by a query or an update is absent from the snapshot."""

    def __init__(self, node: Hashable, version: int, context: str):
        self.node = node
        self.version = version
        super().__init__(
            f"node {node!r} is not present in snapshot version {version} ({context})"
        )


class BatchUpdateError(GateIndexError):
    """A batch of edge edits is invalid; the whole batch was rejected.

    The snapshot the update was issued against is left unchanged and no new
    version is published.
    """


@dataclass(frozen=True)
class Path:
    """A directed path: ordered ``nodes`` plus the concrete ``edges`` used.

    ``edges`` are ``(u, v, key)`` triples; ``key`` is ``None`` for plain
    ``DiGraph`` snapshots and the real edge key for ``MultiDiGraph``
    snapshots, so counterexamples stay unambiguous with parallel edges.
    """

    nodes: tuple
    edges: tuple


@dataclass(frozen=True)
class QueryResult:
    """Outcome of a single gate query, all taken from one snapshot version."""

    version: int
    start: Hashable
    end: Hashable
    gate: Hashable
    reachable: bool
    unavoidable: bool
    counterexample: Optional[Path]


class GateIndex:
    """Immutable, versioned snapshot answering gate (unavoidability) queries.

    Parameters
    ----------
    graph : networkx.DiGraph or networkx.MultiDiGraph
        The graph to snapshot.  It is copied, never mutated.
    start : hashable
        The fixed source node every path is measured from.
    """

    def __init__(self, graph, start, *, _succ=None, _edges=None, _multi=None,
                 _version=0):
        if _succ is None:
            if not graph.is_directed():
                raise GateIndexError(
                    "GateIndex requires a directed graph "
                    "(networkx.DiGraph or networkx.MultiDiGraph)"
                )
            multi = graph.is_multigraph()
            succ = {n: {} for n in graph.nodes}
            edges = set()
            if multi:
                for u, v, k in graph.edges(keys=True):
                    succ[u].setdefault(v, []).append(k)
                    edges.add((u, v, k))
            else:
                for u, v in graph.edges():
                    succ[u].setdefault(v, []).append(None)
                    edges.add((u, v, None))
            _succ, _edges, _multi = succ, frozenset(edges), multi
        if start not in _succ:
            raise MissingNodeError(start, _version, "index construction")
        # Freeze adjacency: neighbour key lists become tuples.
        self._succ = {
            n: {m: tuple(ks) for m, ks in nbrs.items()} for n, nbrs in _succ.items()
        }
        self._edges = frozenset(_edges)
        self._multi = _multi
        self._start = start
        self._version = _version

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def version(self) -> int:
        """Snapshot version; increments with every successful update."""
        return self._version

    @property
    def start(self) -> Hashable:
        return self._start

    @property
    def is_multigraph(self) -> bool:
        return self._multi

    @property
    def nodes(self):
        return self._succ.keys()

    def has_edge(self, u, v, key=None) -> bool:
        if self._multi:
            if key is None:
                return v in self._succ.get(u, {})
            return (u, v, key) in self._edges
        return v in self._succ.get(u, {})

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def _check_node(self, node, context):
        if node not in self._succ:
            raise MissingNodeError(node, self._version, context)

    def _bfs_path(self, end, blocked=None) -> Optional[Path]:
        """Shortest start->end path in this snapshot, optionally skipping a node."""
        prev = {self._start: None}
        queue = deque([self._start])
        while queue:
            u = queue.popleft()
            if u == end:
                break
            for v, keys in self._succ[u].items():
                if v == blocked or v in prev:
                    continue
                prev[v] = (u, keys[0])
                queue.append(v)
        if end not in prev:
            return None
        nodes, edges = [], []
        cur = end
        while prev[cur] is not None:
            u, k = prev[cur]
            nodes.append(cur)
            edges.append((u, cur, k))
            cur = u
        nodes.append(self._start)
        nodes.reverse()
        edges.reverse()
        return Path(tuple(nodes), tuple(edges))

    def query(self, end, gate) -> QueryResult:
        """Ask whether ``gate`` lies on every start->``end`` path.

        Returns a :class:`QueryResult`.  ``unavoidable`` is True only when
        ``end`` is reachable *and* no start->``end`` path avoids ``gate``;
        otherwise ``counterexample`` carries a concrete avoiding path (or
        ``None`` when ``end`` is simply unreachable).
        """
        self._check_node(end, "query end")
        self._check_node(gate, "query gate")
        reachable_path = self._bfs_path(end)
        if reachable_path is None:
            return QueryResult(self._version, self._start, end, gate,
                               reachable=False, unavoidable=False,
                               counterexample=None)
        if gate == self._start or gate == end:
            # Every start->end path trivially visits its own endpoints.
            return QueryResult(self._version, self._start, end, gate,
                               reachable=True, unavoidable=True,
                               counterexample=None)
        avoiding = self._bfs_path(end, blocked=gate)
        return QueryResult(self._version, self._start, end, gate,
                           reachable=True,
                           unavoidable=avoiding is None,
                           counterexample=avoiding)

    def is_unavoidable(self, end, gate) -> bool:
        return self.query(end, gate).unavoidable

    def counterexample(self, end, gate) -> Optional[Path]:
        return self.query(end, gate).counterexample

    # ------------------------------------------------------------------
    # batched updates
    # ------------------------------------------------------------------
    @staticmethod
    def _split_edge(e, what):
        try:
            parts = tuple(e)
        except TypeError:
            raise BatchUpdateError(f"{what} edge {e!r} is not a tuple") from None
        if len(parts) == 2:
            return parts[0], parts[1], None
        if len(parts) == 3:
            return parts
        raise BatchUpdateError(
            f"{what} edge {e!r} must be (u, v) or (u, v, key)"
        )

    @staticmethod
    def _new_key(succ, u, v):
        existing = succ.get(u, {}).get(v, ())
        key = len(existing)
        while key in existing:
            key += 1
        return key

    def update(self, *, add_edges=(), remove_edges=()) -> "GateIndex":
        """Validate a batch of edge edits and publish a new snapshot.

        ``remove_edges`` / ``add_edges`` are iterables of ``(u, v)`` or
        ``(u, v, key)``.  For multigraph snapshots, ``(u, v)`` in a removal
        deletes *all* parallel edges between ``u`` and ``v``; ``(u, v, key)``
        deletes exactly that edge.  For plain digraph snapshots a key must
        not be supplied.

        The entire batch is validated against the current snapshot before
        anything is published: if any removal names an edge that does not
        exist (or any item is malformed), :class:`BatchUpdateError` is
        raised, this snapshot is left unchanged, and no new version appears.
        On success a brand-new :class:`GateIndex` (version + 1) is returned.
        """
        # Work on a private mutable copy; `self` is never touched.
        succ = {n: {m: list(ks) for m, ks in nbrs.items()}
                for n, nbrs in self._succ.items()}
        edges = set(self._edges)

        removals = [self._split_edge(e, "remove") for e in remove_edges]
        additions = [self._split_edge(e, "add") for e in add_edges]

        # --- validate + apply removals on the scratch copy ---
        for u, v, key in removals:
            if u not in succ or v not in succ:
                absent = u if u not in succ else v
                raise BatchUpdateError(
                    f"cannot remove edge ({u!r}, {v!r}): node {absent!r} is "
                    f"absent from snapshot version {self._version}"
                )
            if self._multi:
                if key is None:
                    if v not in succ[u]:
                        raise BatchUpdateError(
                            f"cannot remove edge ({u!r}, {v!r}): no such edge "
                            f"in snapshot version {self._version}"
                        )
                    for k in succ[u].pop(v):
                        edges.discard((u, v, k))
                else:
                    if (u, v, key) not in edges:
                        raise BatchUpdateError(
                            f"cannot remove edge ({u!r}, {v!r}, key={key!r}): "
                            f"no such edge in snapshot version {self._version}"
                        )
                    edges.discard((u, v, key))
                    succ[u][v].remove(key)
                    if not succ[u][v]:
                        del succ[u][v]
            else:
                if key is not None:
                    raise BatchUpdateError(
                        f"edge key {key!r} given for a non-multigraph snapshot"
                    )
                if v not in succ[u]:
                    raise BatchUpdateError(
                        f"cannot remove edge ({u!r}, {v!r}): no such edge "
                        f"in snapshot version {self._version}"
                    )
                del succ[u][v]
                edges.discard((u, v, None))

        # --- validate + apply additions on the scratch copy ---
        for u, v, key in additions:
            if self._multi:
                succ.setdefault(u, {})
                succ.setdefault(v, {})
                if key is None:
                    key = self._new_key(succ, u, v)
                elif key in succ[u].get(v, ()):
                    raise BatchUpdateError(
                        f"cannot add edge ({u!r}, {v!r}, key={key!r}): key "
                        f"already exists in snapshot version {self._version}"
                    )
                succ[u].setdefault(v, []).append(key)
                edges.add((u, v, key))
            else:
                if key is not None:
                    raise BatchUpdateError(
                        f"edge key {key!r} given for a non-multigraph snapshot"
                    )
                succ.setdefault(u, {})
                succ.setdefault(v, {})
                if not succ[u].get(v):
                    succ[u][v] = [None]
                    edges.add((u, v, None))

        # Batch fully validated: publish a new immutable snapshot.
        return GateIndex(
            None, self._start,
            _succ=succ, _edges=edges, _multi=self._multi,
            _version=self._version + 1,
        )

    def __repr__(self):
        return (f"<GateIndex version={self._version} start={self._start!r} "
                f"nodes={len(self._succ)} edges={len(self._edges)} "
                f"multi={self._multi}>")
