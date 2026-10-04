"""Small completion-frontier object for scheduling DAG tasks.

A :class:`CompletionFrontier` is constructed from a *snapshot* of a directed
acyclic graph and keeps only two pieces of scheduling state:

* which nodes have already been completed, and
* which nodes are currently executable (all predecessors completed).

It never executes tasks itself; callers drive it through
:meth:`CompletionFrontier.complete`.

The graph snapshot is copied: adding, removing or mutating nodes/edges on the
original graph after construction has no effect on an existing frontier.
Node iteration order of the supplied graph (insertion order for the standard
NetworkX graph classes) is used as the stable output order everywhere.

Checkpoints
-----------

:meth:`CompletionFrontier.checkpoint` returns a JSON-serializable ``dict``::

    {
        "format": "networkx-completion-frontier",
        "version": 1,
        "fingerprint": {"algorithm": "sha256", "hexdigest": "<hex>"},
        "completed": [0, 3],
    }

* ``completed`` lists completed nodes as **ascending, unique indices** into
  the graph's stable node order (``list(G)``), not as the node values
  themselves, so arbitrary hashable node labels checkpoint without loss.
* ``fingerprint`` binds the state to the exact graph structure. It is a
  SHA-256 digest over the *ordered* node list and the *ordered* edge list
  (each endpoint given by its node index); node and edge attributes are
  ignored. Reordering nodes or edges, adding/removing nodes or edges, or
  pointing the checkpoint at a different graph all change the fingerprint.
* :meth:`CompletionFrontier.restore` refuses a checkpoint whose fingerprint
  does not match the supplied graph, so state from one graph can never be
  applied to another.

Checkpoints are plain data; copy them freely, they share no state with the
frontier that produced them.
"""

from hashlib import sha256

import networkx as nx

__all__ = [
    "CompletionFrontier",
    "CompletionFrontierError",
    "DuplicateNodeIdentityError",
    "GraphStructureError",
    "IllegalCompletionError",
    "CheckpointMismatchError",
]

_CHECKPOINT_FORMAT = "networkx-completion-frontier"
_CHECKPOINT_VERSION = 1
_FINGERPRINT_ALGORITHM = "sha256"


class CompletionFrontierError(nx.NetworkXException):
    """Base class for errors raised by a :class:`CompletionFrontier`."""


class DuplicateNodeIdentityError(CompletionFrontierError):
    """Raised when the node list contains the same identity more than once."""


class GraphStructureError(CompletionFrontierError):
    """Raised when the supplied graph is undirected or contains a cycle."""


class IllegalCompletionError(CompletionFrontierError):
    """Raised when completing an unknown or not-yet-executable node."""


class CheckpointMismatchError(CompletionFrontierError):
    """Raised when a checkpoint is malformed or bound to another graph."""


def _validate(nodes, edges):
    """Return de-duplicated ``(nodes, edges)`` after structural checks.

    Nodes must be hashable and unique, edges must be 2-tuples whose endpoints
    are known nodes. Cycle detection happens in the caller after
    de-duplication.
    """
    clean_nodes = []
    seen_nodes = set()
    for node in nodes:
        try:
            duplicate = node in seen_nodes
        except TypeError as err:
            raise TypeError(f"node is not hashable: {node!r}") from err
        if duplicate:
            raise DuplicateNodeIdentityError(
                f"duplicate node identity: {node!r}"
            )
        seen_nodes.add(node)
        clean_nodes.append(node)

    index = {node: i for i, node in enumerate(clean_nodes)}
    clean_edges = []
    seen_edges = set()
    for edge in edges:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError(f"edge must be a 2-tuple: {edge!r}")
        u, v = edge
        if u not in index or v not in index:
            raise ValueError(f"edge references unknown node: {edge!r}")
        key = (index[u], index[v])
        if key not in seen_edges:
            seen_edges.add(key)
            clean_edges.append(key)
    return clean_nodes, clean_edges


def _assert_acyclic(num_nodes, edges):
    """Kahn's algorithm; raise :class:`GraphStructureError` on a cycle."""
    indegree = [0] * num_nodes
    successors = [[] for _ in range(num_nodes)]
    for u, v in edges:
        indegree[v] += 1
        successors[u].append(v)
    ready = [i for i, d in enumerate(indegree) if d == 0]
    visited = 0
    while ready:
        u = ready.pop()
        visited += 1
        for v in successors[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                ready.append(v)
    if visited != num_nodes:
        raise GraphStructureError("graph must be a directed acyclic graph")


def _compute_fingerprint(num_nodes, edges):
    """SHA-256 over node count and the ordered, index-based edge list.

    Endpoints are identified by their position in the stable node order, so
    the digest is independent of node labels while still distinguishing
    different structures and different node orderings.
    """
    digest = sha256()
    digest.update(f"nodes:{num_nodes}\n".encode("utf-8"))
    for u, v in edges:
        digest.update(f"edge:{u}>{v}\n".encode("utf-8"))
    return digest.hexdigest()


class CompletionFrontier:
    """Track completed nodes and currently-executable nodes of a DAG.

    The frontier does not execute tasks; use :meth:`ready_nodes` to learn what
    may run and :meth:`complete` to report finished work.

    Parameters
    ----------
    nodes : iterable of hashable
        Nodes in stable output order. Identities must be unique.
    edges : iterable of (u, v) tuples
        Directed precedence edges ``u -> v``: ``v`` becomes executable only
        after every predecessor (including ``u``) is completed. Endpoints
        must appear in ``nodes``. The graph must be acyclic.

    Raises
    ------
    DuplicateNodeIdentityError
        If ``nodes`` contains the same identity twice.
    GraphStructureError
        If the resulting graph contains a cycle.
    ValueError
        If an edge is malformed or references an unknown node.

    Examples
    --------
    >>> frontier = CompletionFrontier(["a", "b", "c"], [("a", "c"), ("b", "c")])
    >>> frontier.ready_nodes()
    ['a', 'b']
    >>> frontier.complete("a")
    []
    >>> frontier.complete("b")
    ['c']
    """

    def __init__(self, nodes, edges):
        self._nodes, self._edges = _validate(nodes, edges)
        _assert_acyclic(len(self._nodes), self._edges)

        # Internal adjacency built from the validated snapshot only: the
        # caller never holds a reference to these containers.
        self._successors = [[] for _ in self._nodes]
        self._predecessors = [set() for _ in self._nodes]
        for u, v in self._edges:
            self._successors[u].append(v)
            self._predecessors[v].add(u)
        for succ in self._successors:
            succ.sort()

        self._completed = set()
        self._ready = {
            i for i, preds in enumerate(self._predecessors) if not preds
        }
        self._fingerprint = _compute_fingerprint(len(self._nodes), self._edges)

    @classmethod
    def from_graph(cls, graph):
        """Create a frontier from a directed acyclic NetworkX graph.

        A private snapshot of ``list(graph.nodes)`` and
        ``list(graph.edges)`` is taken, so later external mutations of
        ``graph`` never affect the frontier.

        Parameters
        ----------
        graph : NetworkX graph
            A directed graph (``graph.is_directed()`` must be true). Node
            iteration order is the stable output order.

        Raises
        ------
        GraphStructureError
            If ``graph`` is undirected or contains a cycle.
        DuplicateNodeIdentityError
            If the node list contains duplicate identities.
        """
        if not graph.is_directed():
            raise GraphStructureError("graph must be directed")
        nodes = list(graph.nodes)
        edges = [(u, v) for u, v in graph.edges]
        return cls(nodes, edges)

    @property
    def nodes(self):
        """All nodes in stable order (a fresh list)."""
        return list(self._nodes)

    @property
    def fingerprint(self):
        """Hex SHA-256 digest identifying the bound graph structure."""
        return self._fingerprint

    def ready_nodes(self):
        """Nodes executable right now, in stable order (a fresh list).

        These are exactly the nodes that are not completed and whose every
        predecessor is completed.
        """
        return [self._nodes[i] for i in sorted(self._ready)]

    def completed_nodes(self):
        """Completed nodes in stable order (a fresh list)."""
        return [
            self._nodes[i]
            for i in range(len(self._nodes))
            if i in self._completed
        ]

    def is_ready(self, node):
        """Whether ``node`` is currently executable but not yet completed."""
        self._index(node)
        return node in {self._nodes[i] for i in self._ready}

    def is_completed(self, node):
        """Whether ``node`` has been completed."""
        i = self._index(node)
        return i in self._completed

    def is_finished(self):
        """Whether every node has been completed."""
        return len(self._completed) == len(self._nodes)

    def complete(self, node):
        """Mark ``node`` completed and release newly executable successors.

        ``node`` must currently belong to the executable set. Confirming a
        node that is already completed is a no-op (idempotent). Confirming an
        unknown node or a node whose predecessors are not all completed
        raises :class:`IllegalCompletionError`.

        Parameters
        ----------
        node : node
            The node that has finished executing.

        Returns
        -------
        list of nodes
            The successors that become executable *as a result of this
            confirmation*, in stable order. Empty if no successor was
            released (or the confirmation was an idempotent repeat).
        """
        i = self._index(node)
        if i in self._completed:
            # Repeated confirmation is idempotent: nothing changes and no
            # successor is released again.
            return []
        if i not in self._ready:
            raise IllegalCompletionError(
                f"node is not currently executable: {node!r}"
            )

        self._completed.add(i)
        self._ready.discard(i)

        released = []
        for w in self._successors[i]:
            if (
                w not in self._completed
                and w not in self._ready
                and self._predecessors[w] <= self._completed
            ):
                self._ready.add(w)
                released.append(w)
        return [self._nodes[w] for w in released]

    def checkpoint(self):
        """Export a JSON-serializable checkpoint bound to the graph fingerprint.

        See the module docstring for the format. The returned ``dict`` and
        lists are fresh; mutating them does not affect the frontier.
        """
        return {
            "format": _CHECKPOINT_FORMAT,
            "version": _CHECKPOINT_VERSION,
            "fingerprint": {
                "algorithm": _FINGERPRINT_ALGORITHM,
                "hexdigest": self._fingerprint,
            },
            "completed": sorted(self._completed),
        }

    @classmethod
    def restore(cls, checkpoint, nodes=None, edges=None, graph=None):
        """Rebuild a frontier from ``checkpoint``.

        Exactly one of ``graph`` or ``(nodes, edges)`` must describe the same
        graph structure the checkpoint was taken from. The checkpoint's
        fingerprint must match, otherwise :class:`CheckpointMismatchError` is
        raised and no stale state is applied.

        Parameters
        ----------
        checkpoint : dict
            Object previously returned by :meth:`checkpoint`.
        nodes, edges : iterable, optional
            Stable node list and precedence edges.
        graph : NetworkX graph, optional
            Directed graph whose structure matches the checkpoint.

        Raises
        ------
        CheckpointMismatchError
            If the checkpoint is malformed, has an unsupported version, lists
            invalid completed nodes, is not predecessor-closed, or its
            fingerprint does not match the supplied graph.
        GraphStructureError, DuplicateNodeIdentityError, ValueError
            Propagated from construction when the graph itself is invalid.
        """
        if graph is not None:
            if nodes is not None or edges is not None:
                raise ValueError(
                    "provide either graph or nodes/edges, not both"
                )
            if not graph.is_directed():
                raise GraphStructureError("graph must be directed")
            nodes = list(graph.nodes)
            edges = [(u, v) for u, v in graph.edges]
        elif nodes is None or edges is None:
            raise ValueError("provide graph or both nodes and edges")

        completed = cls._parse_checkpoint(checkpoint)

        frontier = cls(nodes, edges)
        fp = checkpoint.get("fingerprint")
        hexdigest = fp.get("hexdigest") if isinstance(fp, dict) else None
        if hexdigest != frontier._fingerprint:
            raise CheckpointMismatchError(
                "checkpoint fingerprint does not match the supplied graph"
            )

        n = len(frontier._nodes)
        if any(
            (not isinstance(i, int) or isinstance(i, bool)) for i in completed
        ):
            raise CheckpointMismatchError(
                "completed entries must be integer node indices"
            )
        if any(i < 0 or i >= n for i in completed):
            raise CheckpointMismatchError("completed index out of range")
        if len(set(completed)) != len(completed):
            raise CheckpointMismatchError("completed indices must be unique")

        # State must be predecessor-closed, otherwise it could never have
        # been produced by legal complete() calls.
        completed_set = set(completed)
        for i in completed:
            if not frontier._predecessors[i] <= completed_set:
                raise CheckpointMismatchError(
                    "checkpoint completed set is not predecessor-closed"
                )

        frontier._completed = completed_set
        frontier._ready = {
            i
            for i in range(n)
            if i not in completed_set
            and frontier._predecessors[i] <= completed_set
        }
        return frontier

    @staticmethod
    def _parse_checkpoint(checkpoint):
        if not isinstance(checkpoint, dict):
            raise CheckpointMismatchError("checkpoint must be a dict")
        if checkpoint.get("format") != _CHECKPOINT_FORMAT:
            raise CheckpointMismatchError("unknown checkpoint format")
        if checkpoint.get("version") != _CHECKPOINT_VERSION:
            raise CheckpointMismatchError("unsupported checkpoint version")
        fp = checkpoint.get("fingerprint")
        if not isinstance(fp, dict):
            raise CheckpointMismatchError("invalid fingerprint entry")
        if fp.get("algorithm") != _FINGERPRINT_ALGORITHM:
            raise CheckpointMismatchError("unsupported fingerprint algorithm")
        if not isinstance(fp.get("hexdigest"), str):
            raise CheckpointMismatchError("invalid fingerprint hexdigest")
        completed = checkpoint.get("completed")
        if not isinstance(completed, list):
            raise CheckpointMismatchError("completed must be a list")
        return completed

    def copy(self):
        """Return an independent deep copy of the frontier."""
        clone = self.__class__.__new__(self.__class__)
        clone._nodes = list(self._nodes)
        clone._edges = list(self._edges)
        clone._successors = [list(s) for s in self._successors]
        clone._predecessors = [set(p) for p in self._predecessors]
        clone._completed = set(self._completed)
        clone._ready = set(self._ready)
        clone._fingerprint = self._fingerprint
        return clone

    def _index(self, node):
        try:
            return self._nodes.index(node)
        except ValueError:
            raise IllegalCompletionError(f"unknown node: {node!r}") from None

    def __eq__(self, other):
        if not isinstance(other, CompletionFrontier):
            return NotImplemented
        return (
            self._fingerprint == other._fingerprint
            and self._completed == other._completed
        )

    __hash__ = None

    def __repr__(self):
        return (
            f"CompletionFrontier(nodes={len(self._nodes)}, "
            f"completed={len(self._completed)}, ready={len(self._ready)})"
        )
