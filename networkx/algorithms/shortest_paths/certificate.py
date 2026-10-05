"""Independent verification of shortest-path certificates.

A shortest-path *certificate* claims, for a graph ``G`` and a non-empty set of
*sources*, the shortest distance from the sources to every (reachable) node
together with a predecessor edge for each reachable non-source node.

Unlike the algorithms in :mod:`networkx.algorithms.shortest_paths`, this module
does **not** run Dijkstra (or any other shortest-path algorithm) to produce an
answer.  It only checks a certificate that was produced elsewhere.  For
nonnegative edge weights a certificate is optimal exactly when:

1. Every edge ``(u, v)`` of weight ``w`` obeys the distance constraint
   ``dist[v] <= dist[u] + w`` (in both directions for undirected graphs),
   treating nodes missing from ``dist`` as unreachable (infinite distance).
2. Each predecessor edge genuinely exists in the graph and, on multigraphs,
   identifies the concrete edge through its key.
3. Each predecessor edge is *tight*, i.e. ``dist[v] == dist[u] + w``.
4. Following predecessors forms an acyclic forest whose every chain ends at a
   source.

Negative edge weights are explicitly unsupported: a single negative edge
raises :exc:`NegativeEdgeError`.  This module never modifies ``G``.

Examples
--------
>>> G = nx.path_graph(3)
>>> dist = {0: 0, 1: 1, 2: 2}
>>> pred = {1: 0, 2: 1}
>>> nx.verify_shortest_path_certificate(G, 0, dist, pred)  # passes

Shrinking a claimed distance below the value forced by its predecessor edge is
reported precisely: the predecessor edge ``(1, 2)`` is no longer tight, so the
certificate cannot describe a real shortest path:

>>> try:
...     nx.verify_shortest_path_certificate(G, 0, {0: 0, 1: 1, 2: 0}, pred)
... except CertificateError as err:
...     print(err.violations[0].kind, err.violations[0].edge)
loose_predecessor (1, 2)
"""

import math
from dataclasses import dataclass

import networkx as nx

__all__ = [
    "CertificateError",
    "NegativeEdgeError",
    "Violation",
    "is_valid_shortest_path_certificate",
    "verify_shortest_path_certificate",
]


class CertificateError(nx.NetworkXError):
    """Raised when a shortest-path certificate fails verification.

    The structured findings are attached as a tuple of :class:`Violation`
    objects in the ``violations`` attribute; each violation names the exact
    edge, node or predecessor chain that is wrong.
    """

    def __init__(self, violations):
        self.violations = tuple(violations)
        n = len(self.violations)
        detail = "; ".join(str(v) for v in self.violations)
        noun = "violation" if n == 1 else "violations"
        super().__init__(
            f"invalid shortest-path certificate ({n} {noun}): {detail}"
        )


class NegativeEdgeError(nx.NetworkXError, ValueError):
    """Raised when the graph contains a negative-weight edge.

    Certificate verification only supports nonnegative edge weights; negative
    edges require an algorithm (e.g. Bellman-Ford) that handles them and are
    not accepted here even when no negative cycle exists.
    """


@dataclass(frozen=True)
class Violation:
    """A single certificate defect.

    Attributes
    ----------
    kind : str
        Machine-readable category: ``"edge_constraint"``,
        ``"missing_predecessor"``, ``"nonexistent_edge"``,
        ``"ambiguous_parallel_edge"``, ``"loose_predecessor"``,
        ``"predecessor_cycle"``, ``"bad_chain_end"``,
        ``"unreachable_has_predecessor"``, ``"invalid_distance"``,
        ``"unreachable_distance"`` or ``"unknown_node"``.
    message : str
        Human-readable description naming the offending edge, node or chain.
    edge : tuple, optional
        Edge identity ``(u, v)`` or ``(u, v, key)`` involved, if any.
    node : node, optional
        The node involved, if any.
    chain : tuple, optional
        The predecessor chain (sequence of nodes) involved, if any.
    """

    kind: str
    message: str
    edge: tuple | None = None
    node: object = None
    chain: tuple | None = None

    def __str__(self):
        return self.message


@nx._dispatchable
def verify_shortest_path_certificate(G, sources, dist, pred=None, weight="weight"):
    """Verify that ``dist``/``pred`` is a valid shortest-path certificate.

    This independently checks the certificate against the graph instead of
    recomputing shortest paths.  See the module docstring for the checks
    performed.

    Parameters
    ----------
    G : NetworkX graph
        The graph the certificate refers to.  It is never modified.
    sources : node or non-empty iterable of nodes
        The source set (a single node is accepted as shorthand).  Every source
        must belong to ``G``.
    dist : dict
        Mapping ``node -> distance``.  Nodes not present in ``dist`` are
        considered *unreachable*: an unreachable node must be omitted rather
        than carrying a finite distance of ``0``.  Distances must be finite,
        real and nonnegative.
    pred : dict, optional
        Mapping from each reachable non-source node ``v`` to its predecessor.
        For ordinary graphs the predecessor is the neighbor node ``u``.  For
        multigraphs it may be the pair ``(u, key)`` identifying which parallel
        edge is used; a bare ``u`` is rejected when ``u`` and ``v`` are joined
        by more than one edge.  Source or unreachable nodes must not carry a
        predecessor entry.  When ``pred`` is ``None`` only the distance
        constraints are verified.
    weight : string or callable, optional (default: ``"weight"``)
        Edge attribute holding the weight, or a function
        ``f(u, v, data) -> number``.  Weights must be finite, real and
        nonnegative; infinite-weight edges are treated as absent.

    Raises
    ------
    CertificateError
        If one or more certificate constraints are violated.  All detected
        violations are reported together in the exception's ``violations``
        attribute, each naming the exact edge, node or chain at fault.
    NegativeEdgeError
        If the graph contains a negative-weight edge.
    NetworkXError
        If ``sources`` is empty or contains a node not in ``G``, or if
        ``dist``/``pred`` are not dictionaries.

    Returns
    -------
    None
        Returned when the certificate is valid.
    """
    violations = _collect_violations(G, sources, dist, pred, weight)
    if violations:
        raise CertificateError(violations)


def is_valid_shortest_path_certificate(
    G, sources, dist, pred=None, weight="weight"
):
    """Return whether a shortest-path certificate verifies.

    Same arguments and checks as :func:`verify_shortest_path_certificate`, but
    returns ``True``/``False`` instead of raising :class:`CertificateError`.
    A negative-weight edge still raises :exc:`NegativeEdgeError`, since that is
    a property of the graph rather than of the certificate.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> nx.is_valid_shortest_path_certificate(
    ...     G, 0, {0: 0, 1: 1, 2: 2}, {1: 0, 2: 1}
    ... )
    True
    >>> nx.is_valid_shortest_path_certificate(
    ...     G, 0, {0: 0, 1: 1, 2: 0}, {1: 0, 2: 1}
    ... )
    False
    """
    try:
        verify_shortest_path_certificate(G, sources, dist, pred, weight)
    except CertificateError:
        return False
    return True


def _edge_weight_function(G, weight):
    """Return a per-edge ``f(u, v, edgedata) -> weight`` function.

    Unlike NetworkX's shortest-path weight helper, this never aggregates
    parallel edges: on a multigraph ``edgedata`` is a *single* edge's
    attribute dict, so every parallel edge is checked with its own weight.
    """
    if callable(weight):
        return weight
    return lambda u, v, data: data.get(weight, 1)


def _normalize_sources(G, sources):
    if sources in G:
        sources = (sources,)
    try:
        source_set = set(sources)
    except TypeError as err:
        raise nx.NetworkXError(
            "sources must be a node or an iterable of nodes"
        ) from err
    if not source_set:
        raise nx.NetworkXError("sources must contain at least one node")
    missing = [s for s in source_set if s not in G]
    if missing:
        raise nx.NetworkXError(
            f"sources not in graph: {sorted(repr(s) for s in missing)}"
        )
    return source_set


def _check_number(value, edge_id):
    """Validate an edge weight, returning its float value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"edge {edge_id}: weight {value!r} is not a real number; "
            "certificate verification requires numeric nonnegative weights"
        )
    fv = float(value)
    if math.isnan(fv):
        raise ValueError(f"edge {edge_id}: weight is NaN")
    if fv < 0:
        raise NegativeEdgeError(
            f"edge {edge_id}: negative weight {value} is not supported by the "
            "shortest-path certificate checker"
        )
    return fv


def _check_distance(value, node):
    """Return ``(finite_value_or_None, violation_or_None)`` for a dist entry."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, Violation(
            "invalid_distance",
            f"node {node!r}: distance {value!r} is not a real number",
            node=node,
        )
    fv = float(value)
    if math.isnan(fv):
        return None, Violation(
            "invalid_distance", f"node {node!r}: distance is NaN", node=node
        )
    if fv < 0:
        return None, Violation(
            "invalid_distance",
            f"node {node!r}: negative distance {value} is not a valid "
            "shortest-path distance",
            node=node,
        )
    if math.isinf(fv):
        return None, Violation(
            "unreachable_distance",
            f"node {node!r}: unreachable nodes must be omitted from dist, not "
            f"given an infinite distance ({value!r})",
            node=node,
        )
    return fv, None


def _fmt(x):
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def _edge_constraint_violation(edge_id, u, v, du, dv, w):
    """Distance-constraint violation for one edge direction, or ``None``.

    ``du``/``dv`` are the finite claimed distances of ``u``/``v``, or
    ``None`` when an endpoint is claimed unreachable.
    """
    if du is not None and dv is None:
        # A finite-weight edge leaves a reachable node: v must be reachable
        # too, with dist[v] <= dist[u] + w.  Omitting v is wrong.
        return Violation(
            "edge_constraint",
            f"edge {edge_id}: {u!r} is reachable with dist={_fmt(du)} but "
            f"{v!r} is claimed unreachable (absent from dist)",
            edge=edge_id,
            node=v,
        )
    if du is None or dv is None:
        # Edge out of an unreachable node (possibly into the reachable set)
        # imposes no constraint; two unreachable endpoints neither.
        return None
    if dv > du + w:
        return Violation(
            "edge_constraint",
            f"edge {edge_id}: distance constraint violated: "
            f"dist[{v!r}]={_fmt(dv)} > dist[{u!r}]+{_fmt(w)}="
            f"{_fmt(du + w)}",
            edge=edge_id,
            node=v,
        )
    return None


def _collect_violations(G, sources, dist, pred, weight):
    sources = _normalize_sources(G, sources)
    if not isinstance(dist, dict):
        raise nx.NetworkXError("dist must be a dict mapping nodes to distances")
    if pred is not None and not isinstance(pred, dict):
        raise nx.NetworkXError("pred must be a dict mapping nodes to predecessors")

    weight_fn = _edge_weight_function(G, weight)
    is_multi = G.is_multigraph()

    # ---- Validate the distance table ------------------------------------
    violations = []
    for n in dist:
        if n not in G:
            violations.append(
                Violation(
                    "unknown_node",
                    f"node {n!r} appears in dist but is not a node of the "
                    "graph",
                    node=n,
                )
            )
    distances = {}
    for n, raw in dist.items():
        if n not in G:
            continue
        value, err = _check_distance(raw, n)
        if err is not None:
            violations.append(err)
        else:
            distances[n] = value
    reachable = set(distances)

    # ---- Scan every edge: negative weights + distance constraints -------
    # adjacency() yields each directed adjacency once, so on undirected graphs
    # an edge appears in both endpoint neighborhoods -- exactly the two
    # directions that must satisfy the constraint.
    for u, nbrs in G.adjacency():
        for v, edgedata in nbrs.items():
            if is_multi:
                for key, data in edgedata.items():
                    edge_id = (u, v, key)
                    w = _check_number(weight_fn(u, v, data), edge_id)
                    if not math.isfinite(w):
                        continue
                    err = _edge_constraint_violation(
                        edge_id, u, v, distances.get(u), distances.get(v), w
                    )
                    if err is not None:
                        violations.append(err)
            else:
                edge_id = (u, v)
                w = _check_number(weight_fn(u, v, edgedata), edge_id)
                if not math.isfinite(w):
                    continue
                err = _edge_constraint_violation(
                    edge_id, u, v, distances.get(u), distances.get(v), w
                )
                if err is not None:
                    violations.append(err)

    # Sources are reachable nodes and must claim distance zero, even when no
    # predecessor table is supplied.
    for s in sources:
        if s not in reachable:
            violations.append(
                Violation(
                    "invalid_distance",
                    f"source {s!r} is missing from dist; sources must claim "
                    "distance 0",
                    node=s,
                )
            )
        elif distances[s] != 0:
            violations.append(
                Violation(
                    "invalid_distance",
                    f"source {s!r} must have distance 0, got {_fmt(distances[s])}",
                    node=s,
                )
            )

    if pred is None:
        return violations

    # ---- Validate pred entry locations -----------------------------------
    # Record the declared predecessor node of every plausible entry so that
    # predecessor chains (including self-loops and cycles) can be diagnosed
    # even when the claimed edge does not exist in the graph.
    raw_parent = {}
    for n, claim in pred.items():
        if n not in G:
            violations.append(
                Violation(
                    "unknown_node",
                    f"node {n!r} appears in pred but is not a node of the "
                    "graph",
                    node=n,
                )
            )
        elif n in sources:
            violations.append(
                Violation(
                    "loose_predecessor",
                    f"node {n!r} is a source and must not have a predecessor "
                    f"(got {claim!r})",
                    node=n,
                )
            )
        elif n not in reachable:
            violations.append(
                Violation(
                    "unreachable_has_predecessor",
                    f"node {n!r} is unreachable (absent from dist) but has a "
                    f"predecessor {claim!r}; unreachable nodes must be omitted, "
                    "not given distance 0",
                    node=n,
                )
            )
        else:
            parent_node = claim[0] if (
                is_multi and isinstance(claim, tuple) and len(claim) == 2
            ) else claim
            raw_parent[n] = parent_node

    # ---- Resolve each reachable non-source predecessor claim -------------
    for n in reachable - sources:
        if n not in pred:
            violations.append(
                Violation(
                    "missing_predecessor",
                    f"reachable node {n!r} has no predecessor entry",
                    node=n,
                )
            )
            continue
        claim = pred[n]

        pu, key, data = _resolve_predecessor_claim(G, n, claim, is_multi)
        if data is None:
            # Here key carries the Violation describing the bad claim.
            violations.append(key)
            continue

        edge_id = (pu, n, key) if is_multi else (pu, n)
        w = _check_number(weight_fn(pu, n, data), edge_id)
        if not math.isfinite(w):
            violations.append(
                Violation(
                    "nonexistent_edge",
                    f"predecessor edge {edge_id} has infinite weight and "
                    "cannot belong to a finite shortest path",
                    edge=edge_id,
                    node=n,
                )
            )
            continue

        if pu not in reachable:
            violations.append(
                Violation(
                    "bad_chain_end",
                    f"predecessor chain of {n!r} reaches {pu!r}, which is "
                    "neither reachable nor a source",
                    edge=edge_id,
                    node=n,
                    chain=(n, pu),
                )
            )
        else:
            dv, du = distances[n], distances[pu]
            if dv != du + w:
                violations.append(
                    Violation(
                        "loose_predecessor",
                        f"predecessor edge {edge_id} is not tight: "
                        f"dist[{n!r}]={_fmt(dv)} != dist[{pu!r}]+"
                        f"{_fmt(w)}={_fmt(du + w)}",
                        edge=edge_id,
                        node=n,
                    )
                )

    # ---- Acyclic predecessor chains ending at a source -------------------
    # Chains are walked from the *declared* predecessor table, independently
    # of whether the claimed edges exist, so a self-loop such as
    # ``pred[v] = v`` is always diagnosed as a cycle.  Each cycle is reported
    # once, using the smallest (by repr) reachable node on it.
    state = {}  # node -> 1 finished clean, 2 in a predecessor cycle
    for start in sorted(reachable, key=repr):
        if start in state:
            continue
        order = []
        cur = start
        end_kind = None
        while True:
            if cur in sources:
                end_kind = "ok"
                break
            st = state.get(cur)
            if st is not None:
                # Reached a chain processed in an earlier walk; ``cur`` itself
                # is not part of this walk's ``order``.
                end_kind = "cached-cycle" if st == 2 else "cached-ok"
                break
            if cur in order:
                end_kind = "cycle"
                break
            if cur not in raw_parent:
                end_kind = "ok"  # bad/missing predecessor diagnosed separately
                break
            order.append(cur)
            cur = raw_parent[cur]

        if end_kind == "cycle":
            i = order.index(cur)
            cyc_nodes = order[i:]
            anchor = min(cyc_nodes, key=repr)
            k = cyc_nodes.index(anchor)
            cyc = tuple(cyc_nodes[k:] + cyc_nodes[:k] + [anchor])
            violations.append(
                Violation(
                    "predecessor_cycle",
                    f"predecessor chain contains a cycle: "
                    + " -> ".join(repr(x) for x in cyc),
                    node=anchor,
                    chain=cyc,
                )
            )
            walk_state = 2
        elif end_kind == "cached-cycle":
            walk_state = 2
        else:
            walk_state = 1
        for node in order:
            state[node] = walk_state

    return violations


def _resolve_predecessor_claim(G, n, claim, is_multi):
    """Resolve ``pred[n]`` to ``(predecessor, key, edgedata)``.

    On failure returns ``(predecessor_or_None, violation, None)``.
    """
    if is_multi:
        if isinstance(claim, tuple) and len(claim) == 2:
            pu, key = claim
            if pu not in G or n not in G[pu] or key not in G[pu][n]:
                return (
                    pu if pu in G else None,
                    Violation(
                        "nonexistent_edge",
                        f"predecessor edge ({pu!r}, {n!r}, key={key!r}) does "
                        "not exist in the graph",
                        edge=(pu, n, key),
                        node=n,
                    ),
                    None,
                )
            return pu, key, G[pu][n][key]
        pu = claim
        if pu not in G or n not in G[pu]:
            return (
                pu if pu in G else None,
                Violation(
                    "nonexistent_edge",
                    f"predecessor edge ({pu!r}, {n!r}) does not exist in the "
                    "graph",
                    edge=(pu, n),
                    node=n,
                ),
                None,
            )
        keys = list(G[pu][n])
        if len(keys) != 1:
            return (
                pu,
                Violation(
                    "ambiguous_parallel_edge",
                    f"predecessor of {n!r} given as bare node {pu!r} but "
                    f"{len(keys)} parallel edges (keys {keys!r}) join them; "
                    "identify the edge with a (node, key) pair",
                    node=n,
                ),
                None,
            )
        key = keys[0]
        return pu, key, G[pu][n][key]

    pu = claim
    if pu not in G or n not in G[pu]:
        return (
            pu if pu in G else None,
            Violation(
                "nonexistent_edge",
                f"predecessor edge ({pu!r}, {n!r}) does not exist in the graph",
                edge=(pu, n),
                node=n,
            ),
            None,
        )
    return pu, None, G[pu][n]

