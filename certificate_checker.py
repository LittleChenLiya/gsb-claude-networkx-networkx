"""Independent shortest-path certificate checker for NetworkX.

This module verifies a *certificate* of shortest distances without running
a shortest-path algorithm itself (no Dijkstra/Bellman-Ford/A* is executed).
A certificate consists of:

  * sources    -- the set of start nodes (distance 0, no predecessor);
  * dist       -- claimed shortest distance for every node
                  (None/inf for unreachable nodes);
  * pred       -- claimed predecessor of every node (None for sources
                  and unreachable nodes);
  * edge_keys  -- for multigraphs, the key of the parallel edge used by
                  each predecessor link (edge identity must be explicit:
                  two parallel edges between the same nodes are not
                  interchangeable).

The checker verifies:

  1. every source has distance 0 and no predecessor;
  2. no edge has negative weight -- negative weights are explicitly
     unsupported (the checker refuses to certify them instead of running
     Bellman-Ford);
  3. reachable nodes have finite distances and predecessors, unreachable
     nodes do not (they are not mixed up as distance 0);
  4. every edge satisfies the relaxation inequality
     dist[v] <= dist[u] + w(u, v) (both directions for undirected
     graphs);
  5. every predecessor link is a real edge with the claimed identity
     (edge_keys for multigraphs) and is *tight*:
     dist[v] == dist[u] + w;
  6. predecessor chains are acyclic (self-loops and longer cycles are
     reported as chains), terminate at a source, and the summed edge
     weights telescope to the claimed distance.

The input graph is never modified.

Run the local tests with::

    python -m pytest test_certificate_checker.py -q -p no:asyncio -p no:aiohttp

(The extra ``-p`` flags disable unrelated third-party pytest plugins that
are installed in this environment and crash during startup.)
"""

from __future__ import annotations

import math
from collections import deque
from numbers import Real

import networkx as nx

__all__ = ["CertificateError", "check_shortest_path_certificate"]

TOL = 1e-9
_INF = math.inf


class CertificateError(ValueError):
    """Raised when a shortest-path certificate fails verification.

    The message identifies the offending edge, predecessor link or
    predecessor chain so the bad certificate can be fixed directly.
    """


def _is_finite_number(value):
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _weight(G, u, v, key, weight_attr):
    """Return the weight of edge (u, v[, key]); missing attribute -> 1."""
    if weight_attr is None:
        return 1
    data = G[u][v] if key is None else G[u][v][key]
    if callable(weight_attr):
        return weight_attr(u, v, data)
    return data.get(weight_attr, 1)


def _edge_id(G, u, v, key):
    if G.is_multigraph():
        return "edge (" + repr(u) + ", " + repr(v) + ", key=" + repr(key) + ")"
    return "edge (" + repr(u) + ", " + repr(v) + ")"


def check_shortest_path_certificate(
    G, sources, dist, pred, weight="weight", edge_keys=None
):
    """Verify a shortest-path certificate for graph G.

    Parameters
    ----------
    G : networkx.Graph
        The graph the certificate is about (directed/undirected,
        multi/simple).  It is never modified.
    sources : node or iterable of nodes
        The start set.  Sources must have distance 0 and no predecessor.
    dist : dict
        Claimed distance for every node; None or inf means unreachable.
    pred : dict
        Claimed predecessor for every node; None for sources and
        unreachable nodes.
    weight : str, callable or None, optional
        Edge weight attribute (NetworkX convention).  None means every
        edge has weight 1.
    edge_keys : dict, optional
        node -> key identifying which parallel edge backs each
        predecessor link in a multigraph.  Required for multigraphs.

    Returns
    -------
    True
        If the certificate is valid.

    Raises
    ------
    TypeError
        If the inputs are not of the expected types.
    CertificateError
        If any certificate condition fails; the message names the specific
        edge, predecessor link or chain.
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a NetworkX graph")
    if isinstance(sources, str):
        sources = [sources]
    else:
        sources = list(sources)
    multigraph = G.is_multigraph()
    directed = G.is_directed()

    # --- validate mappings and source set ---------------------------------
    node_set = set(G.nodes)
    for name, mapping in (("dist", dist), ("pred", pred)):
        if not isinstance(mapping, dict):
            raise TypeError(name + " must be a dict node -> value")
        unknown = [n for n in mapping if n not in node_set]
        if unknown:
            raise CertificateError(
                name + " references nodes not in the graph: " + repr(unknown)
            )
    if edge_keys is not None and not isinstance(edge_keys, dict):
        raise TypeError("edge_keys must be a dict node -> key or None")

    source_set = set()
    for s in sources:
        if s not in node_set:
            raise CertificateError("source " + repr(s) + " is not a node of the graph")
        source_set.add(s)
    if not source_set:
        raise CertificateError("source set is empty")

    # --- sources: distance exactly 0, no predecessor ----------------------
    for s in source_set:
        ds = dist.get(s)
        if not _is_finite_number(ds) or ds != 0:
            raise CertificateError(
                "source " + repr(s) + " must have distance 0, got " + repr(ds)
            )
        if pred.get(s) is not None:
            raise CertificateError(
                "source " + repr(s) + " must have no predecessor, got " + repr(pred[s])
            )

    # --- negative weights: explicitly unsupported --------------------------
    edge_iter = (
        G.edges(keys=True, data=True) if multigraph else G.edges(data=True)
    )
    for item in edge_iter:
        if multigraph:
            u, v, key, _data = item
        else:
            u, v, _data = item
            key = None
        w = _weight(G, u, v, key, weight)
        if not _is_finite_number(w):
            raise CertificateError(
                _edge_id(G, u, v, key) + " has non-numeric weight " + repr(w)
            )
        if w < 0:
            raise CertificateError(
                _edge_id(G, u, v, key) + " has negative weight " + repr(w)
                + "; negative weights are not supported by this checker"
            )

    # --- reachability from sources (plain traversal, no shortest paths) ----
    reachable = set()
    dq = deque(source_set)
    while dq:
        u = dq.popleft()
        if u in reachable:
            continue
        reachable.add(u)
        for v in G.neighbors(u):
            if v not in reachable:
                dq.append(v)

    # --- reachability vs claimed distances/predecessors --------------------
    for v in G.nodes:
        dv = dist.get(v)
        pv = pred.get(v)
        if v in reachable:
            if not _is_finite_number(dv):
                raise CertificateError(
                    "node " + repr(v) + " is reachable from sources but claims "
                    "non-finite distance " + repr(dv)
                )
            if v not in source_set and pv is None:
                raise CertificateError(
                    "node " + repr(v) + " is reachable from sources but has no "
                    "predecessor"
                )
        else:
            if _is_finite_number(dv):
                raise CertificateError(
                    "node " + repr(v) + " is unreachable from sources but claims "
                    "finite distance " + repr(dv) + " (unreachable nodes must not "
                    "be given a distance)"
                )
            if dv is not None and dv != _INF:
                raise CertificateError(
                    "node " + repr(v) + " is unreachable; distance must be None or "
                    "inf, got " + repr(dv)
                )
            if pv is not None:
                raise CertificateError(
                    "node " + repr(v) + " is unreachable but claims predecessor "
                    + repr(pv)
                )

    # --- edge relaxation constraints --------------------------------------
    for item in edge_iter:
        if multigraph:
            u, v, key, _data = item
        else:
            u, v, _data = item
            key = None
        w = _weight(G, u, v, key, weight)
        du, dv = dist.get(u), dist.get(v)
        du_f, dv_f = _is_finite_number(du), _is_finite_number(dv)
        eid = _edge_id(G, u, v, key)
        if du_f and not dv_f:
            raise CertificateError(
                eid + " leaves reachable node " + repr(u) + " but " + repr(v)
                + " claims to be unreachable"
            )
        if du_f and dv_f and dv > du + w + TOL:
            raise CertificateError(
                eid + " violates the relaxation constraint: dist[" + repr(v)
                + "]=" + repr(dv) + " > dist[" + repr(u) + "]+w=" + repr(du + w)
            )
        if not directed:
            if dv_f and not du_f:
                raise CertificateError(
                    eid + " leaves reachable node " + repr(v) + " but " + repr(u)
                    + " claims to be unreachable"
                )
            if dv_f and du_f and du > dv + w + TOL:
                raise CertificateError(
                    "undirected " + eid + " violates the relaxation constraint: "
                    "dist[" + repr(u) + "]=" + repr(du) + " > dist[" + repr(v)
                    + "]+w=" + repr(dv + w)
                )

    # --- predecessor links: identity, existence, tightness -----------------
    for v in reachable:
        if v in source_set:
            continue
        p = pred.get(v)
        if p is None:
            continue  # already reported above
        if p not in node_set:
            raise CertificateError(
                "predecessor " + repr(p) + " of node " + repr(v)
                + " is not a node of the graph"
            )
        if p not in reachable:
            raise CertificateError(
                "predecessor " + repr(p) + " of node " + repr(v)
                + " is not reachable from the sources"
            )
        if multigraph:
            if not isinstance(edge_keys, dict) or v not in edge_keys:
                raise CertificateError(
                    "parallel-edge identity required: node " + repr(v) + " has "
                    "predecessor " + repr(p) + " in a multigraph; provide "
                    "edge_keys[" + repr(v) + "]=key to identify which of the "
                    "parallel edges is used"
                )
            key = edge_keys[v]
            if not G.has_edge(p, v, key):
                raise CertificateError(
                    "predecessor " + _edge_id(G, p, v, key) + " for node "
                    + repr(v) + " does not exist"
                )
        else:
            key = None
            if not G.has_edge(p, v):
                raise CertificateError(
                    "predecessor " + _edge_id(G, p, v, key) + " for node "
                    + repr(v) + " does not exist"
                )
        w = _weight(G, p, v, key, weight)
        expected = dist[p] + w
        if abs(dist[v] - expected) > TOL:
            if dist[v] < expected:
                raise CertificateError(
                    "predecessor " + _edge_id(G, p, v, key) + " for node "
                    + repr(v) + " is not tight: dist[" + repr(v) + "]="
                    + repr(dist[v]) + " but dist[" + repr(p) + "]+w="
                    + repr(expected) + " (the claimed distance is smaller "
                    "than the predecessor edge justifies)"
                )
            raise CertificateError(
                "predecessor " + _edge_id(G, p, v, key) + " for node "
                + repr(v) + " is not tight: dist[" + repr(v) + "]="
                + repr(dist[v]) + " but dist[" + repr(p) + "]+w="
                + repr(expected)
            )

    # --- predecessor chains: acyclic, end at a source, telescope -----------
    for v in reachable:
        if v in source_set:
            continue
        chain = []
        seen = {}
        cur = v
        while cur is not None:
            if cur in seen:
                cyc = chain[seen[cur]:] + [cur]
                if len(cyc) == 2 and cyc[0] == cyc[1]:
                    diag = "self-loop: " + repr(cyc[0]) + " -> " + repr(cyc[0])
                else:
                    diag = "cycle: " + " -> ".join(repr(x) for x in cyc)
                raise CertificateError(
                    "predecessor chain for node " + repr(v) + " contains a "
                    + diag + "; the chain must terminate at a source"
                )
            seen[cur] = len(chain)
            chain.append(cur)
            cur = pred.get(cur)
        if not chain:
            continue
        end = chain[-1]
        if end not in source_set:
            raise CertificateError(
                "predecessor chain for node " + repr(v) + " does not end at a "
                "source: " + " -> ".join(repr(x) for x in chain)
            )
        total = 0
        for i in range(len(chain) - 1):
            child = chain[i]
            parent = chain[i + 1]
            key = edge_keys[child] if multigraph else None
            total += _weight(G, parent, child, key, weight)
        if abs(total - (dist[v] - dist[end])) > TOL:
            raise CertificateError(
                "predecessor chain for node " + repr(v) + " does not telescope: "
                "sum of edge weights " + repr(total) + " != dist[" + repr(v)
                + "]-dist[" + repr(end) + "]=" + repr(dist[v] - dist[end])
                + "; chain: " + " -> ".join(repr(x) for x in chain)
            )

    return True
