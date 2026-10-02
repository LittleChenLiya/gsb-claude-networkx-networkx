"""Projection of a multidigraph onto teams defined by a node attribute.

This module provides :func:`team_projection`, which collapses a
:class:`~networkx.MultiDiGraph` whose nodes carry a ``team`` attribute into
a team-level relation graph.  Edges between different teams are grouped by
their (source team, target team) pair and merged by a user-supplied
aggregator, while edges inside a single team are reported separately.
"""

from collections import namedtuple
from copy import deepcopy

import networkx as nx

__all__ = ["TeamProjection", "team_projection"]

#: Result of :func:`team_projection`.
#:
#: ``graph``
#:     A :class:`~networkx.MultiDiGraph` whose nodes are team labels and
#:     whose edges carry the aggregated attributes plus a ``"sources"``
#:     attribute listing the original ``(u, v, key, attrs)`` snapshots.
#: ``internal_edges``
#:     List of ``(u, v, key, attrs)`` snapshots for edges whose endpoints
#:     belong to the same team.  These never reach the aggregator.
TeamProjection = namedtuple("TeamProjection", ["graph", "internal_edges"])


def team_projection(G, aggregate, team_attribute="team"):
    """Project a multidigraph onto teams given by a node attribute.

    Each node of ``G`` is mapped to the team stored in its
    ``team_attribute`` node attribute.  Edges between nodes of different
    teams ("external" edges) are grouped by their ``(source team, target
    team)`` pair, and each group is merged into a single edge of the
    projected graph by ``aggregate``.  Edges whose endpoints share a team
    ("internal" edges) are not aggregated; they are returned separately in
    ``TeamProjection.internal_edges``.

    The input graph is never modified: the aggregator only receives deep
    copies of the edge attribute dicts, and all attribute snapshots stored
    in the result are deep copies as well.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        A directed multigraph.  Every node that takes part in the
        projection must carry the ``team_attribute`` attribute.
    aggregate : callable
        Called once per ``(source team, target team)`` group of external
        edges as ``aggregate(attr_dicts)``, where ``attr_dicts`` is a list
        of deep copies of the edge attribute dicts in the original
        ``(u, v, key)`` iteration order.  It must return a mapping of
        attributes for the aggregated edge.  If it raises, the projection
        aborts and no result is produced; the input graph is left
        untouched.
    team_attribute : str, optional (default="team")
        Name of the node attribute holding the team label.

    Returns
    -------
    TeamProjection
        A named tuple with fields:

        ``graph``
            A new :class:`~networkx.MultiDiGraph` on the team labels.
            Each aggregated edge carries the attributes returned by
            ``aggregate`` plus a ``"sources"`` attribute: the list of
            ``(u, v, key, attrs)`` snapshots of the original parallel
            edges, in their original order.
        ``internal_edges``
            List of ``(u, v, key, attrs)`` snapshots for edges within a
            single team, in their original order.

    Raises
    ------
    networkx.NetworkXNotImplemented
        If ``G`` is undirected.
    networkx.NetworkXError
        If ``G`` is not a multigraph.

    Notes
    -----
    Edges with an endpoint that lacks the ``team_attribute`` attribute do
    not contribute to the projection at all: they appear neither in the
    projected graph nor in the internal edge list.

    The order of ``"sources"`` and of ``internal_edges`` follows the
    iteration order of ``G.edges(keys=True, data=True)``, so results are
    deterministic for a deterministically built input graph.

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_node("a", team="X")
    >>> G.add_node("b", team="X")
    >>> G.add_node("c", team="Y")
    >>> _ = G.add_edge("a", "c", weight=2)
    >>> _ = G.add_edge("b", "c", weight=3)
    >>> _ = G.add_edge("a", "b", weight=10)  # internal to team X
    >>> result = nx.team_projection(
    ...     G, lambda attrs: {"weight": sum(a["weight"] for a in attrs)}
    ... )
    >>> result.graph["X"]["Y"][0]["weight"]
    5
    >>> [(u, v, k) for u, v, k, _ in result.graph["X"]["Y"][0]["sources"]]
    [('a', 'c', 0), ('b', 'c', 0)]
    >>> [(u, v, k) for u, v, k, _ in result.internal_edges]
    [('a', 'b', 0)]
    """
    if not G.is_directed():
        raise nx.NetworkXNotImplemented("team_projection requires a directed graph")
    if not G.is_multigraph():
        raise nx.NetworkXError("team_projection requires a multigraph")

    # Team label per node; nodes without the attribute map to nothing.
    teams = {n: data.get(team_attribute) for n, data in G.nodes(data=True)}

    internal_edges = []
    # group key -> list of (u, v, key, attrs_snapshot); dicts keep
    # first-seen insertion order, giving a stable, deterministic result.
    groups = {}
    for u, v, key, attrs in G.edges(keys=True, data=True):
        team_u = teams.get(u)
        team_v = teams.get(v)
        if team_u is None or team_v is None:
            # Missing mapping entry: this edge produces no result.
            continue
        snapshot = (u, v, key, deepcopy(attrs))
        if team_u == team_v:
            internal_edges.append(snapshot)
        else:
            groups.setdefault((team_u, team_v), []).append(snapshot)

    H = nx.MultiDiGraph()
    for (team_u, team_v), sources in groups.items():
        # The aggregator only ever sees deep copies, so it cannot mutate
        # the input graph (or the snapshots stored on the result).
        merged = dict(aggregate([deepcopy(attrs) for _, _, _, attrs in sources]))
        merged["sources"] = list(sources)
        H.add_edge(team_u, team_v, **merged)

    return TeamProjection(graph=H, internal_edges=internal_edges)
