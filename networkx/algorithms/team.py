"""Projection of multigraphs onto node attributes.

This module provides :func:`team_projection`, which projects a
(multi)digraph whose nodes carry a *team* attribute onto a graph of
team-to-team relationships.  Edges within a team are collected in a
separate list of internal edges, while edges between different teams are
passed to an *aggregator* that summarizes the parallel edges between two
teams into a single aggregated edge.

The original graph is never modified: the aggregator receives a sorted
list of ``(u, v, key, data)`` records where ``data`` is a deep copy of the
original edge attributes, so mutating the records cannot affect the input
graph.
"""

import copy

import networkx as nx

__all__ = ["team_projection"]


def _sum_weights(records, weight="weight"):
    """Default aggregator: sum the ``weight`` attribute of all records.

    Edges without a ``weight`` attribute count as ``1.0``.
    """
    return {weight: sum(data.get(weight, 1.0) for _, _, _, data in records)}


def _sort_key(item):
    """Type-safe sort key for ``(u, v, key)`` records and team pairs.

    Orders by the natural value when types agree, falling back to the type
    name so that keys of mixed types (e.g. ``0`` and ``"k1"``) still have a
    deterministic, stable order.
    """
    if isinstance(item, tuple):
        return tuple(_sort_key(x) for x in item)
    return (type(item).__name__, item)


def team_projection(G, team_attr="team", aggregator=None, weight="weight"):
    r"""Project a multigraph onto a team-level relationship graph.

    Nodes of ``G`` are mapped to teams through their ``team_attr`` node
    attribute.  An edge whose endpoints belong to different teams is an
    *external* edge: all external edges between the same ordered pair of
    teams are grouped and passed to ``aggregator``, whose result becomes
    the attribute dictionary of the single aggregated edge between those
    two teams.  An edge whose endpoints belong to the same team is an
    *internal* edge and is collected, untouched, in the returned list of
    internal edges instead of being aggregated.

    Edges touching a node that lacks the ``team_attr`` attribute (or whose
    value is ``None``) do not produce any result: they are neither
    aggregated nor listed as internal edges.  Likewise, if the aggregator
    raises an exception for a group of external edges, that group produces
    no aggregated edge; other groups are unaffected.

    Every aggregated edge records its provenance: the ``sources`` edge
    attribute holds the list of original ``(u, v, key)`` triples, in stable
    ``(u, v, key)`` order, that were aggregated.  The aggregator receives
    deep copies of the edge attribute dictionaries, so the input graph is
    never modified.

    Parameters
    ----------
    G : NetworkX multigraph
      A NetworkX graph, typically a :class:`~networkx.MultiDiGraph`, whose
      nodes carry a team attribute.  :class:`~networkx.MultiGraph` inputs
      are also accepted; for undirected graphs the two team endpoints are
      sorted so that ``(team_a, team_b)`` and ``(team_b, team_a)`` are
      grouped together.

    team_attr : string, optional (default="team")
      Name of the node attribute holding the team mapping.  Nodes without
      this attribute (or with a ``None`` value) are treated as unmapped.

    aggregator : callable, optional
      A function ``aggregator(records) -> dict`` that summarizes the
      external edges between one pair of teams.  ``records`` is a list of
      ``(u, v, key, data)`` tuples sorted by ``(u, v, key)``, where
      ``data`` is a deep copy of the original edge attribute dictionary.
      The returned dictionary is used as the attributes of the aggregated
      edge (the ``sources`` attribute is added by this function).  If
      ``None``, the default aggregator sums the ``weight`` attribute of
      the records.

    weight : string, optional (default="weight")
      Name of the edge attribute used by the default aggregator.  Ignored
      when a custom ``aggregator`` is given.

    Returns
    -------
    (H, internal_edges) : tuple
      ``H`` is a :class:`~networkx.DiGraph` (or :class:`~networkx.Graph`
      for undirected inputs) whose nodes are the teams and whose edges are
      the aggregated team-to-team relationships.  Every aggregated edge
      carries a ``sources`` attribute with the list of original
      ``(u, v, key)`` triples it represents.  Teams that only have
      internal edges appear as isolated nodes of ``H``.

      ``internal_edges`` is a list of ``(u, v, key, data)`` tuples, sorted
      by ``(u, v, key)``, for edges whose endpoints belong to the same
      team.  ``data`` is a deep copy of the original edge attributes.

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_node("a", team="X")
    >>> G.add_node("b", team="X")
    >>> G.add_node("c", team="Y")
    >>> _ = G.add_edges_from([
    ...     ("a", "c", 0, {"weight": 2}),
    ...     ("b", "c", 0, {"weight": 3}),
    ...     ("a", "b", 0, {"weight": 10}),  # internal edge
    ... ])
    >>> H, internal = nx.team_projection(G)
    >>> H["X"]["Y"]["weight"]
    5
    >>> H["X"]["Y"]["sources"]
    [('a', 'c', 0), ('b', 'c', 0)]
    >>> len(internal)
    1

    Notes
    -----
    The input graph ``G`` is never modified.  The aggregator and the
    internal-edge list only ever see deep copies of the edge attribute
    dictionaries, so even a misbehaving aggregator that mutates its input
    cannot affect the source graph.

    See Also
    --------
    bipartite.projected_graph : one-mode projection of bipartite graphs.
    quotient_graph : collapse a graph according to a node partition.
    """
    if aggregator is None:
        aggregator = lambda records: _sum_weights(records, weight=weight)

    directed = G.is_directed()

    # Group external edges by team pair; collect internal edges separately.
    groups = {}
    internal_edges = []
    for u, v, key, data in G.edges(keys=True, data=True):
        team_u = G.nodes[u].get(team_attr)
        team_v = G.nodes[v].get(team_attr)
        if team_u is None or team_v is None:
            # Missing mapping: this edge produces no result at all.
            continue
        record = (u, v, key, copy.deepcopy(data))
        if team_u == team_v:
            internal_edges.append(record)
            continue
        if directed:
            group_key = (team_u, team_v)
        else:
            # Canonicalize the unordered pair in a type-safe way.
            if _sort_key(team_u) <= _sort_key(team_v):
                group_key = (team_u, team_v)
            else:
                group_key = (team_v, team_u)
        groups.setdefault(group_key, []).append(record)

    # Stable ordering everywhere: records and internal edges are sorted by
    # the original (u, v, key) triple, groups by team pair.
    for records in groups.values():
        records.sort(key=lambda rec: _sort_key(rec[:3]))
    internal_edges.sort(key=lambda rec: _sort_key(rec[:3]))

    H = nx.DiGraph() if directed else nx.Graph()
    # Every team that appears on an edge with both endpoints mapped is a
    # node of the projected graph, even if it only has internal edges.
    teams = set()
    for team_u, team_v in groups:
        teams.add(team_u)
        teams.add(team_v)
    for u, v, _, _ in internal_edges:
        teams.add(G.nodes[u][team_attr])
    H.add_nodes_from(teams)

    for (team_u, team_v), records in sorted(
        groups.items(), key=lambda kv: _sort_key(kv[0])
    ):
        try:
            aggregated = aggregator(records)
        except Exception:
            # Aggregator failed: this group produces no aggregated edge.
            continue
        edge_data = dict(aggregated)
        # Provenance is always recorded by this function, not the
        # aggregator, so parallel edges can never be collapsed away.
        edge_data["sources"] = [(u, v, key) for u, v, key, _ in records]
        H.add_edge(team_u, team_v, **edge_data)

    return H, internal_edges
