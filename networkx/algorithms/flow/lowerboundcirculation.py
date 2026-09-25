"""
Circulation with lower and upper bounds on edges.
"""

import operator

import networkx as nx

from .maxflow import default_flow_func

__all__ = ["InfeasibleCirculationError", "lower_bound_circulation"]

_INF = float("inf")


class InfeasibleCirculationError(nx.NetworkXUnfeasible):
    """Raised when no feasible circulation exists.

    The exception carries a machine-checkable certificate of
    infeasibility, not just a message.

    Attributes
    ----------
    cut : frozenset
        The nodes on the source side ``S`` of a cut that violates the
        circulation (Hoffman) condition. Writing ``T`` for the remaining
        nodes of the graph, the certificate satisfies

        ``sum(demand[v] for v in T) - (upper(S, T) - lower(T, S)) == deficit > 0``

        where ``upper(S, T)`` is the total upper capacity of edges from
        ``S`` to ``T`` and ``lower(T, S)`` is the total lower bound of
        edges from ``T`` to ``S``. The net inflow required by ``T``
        exceeds what can cross the cut, so no feasible flow exists.
    deficit : int
        The amount by which the cut condition is violated.
    """

    def __init__(self, cut, deficit, message=None):
        self.cut = frozenset(cut)
        self.deficit = deficit
        if message is None:
            message = (
                "No feasible circulation exists: the net inflow required by "
                f"the complement of S={sorted(self.cut, key=repr)!r} exceeds "
                f"the capacity crossing the cut by {deficit}."
            )
        super().__init__(message)


def _as_integer(value, what):
    if isinstance(value, bool):
        raise nx.NetworkXError(f"{what} must be an integer, got {value!r}.")
    try:
        return int(operator.index(value))
    except TypeError:
        raise nx.NetworkXError(f"{what} must be an integer, got {value!r}.") from None


@nx._dispatchable(node_attrs="demand", edge_attrs={"lower": 0, "upper": _INF})
def lower_bound_circulation(
    G, demand="demand", lower="lower", upper="upper", flow_func=None
):
    """Find a feasible circulation respecting edge lower and upper bounds.

    Given a directed graph with an integer lower bound and upper capacity
    on every edge and an integer demand on every node (positive demand
    means net inflow), find an integer flow ``f`` such that for every edge
    ``lower <= f <= upper`` and for every node ``v``
    ``inflow(v) - outflow(v) == demand[v]``.

    Parameters
    ----------
    G : NetworkX DiGraph or MultiDiGraph
        Edges may carry the attributes ``lower`` and ``upper``; nodes may
        carry the attribute ``demand``. Parallel edges of a MultiDiGraph
        are distinguished by their edge key.

    demand : string (default='demand')
        Name of the node attribute holding the demand of a node. Nodes
        without the attribute have demand 0. Positive demand means the
        node must receive a net inflow. The demands must sum to zero.

    lower : string (default='lower')
        Name of the edge attribute holding the lower bound of an edge.
        Edges without the attribute have lower bound 0.

    upper : string (default='upper')
        Name of the edge attribute holding the upper capacity of an edge.
        Edges without the attribute (or with value None) have unbounded
        capacity.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes
        in a capacitated graph, following the NetworkX residual network
        conventions (see :meth:``maximum_flow``). If None, the default
        (:meth:``preflow_push``) is used.

    Returns
    -------
    flow_dict : dict
        For a DiGraph, a nested dictionary ``flow[u][v]`` with the flow
        on edge (u, v). For a MultiDiGraph, ``flow[u][v][k]`` gives the
        flow on the parallel edge with key ``k``. All nodes of ``G``
        appear as keys of the outer dictionary.

    Raises
    ------
    NetworkXError
        If ``G`` is undirected, if a lower bound exceeds the corresponding
        upper bound, or if a demand, lower bound or upper bound is not an
        integer.

    NetworkXUnfeasible
        If the node demands do not sum to zero.

    InfeasibleCirculationError
        If no feasible circulation exists. The exception carries a cut
        witness (``cut``) and the ``deficit`` by which the cut
        condition is violated; see :class:``InfeasibleCirculationError``.

    Notes
    -----
    The classical reduction to a maximum flow problem is used [1]_.
    Each lower bound is subtracted from its edge and folded into node
    balances ``m(v) = demand[v] - lower_in(v) + lower_out(v)``. A super
    source is linked to every node with ``m(v) < 0`` (capacity ``-m(v)``)
    and every node with ``m(v) > 0`` is linked to a super sink (capacity
    ``m(v)``). A feasible circulation exists exactly when a maximum flow
    saturates all these links; the original edge flows are recovered by
    adding the lower bounds back. Since all finite capacities are
    integers, the resulting flow is integral. When the maximum flow falls
    short, the source side of a minimum cut in the residual network
    yields the cut witness stored on the exception.

    The input graph and its attributes are not modified.

    References
    ----------
    .. [1] Ahuja, R. K., Magnanti, T. L., Orlin, J. B.
       *Network Flows: Theory, Algorithms, and Applications.*
       Prentice Hall, 1993.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge("A", "B", lower=2, upper=5)
    >>> G.nodes["A"]["demand"] = -3
    >>> G.nodes["B"]["demand"] = 3
    >>> nx.lower_bound_circulation(G)
    {'A': {'B': 3}, 'B': {}}

    Tightening the upper bound makes the instance infeasible; the
    exception reports a cut witness and the deficit:

    >>> G["A"]["B"]["upper"] = 2
    >>> try:
    ...     nx.lower_bound_circulation(G)
    ... except nx.InfeasibleCirculationError as exc:
    ...     print("deficit:", exc.deficit, "cut:", sorted(exc.cut))
    deficit: 1 cut: ['A']

    Parallel edges of a MultiDiGraph are reported by key:

    >>> M = nx.MultiDiGraph()
    >>> k0 = M.add_edge("x", "y", lower=1, upper=2)
    >>> k1 = M.add_edge("x", "y", lower=0, upper=2)
    >>> M.nodes["x"]["demand"] = -2
    >>> M.nodes["y"]["demand"] = 2
    >>> nx.lower_bound_circulation(M)
    {'x': {'y': {0: 2, 1: 0}}, 'y': {}}
    """
    if not G.is_directed():
        raise nx.NetworkXError("lower_bound_circulation requires a directed graph.")
    multigraph = G.is_multigraph()

    demands = {}
    for n, data in G.nodes(data=True):
        demands[n] = _as_integer(data.get(demand, 0), f"demand of node {n!r}")
    total = sum(demands.values())
    if total != 0:
        raise nx.NetworkXUnfeasible(
            f"Node demands sum to {total}, not 0: supply and demand are unbalanced."
        )

    if multigraph:
        edge_iter = ((u, v, k, data) for u, v, k, data in G.edges(data=True, keys=True))
    else:
        edge_iter = ((u, v, None, data) for u, v, data in G.edges(data=True))

    # Net inflow that the flow above the lower bounds must still provide.
    balance = dict(demands)
    bundles = {}  # (u, v) -> [(key, lo, hi), ...] for non-self-loop edges
    self_loops = []  # (u, key, lo); self-loops cannot affect conservation
    for u, v, key, data in edge_iter:
        if key is None:
            what = f"edge ({u!r}, {v!r})"
        else:
            what = f"edge ({u!r}, {v!r}, {key!r})"
        lo = _as_integer(data.get(lower, 0), f"lower bound of {what}")
        hi_raw = data.get(upper, None)
        if hi_raw is None or hi_raw == _INF:
            hi = _INF
        else:
            hi = _as_integer(hi_raw, f"upper bound of {what}")
        if lo > hi:
            raise nx.NetworkXError(
                f"lower bound {lo} exceeds upper bound {hi} on {what}."
            )
        if u == v:
            self_loops.append((u, key, lo))
            continue
        balance[u] += lo
        balance[v] -= lo
        bundles.setdefault((u, v), []).append((key, lo, hi))

    source = object()  # super source/sink; cannot collide with graph nodes
    sink = object()
    H = nx.DiGraph()
    H.add_nodes_from(G)
    for (u, v), edges in bundles.items():
        H.add_edge(u, v, capacity=sum(hi - lo for _, lo, hi in edges))
    required = 0
    for v, m in balance.items():
        if m > 0:
            H.add_edge(v, sink, capacity=m)
            required += m
        elif m < 0:
            H.add_edge(source, v, capacity=-m)

    residual = None
    if required > 0:
        func = default_flow_func if flow_func is None else flow_func
        residual = func(H, source, sink, capacity="capacity")
        flow_value = residual.graph["flow_value"]
        if flow_value < required:
            # Nodes reachable from the super source over residual edges
            # with spare capacity form the source side of a minimum cut.
            seen = {source}
            stack = [source]
            succ = residual._succ
            while stack:
                u = stack.pop()
                for v, attr in succ[u].items():
                    if v not in seen and attr["flow"] < attr["capacity"]:
                        seen.add(v)
                        stack.append(v)
            raise InfeasibleCirculationError(seen - {source}, required - flow_value)

    flow = {n: {} for n in G}
    for (u, v), edges in bundles.items():
        # Zero-capacity bundles are absent from the residual network, and
        # antiparallel edge pairs share a single residual edge whose flow
        # may be negative (net flow in the reverse direction); both cases
        # mean zero flow on this bundle (cf. build_flow_dict).
        if residual is not None and residual.has_edge(u, v):
            remaining = max(0, residual[u][v]["flow"])
        else:
            remaining = 0
        for key, lo, hi in edges:
            take = min(hi - lo, remaining)
            remaining -= take
            if multigraph:
                flow[u].setdefault(v, {})[key] = lo + take
            else:
                flow[u][v] = lo + take
    for u, key, lo in self_loops:
        if multigraph:
            flow[u].setdefault(u, {})[key] = lo
        else:
            flow[u][u] = lo
    return flow
