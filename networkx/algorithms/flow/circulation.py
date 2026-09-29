"""
Feasible circulation with lower and upper bounds on edge capacities.
"""

import operator
from collections import deque

import networkx as nx
from networkx.utils import not_implemented_for

from .maxflow import default_flow_func, maximum_flow

__all__ = ["InfeasibleCirculation", "lower_bound_circulation"]


class InfeasibleCirculation(nx.NetworkXUnfeasible):
    """Exception raised when no feasible circulation exists.

    Attributes
    ----------
    cut : pair of frozensets of nodes
        Node partition `(S, T)` witnessing infeasibility. The total
        adjusted demand of the nodes in `T` (demand plus lower bounds
        of entering edges minus lower bounds of leaving edges) exceeds
        the total residual capacity (upper bound minus lower bound) of
        the edges from `S` to `T`.
    deficit : int
        Amount by which the adjusted demand of `T` exceeds the
        residual capacity of the cut, i.e. the number of additional
        units of flow that would have to enter `T` for a feasible
        circulation to exist.
    """

    def __init__(self, message=None, *, cut=None, deficit=None):
        super().__init__(message)
        self.cut = cut
        self.deficit = deficit


def _as_int(value, what):
    """Return `value` as an int or raise if it is not an integer."""
    try:
        return operator.index(value)
    except TypeError:
        raise nx.NetworkXError(f"{what} must be an integer, got {value!r}.") from None


@not_implemented_for("undirected")
@nx._dispatchable(node_attrs="demand", edge_attrs={"lower": 0, "upper": None})
def lower_bound_circulation(
    G, demand="demand", lower="lower", upper="upper", flow_func=None
):
    r"""Find a feasible circulation satisfying edge lower/upper bounds.

    Given a directed graph with integer lower and upper bounds on each
    edge and integer demands on the nodes, find an integer flow $f$
    such that for every edge $e$

    .. math::

        lower(e) \leq f(e) \leq upper(e)

    and for every node $v$ the net inflow equals its demand:

    .. math::

        \sum_{u} f(u, v) - \sum_{w} f(v, w) = demand(v).

    A positive demand means the node receives flow (a sink); a negative
    demand means the node supplies flow (a source). Self-loops carry
    their lower bound and do not affect feasibility.

    Parameters
    ----------
    G : NetworkX DiGraph or MultiDiGraph
        Edges are expected to have integer attributes `lower` and
        `upper` (see below). Nodes may have an integer attribute
        `demand`. For a MultiDiGraph, parallel edges are distinguished
        by their edge keys.

    demand : string (default='demand')
        Name of the node attribute holding the demand of each node.
        Nodes without this attribute have demand zero. The demands must
        sum to zero.

    lower : string (default='lower')
        Name of the edge attribute holding the lower bound of each
        edge. Edges without this attribute have lower bound zero.

    upper : string (default='upper')
        Name of the edge attribute holding the upper bound (capacity)
        of each edge. Edges without this attribute are unbounded above.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes
        in a capacitated graph. The function has to accept at least
        three parameters: a Graph or Digraph, a source node, and a
        target node, and return a residual network that follows
        NetworkX conventions (see :meth:`maximum_flow` for details).
        If flow_func is None, the default maximum flow function
        (:meth:`preflow_push`) is used.

    Returns
    -------
    flow_dict : dictionary
        Dictionary of dictionaries (of dictionaries) keyed by nodes
        such that `flow_dict[u][v]` is the flow on edge `(u, v)`,
        or, for a MultiDiGraph, `flow_dict[u][v][key]` is the flow on
        edge `(u, v, key)`. Every edge of `G` appears in the
        dictionary. The flow is integral whenever the bounds and
        demands are integral.

    Raises
    ------
    NetworkXError
        If a bound or demand is not an integer, if some edge has
        `lower > upper`, or if the demands do not sum to zero.

    NetworkXNotImplemented
        If `G` is undirected.

    InfeasibleCirculation
        If no feasible flow exists. The exception carries a verifiable
        cut witness: `exc.cut` is a pair of node sets `(S, T)` and
        `exc.deficit` is a positive integer such that the adjusted
        demand of `T` exceeds the residual capacity of the edges from
        `S` to `T` by exactly `exc.deficit`.

    Notes
    -----
    The lower bounds are eliminated by the standard transformation:
    replacing the flow $f$ by $f' = f - lower$ leaves capacities
    $upper - lower$ and adjusted demands
    $demand'(v) = demand(v) - inLower(v) + outLower(v)$. A feasible
    $f'$ is found with a super-source/super-sink maximum flow
    computation, and the lower bounds are added back to recover $f$.

    By the Gale-Hoffman theorem, the instance is infeasible if and only
    if some set $T$ of nodes has adjusted demand exceeding the residual
    capacity of the edges entering $T$; the minimum cut of the
    auxiliary max-flow network provides such a set together with the
    deficit.

    The input graph and its attributes are not modified.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge("A", "B", lower=2, upper=5)
    >>> G.nodes["A"]["demand"] = -3
    >>> G.nodes["B"]["demand"] = 3
    >>> flow = nx.lower_bound_circulation(G)
    >>> flow["A"]["B"]
    3

    If the upper bound is tightened to 2, no feasible flow exists and
    the exception reports the deficit across the witnessing cut:

    >>> G["A"]["B"]["upper"] = 2
    >>> try:
    ...     nx.lower_bound_circulation(G)
    ... except nx.InfeasibleCirculation as exc:
    ...     print(exc.deficit, exc.cut)
    1 (frozenset({'A'}), frozenset({'B'}))
    """
    if flow_func is None:
        flow_func = default_flow_func

    multigraph = G.is_multigraph()

    # Adjusted demands: demand(v) - in_lower(v) + out_lower(v).
    adjusted = {}
    for v, data in G.nodes(data=True):
        adjusted[v] = _as_int(data.get(demand, 0), f"demand of node {v!r}")
    total = sum(adjusted.values())
    if total != 0:
        raise nx.NetworkXError(
            f"total node demand must be zero; demands sum to {total}."
        )

    # Collect edges, validate bounds, and aggregate residual capacities
    # per ordered pair of distinct endpoints (parallel edges are
    # interchangeable for the max-flow computation).
    edges = []  # (u, v, key, lo, hi) with hi None if unbounded
    pair_capacity = {}
    if multigraph:
        edge_view = G.edges(keys=True, data=True)
    else:
        edge_view = ((u, v, None, d) for u, v, d in G.edges(data=True))
    for u, v, key, data in edge_view:
        lo = _as_int(data.get(lower, 0), f"lower bound of edge {(u, v)!r}")
        hi = data.get(upper, None)
        if hi is not None:
            hi = _as_int(hi, f"upper bound of edge {(u, v)!r}")
            if lo > hi:
                raise nx.NetworkXError(
                    f"edge {(u, v)!r} has lower bound {lo} "
                    f"greater than upper bound {hi}."
                )
        edges.append((u, v, key, lo, hi))
        if u == v:
            continue  # self-loops cancel out of the conservation balance
        adjusted[u] += lo
        adjusted[v] -= lo
        if hi is not None:
            pair_capacity[u, v] = pair_capacity.get((u, v), 0) + hi - lo
        elif (u, v) not in pair_capacity:
            pair_capacity[u, v] = None  # unbounded

    total_demand = sum(d for d in adjusted.values() if d > 0)

    if total_demand == 0:
        # Zero adjusted demands: the zero flow is feasible.
        pair_flow = {pair: 0 for pair in pair_capacity}
    else:
        # Any feasible flow can be assumed to send at most total_demand
        # units across any single edge (circulations can be cancelled),
        # so unbounded edges may be capped at total_demand.
        big_m = total_demand
        s, t = object(), object()
        H = nx.DiGraph()
        H.add_nodes_from(adjusted)
        for v, d in adjusted.items():
            if d > 0:
                H.add_edge(v, t, capacity=d)
            elif d < 0:
                H.add_edge(s, v, capacity=-d)
        for (u, v), cap in pair_capacity.items():
            cap = big_m if cap is None else cap
            if cap > 0:
                H.add_edge(u, v, capacity=cap)
        flow_value, flow_dict = maximum_flow(H, s, t, flow_func=flow_func)
        if flow_value < total_demand:
            cut = _min_cut_partition(H, flow_dict, s, t, G.nodes)
            deficit = total_demand - flow_value
            raise InfeasibleCirculation(
                "no feasible circulation exists: the adjusted demand of "
                f"the node set {sorted(map(repr, cut[1]))} exceeds the "
                "residual capacity of the edges entering it by "
                f"{deficit}.",
                cut=cut,
                deficit=deficit,
            )
        pair_flow = {
            (u, v): flow_dict[u].get(v, 0) - flow_dict[v].get(u, 0)
            for (u, v) in pair_capacity
        }

    # Distribute the aggregated flows back onto the individual edges and
    # add the lower bounds back.
    flow = {u: {} for u in G}
    remaining = dict(pair_flow)
    for u, v, key, lo, hi in edges:
        f = lo
        if u != v:
            # Net flow on the pair may be negative (it then goes the
            # other way); only the positive part is distributed here.
            avail = max(remaining[(u, v)], 0)
            if hi is None:
                take = avail  # unbounded edges absorb the rest
            else:
                take = min(avail, hi - lo)
            f = lo + take
            remaining[(u, v)] -= take
        if multigraph:
            flow[u].setdefault(v, {})[key] = f
        else:
            flow[u][v] = f

    # Re-verify conservation of the recovered flow against the demands.
    balance = dict.fromkeys(G, 0)
    for u, v, key, lo, hi in edges:
        if multigraph:
            f = flow[u][v][key]
        else:
            f = flow[u][v]
        balance[u] -= f
        balance[v] += f
    for v, data in G.nodes(data=True):
        want = _as_int(data.get(demand, 0), f"demand of node {v!r}")
        if balance[v] != want:
            raise nx.NetworkXError(
                "internal error: recovered flow violates conservation "
                f"at node {v!r} (net inflow {balance[v]}, demand {want})."
            )

    return flow


def _min_cut_partition(H, flow_dict, s, t, nodes):
    """Return the (source, sink) side of a minimum s-t cut of H.

    `flow_dict` must hold a maximum s-t flow on `H`. The source
    side is the set of nodes reachable from `s` in the residual
    network. Only nodes of the original graph (given by `nodes`) are
    included in the returned partition.
    """
    seen = {s}
    queue = deque([s])
    while queue:
        u = queue.popleft()
        for v in H.successors(u):
            residual = H[u][v]["capacity"] - (
                flow_dict[u].get(v, 0) - flow_dict[v].get(u, 0)
            )
            if residual > 0 and v not in seen:
                seen.add(v)
                queue.append(v)
        for v in H.predecessors(u):
            backward = flow_dict[v].get(u, 0) - flow_dict[u].get(v, 0)
            if backward > 0 and v not in seen:
                seen.add(v)
                queue.append(v)
    source_side = frozenset(v for v in seen if v in nodes)
    sink_side = frozenset(v for v in nodes if v not in seen)
    return (source_side, sink_side)
