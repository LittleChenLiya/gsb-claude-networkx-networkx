import networkx as nx

__all__ = [
    "RelabelConflict",
    "RelabelPlan",
    "apply_relabel_plan",
    "convert_node_labels_to_integers",
    "plan_relabel_nodes",
    "relabel_nodes",
]


class RelabelConflict(nx.NetworkXError):
    """Raised when a relabel plan is invalid or cannot be applied cleanly.

    A conflict is reported as a batch: all problems found while validating
    a plan or while checking a graph against a plan are collected into a
    single exception, and no partially relabeled graph is returned.
    """


class RelabelPlan:
    """An immutable snapshot plan for batch-relabeling the nodes of a graph.

    A plan records the mapping of old node labels to new labels together
    with a snapshot of the graph's nodes and (keyed) edges at the time the
    plan was generated. Applying the plan checks that the graph is still in
    the same state before any change is made, so a plan can be generated,
    inspected, and applied later without touching the original graph.

    Create plans with :func:`plan_relabel_nodes` and apply them with
    :func:`apply_relabel_plan`.
    """

    __slots__ = ("directed", "edges", "mapping", "multigraph", "nodes")

    def __init__(self, mapping, nodes, edges, multigraph, directed):
        self.mapping = dict(mapping)
        self.nodes = frozenset(nodes)
        self.edges = frozenset(edges)
        self.multigraph = bool(multigraph)
        self.directed = bool(directed)

    def __repr__(self):
        return (
            f"RelabelPlan(mapping={self.mapping!r}, "
            f"nodes={len(self.nodes)}, edges={len(self.edges)})"
        )


def plan_relabel_nodes(G, mapping):
    """Plan a batch relabeling of the nodes of `G`.

    Generates a :class:`RelabelPlan` that records the mapping of old node
    labels to new labels together with a snapshot of the nodes and (keyed)
    edges of `G` at planning time. The graph itself is never modified.

    The mapping must cover every node that is to be renamed (partial
    mappings are allowed) and must not merge two nodes into one: after the
    rename every node must still have a unique label. Circular mappings
    that swap two labels (e.g. ``a->b`` and ``b->a``) are supported, as are
    chains such as ``a->b`` and ``b->c``.

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    mapping : dictionary or callable
       A dictionary with the old labels as keys and new labels as values,
       or a callable mapping an old label to a new label. Every key must be
       a node currently in `G`; keys that are not nodes are reported as a
       conflict.

    Returns
    -------
    plan : RelabelPlan
       A snapshot plan that can be applied with :func:`apply_relabel_plan`.

    Raises
    ------
    RelabelConflict
       If the mapping contains keys that are not nodes of `G`, or if two
       (or more) nodes would be renamed to the same label. All collisions
       are reported together.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> plan = nx.plan_relabel_nodes(G, {0: "a", 1: "b", 2: "c"})
    >>> H = nx.apply_relabel_plan(G, plan)
    >>> sorted(H)
    ['a', 'b', 'c']
    >>> sorted(G)  # the original graph is unchanged
    [0, 1, 2]

    Swapping two labels is supported:

    >>> G = nx.Graph([("a", "b"), ("b", "c")])
    >>> plan = nx.plan_relabel_nodes(G, {"a": "b", "b": "a"})
    >>> H = nx.apply_relabel_plan(G, plan)
    >>> sorted(tuple(sorted(e)) for e in H.edges)
    [('a', 'b'), ('a', 'c')]
    """
    # you can pass any callable e.g. f(old_label) -> new_label, but
    # we'll just make a dictionary here regardless
    if callable(mapping):
        mapping = {n: mapping(n) for n in G}
    mapping = dict(mapping)

    nodes = set(G.nodes)

    # The mapping must cover the nodes to be renamed: every key must name
    # a node that is currently in the graph.
    unknown = sorted((k for k in mapping if k not in nodes), key=repr)
    if unknown:
        raise RelabelConflict(
            f"mapping contains {len(unknown)} node(s) not in the graph: "
            f"{unknown[:5]!r}" + (" ..." if len(unknown) > 5 else "")
        )

    # The mapping must not merge two nodes into one: after the rename every
    # node must still have a unique final label. This also catches renaming
    # a node onto an existing node that is not itself renamed.
    targets = {}
    for n in nodes:
        targets.setdefault(mapping.get(n, n), []).append(n)
    collisions = {t: srcs for t, srcs in targets.items() if len(srcs) > 1}
    if collisions:
        details = ", ".join(
            f"{sorted(srcs, key=repr)!r} -> {t!r}"
            for t, srcs in sorted(collisions.items(), key=lambda kv: repr(kv[0]))
        )
        raise RelabelConflict(
            f"mapping would merge {len(collisions)} target name(s); two nodes "
            f"cannot be renamed to the same label: {details}"
        )

    # Snapshot the nodes and the (keyed) edges so that applying the plan can
    # verify the graph is still in the same state.
    multigraph = G.is_multigraph()
    if multigraph:
        edges = ((u, v, k) for u, v, k in G.edges(keys=True))
    else:
        edges = G.edges()

    return RelabelPlan(mapping, nodes, edges, multigraph, G.is_directed())


def apply_relabel_plan(G, plan):
    """Apply a relabel plan to a copy of `G`, leaving `G` unchanged.

    Before any change is made, the nodes and edges of `G` are checked
    against the snapshot stored in the plan. If nodes or edges were added
    or removed (or the graph type changed), a :class:`RelabelConflict` is
    raised with all differences reported at once. The conflict is rejected
    as a whole: no partially relabeled graph is returned and `G` is never
    modified.

    The relabeling is applied to a copy of `G`. Every node and edge
    attribute dictionary is copied, so modifying the result shares no
    mutable dictionary with the original graph.

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    plan : RelabelPlan
       A plan generated with :func:`plan_relabel_nodes`.

    Returns
    -------
    H : graph
       A new graph with the nodes relabeled according to the plan.

    Raises
    ------
    RelabelConflict
       If `G` is not in the same state as when the plan was generated.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> plan = nx.plan_relabel_nodes(G, {0: "a", 1: "b"})
    >>> H = nx.apply_relabel_plan(G, plan)
    >>> sorted(H, key=str)
    [2, 'a', 'b']
    >>> sorted(G)  # the original graph is unchanged
    [0, 1, 2]

    Adding an edge after planning invalidates the whole batch:

    >>> G = nx.path_graph(3)
    >>> plan = nx.plan_relabel_nodes(G, {0: "a", 1: "b", 2: "c"})
    >>> G.add_edge(0, 2)
    >>> nx.apply_relabel_plan(G, plan)
    Traceback (most recent call last):
        ...
    networkx.relabel.RelabelConflict: cannot apply relabel plan; ...
    """
    if not isinstance(plan, RelabelPlan):
        raise TypeError(
            f"expected a RelabelPlan, got {type(plan).__name__!r}; "
            "create one with plan_relabel_nodes()"
        )

    errors = []

    current_nodes = set(G.nodes)
    if current_nodes != plan.nodes:
        missing = plan.nodes - current_nodes
        added = current_nodes - plan.nodes
        if missing:
            errors.append(
                f"{len(missing)} node(s) missing since the plan was generated: "
                f"{sorted(missing, key=repr)[:5]!r}"
            )
        if added:
            errors.append(
                f"{len(added)} node(s) added since the plan was generated: "
                f"{sorted(added, key=repr)[:5]!r}"
            )

    if G.is_multigraph() != plan.multigraph or G.is_directed() != plan.directed:
        errors.append(
            "graph type changed since the plan was generated "
            f"(plan: multigraph={plan.multigraph}, directed={plan.directed}; "
            f"now: multigraph={G.is_multigraph()}, directed={G.is_directed()})"
        )
    else:
        if plan.multigraph:
            current_edges = set(G.edges(keys=True))
        else:
            current_edges = set(G.edges())
        if current_edges != plan.edges:
            missing = plan.edges - current_edges
            added = current_edges - plan.edges
            if missing:
                errors.append(
                    f"{len(missing)} edge(s) missing since the plan was "
                    f"generated: {sorted(missing, key=repr)[:5]!r}"
                )
            if added:
                errors.append(
                    f"{len(added)} edge(s) added since the plan was generated: "
                    f"{sorted(added, key=repr)[:5]!r}"
                )

    if errors:
        # Reject the whole batch: raise before any copy is built.
        raise RelabelConflict(
            "cannot apply relabel plan; graph changed since the plan was "
            "generated: " + "; ".join(errors)
        )

    # Apply to a copy: relabel_nodes(copy=True) builds a new graph and
    # copies every attribute dictionary, so G is never modified and the
    # result shares no mutable attribute dictionary with the original.
    return relabel_nodes(G, plan.mapping, copy=True)


@nx._dispatchable(
    preserve_all_attrs=True, mutates_input={"not copy": 2}, returns_graph=True
)
def relabel_nodes(G, mapping, copy=True):
    """Relabel the nodes of the graph G according to a given mapping.

    The original node ordering may not be preserved if `copy` is `False` and the
    mapping includes overlap between old and new labels.

    When multiple nodes are mapped to the same target node, attribute
    handling depends on `copy` kwarg. For predictable attribute handling when
    combining nodes, consider using :any:`contracted_nodes`.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    mapping : dictionary
       A dictionary with the old labels as keys and new labels as values.
       A partial mapping is allowed. Mapping 2 nodes to a single node is allowed.
       Any non-node keys in the mapping are ignored.

    copy : bool (optional, default=True)
       If True return a copy, or if False relabel the nodes in place.

    Examples
    --------
    To create a new graph with nodes relabeled according to a given
    dictionary:

    >>> G = nx.path_graph(3)
    >>> sorted(G)
    [0, 1, 2]
    >>> mapping = {0: "a", 1: "b", 2: "c"}
    >>> H = nx.relabel_nodes(G, mapping)
    >>> sorted(H)
    ['a', 'b', 'c']

    Nodes can be relabeled with any hashable object, including numbers
    and strings:

    >>> import string
    >>> G = nx.path_graph(26)  # nodes are integers 0 through 25
    >>> sorted(G)[:3]
    [0, 1, 2]
    >>> mapping = dict(zip(G, string.ascii_lowercase))
    >>> G = nx.relabel_nodes(G, mapping)  # nodes are characters a through z
    >>> sorted(G)[:3]
    ['a', 'b', 'c']
    >>> mapping = dict(zip(G, range(1, 27)))
    >>> G = nx.relabel_nodes(G, mapping)  # nodes are integers 1 through 26
    >>> sorted(G)[:3]
    [1, 2, 3]

    To perform a partial in-place relabeling, provide a dictionary
    mapping only a subset of the nodes, and set the `copy` keyword
    argument to False:

    >>> G = nx.path_graph(3)  # nodes 0-1-2
    >>> mapping = {0: "a", 1: "b"}  # 0->'a' and 1->'b'
    >>> G = nx.relabel_nodes(G, mapping, copy=False)
    >>> sorted(G, key=str)
    [2, 'a', 'b']

    A mapping can also be given as a function:

    >>> G = nx.path_graph(3)
    >>> H = nx.relabel_nodes(G, lambda x: x**2)
    >>> list(H)
    [0, 1, 4]

    In a multigraph, relabeling two or more nodes to the same new node
    will retain all edges, but may change the edge keys in the process:

    >>> G = nx.MultiGraph()
    >>> G.add_edge(0, 1, value="a")  # returns the key for this edge
    0
    >>> G.add_edge(0, 2, value="b")
    0
    >>> G.add_edge(0, 3, value="c")
    0
    >>> mapping = {1: 4, 2: 4, 3: 4}
    >>> H = nx.relabel_nodes(G, mapping, copy=True)
    >>> print(H[0])
    {4: {0: {'value': 'a'}, 1: {'value': 'b'}, 2: {'value': 'c'}}}

    This works for in-place relabeling too:

    >>> G = nx.relabel_nodes(G, mapping, copy=False)
    >>> print(G[0])
    {4: {0: {'value': 'a'}, 1: {'value': 'b'}, 2: {'value': 'c'}}}

    Notes
    -----
    Only the nodes specified in the mapping will be relabeled.
    Any non-node keys in the mapping are ignored.

    The keyword setting copy=False modifies the graph in place.
    Relabel_nodes avoids naming collisions by building a
    directed graph from ``mapping`` which specifies the order of
    relabelings. Naming collisions, such as a->b, b->c, are ordered
    such that "b" gets renamed to "c" before "a" gets renamed "b".
    In cases of circular mappings (e.g. a->b, b->a), modifying the
    graph is not possible in-place and an exception is raised.
    In that case, use copy=True.

    If a relabel operation on a multigraph would cause two or more
    edges to have the same source, target and key, the second edge must
    be assigned a new key to retain all edges. The new key is set
    to the lowest non-negative integer not already used as a key
    for edges between these two nodes. Note that this means non-numeric
    keys may be replaced by numeric keys.

    See Also
    --------
    convert_node_labels_to_integers
    :any:`contracted_nodes`
    """
    # you can pass any callable e.g. f(old_label) -> new_label or
    # e.g. str(old_label) -> new_label, but we'll just make a dictionary here regardless
    m = {n: mapping(n) for n in G} if callable(mapping) else mapping

    if copy:
        return _relabel_copy(G, m)
    else:
        return _relabel_inplace(G, m)


def _relabel_inplace(G, mapping):
    if len(mapping.keys() & mapping.values()) > 0:
        # labels sets overlap
        # can we topological sort and still do the relabeling?
        D = nx.DiGraph(list(mapping.items()))
        D.remove_edges_from(nx.selfloop_edges(D))
        try:
            nodes = reversed(list(nx.topological_sort(D)))
        except nx.NetworkXUnfeasible as err:
            raise nx.NetworkXUnfeasible(
                "The node label sets are overlapping and no ordering can "
                "resolve the mapping. Use copy=True."
            ) from err
    else:
        # non-overlapping label sets, sort them in the order of G nodes
        nodes = [n for n in G if n in mapping]

    multigraph = G.is_multigraph()
    directed = G.is_directed()

    for old in nodes:
        # Test that old is in both mapping and G, otherwise ignore.
        try:
            new = mapping[old]
            G.add_node(new, **G.nodes[old])
        except KeyError:
            continue
        if new == old:
            continue
        if multigraph:
            new_edges = [
                (new, new if old == target else target, key, data)
                for (_, target, key, data) in G.edges(old, data=True, keys=True)
            ]
            if directed:
                new_edges += [
                    (new if old == source else source, new, key, data)
                    for (source, _, key, data) in G.in_edges(old, data=True, keys=True)
                ]
            # Ensure new edges won't overwrite existing ones
            seen = set()
            for i, (source, target, key, data) in enumerate(new_edges):
                if target in G[source] and key in G[source][target]:
                    new_key = 0 if not isinstance(key, int | float) else key
                    while new_key in G[source][target] or (target, new_key) in seen:
                        new_key += 1
                    new_edges[i] = (source, target, new_key, data)
                    seen.add((target, new_key))
        else:
            new_edges = [
                (new, new if old == target else target, data)
                for (_, target, data) in G.edges(old, data=True)
            ]
            if directed:
                new_edges += [
                    (new if old == source else source, new, data)
                    for (source, _, data) in G.in_edges(old, data=True)
                ]
        G.remove_node(old)
        G.add_edges_from(new_edges)
    return G


def _relabel_copy(G, mapping):
    H = G.__class__()
    H.add_nodes_from(mapping.get(n, n) for n in G)
    H._node.update((mapping.get(n, n), d.copy()) for n, d in G.nodes.items())
    if G.is_multigraph():
        new_edges = [
            (mapping.get(n1, n1), mapping.get(n2, n2), k, d.copy())
            for (n1, n2, k, d) in G.edges(keys=True, data=True)
        ]

        # check for conflicting edge-keys
        undirected = not G.is_directed()
        seen_edges = set()
        for i, (source, target, key, data) in enumerate(new_edges):
            while (source, target, key) in seen_edges:
                if not isinstance(key, int | float):
                    key = 0
                key += 1
            seen_edges.add((source, target, key))
            if undirected:
                seen_edges.add((target, source, key))
            new_edges[i] = (source, target, key, data)

        H.add_edges_from(new_edges)
    else:
        H.add_edges_from(
            (mapping.get(n1, n1), mapping.get(n2, n2), d.copy())
            for (n1, n2, d) in G.edges(data=True)
        )
    H.graph.update(G.graph)
    return H


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def convert_node_labels_to_integers(
    G, first_label=0, ordering="default", label_attribute=None
):
    """Returns a copy of the graph G with the nodes relabeled using
    consecutive integers.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    first_label : int, optional (default=0)
       An integer specifying the starting offset in numbering nodes.
       The new integer labels are numbered first_label, ..., n-1+first_label.

    ordering : string
       "default" : inherit node ordering from G.nodes()
       "sorted"  : inherit node ordering from sorted(G.nodes())
       "increasing degree" : nodes are sorted by increasing degree
       "decreasing degree" : nodes are sorted by decreasing degree

    label_attribute : string, optional (default=None)
       Name of node attribute to store old label.  If None no attribute
       is created.

    Notes
    -----
    Node and edge attribute data are copied to the new (relabeled) graph.

    There is no guarantee that the relabeling of nodes to integers will
    give the same two integers for two (even identical graphs).
    Use the `ordering` argument to try to preserve the order.

    See Also
    --------
    relabel_nodes
    """
    N = G.number_of_nodes() + first_label
    if ordering == "default":
        mapping = dict(zip(G.nodes(), range(first_label, N)))
    elif ordering == "sorted":
        nlist = sorted(G.nodes())
        mapping = dict(zip(nlist, range(first_label, N)))
    elif ordering == "increasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    elif ordering == "decreasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        dv_pairs.reverse()
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    else:
        raise nx.NetworkXError(f"Unknown node ordering: {ordering}")
    H = relabel_nodes(G, mapping)
    # create node attribute with the old label
    if label_attribute is not None:
        nx.set_node_attributes(H, {v: k for k, v in mapping.items()}, label_attribute)
    return H
