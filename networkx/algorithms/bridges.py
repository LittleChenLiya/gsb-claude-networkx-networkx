"""Bridge-finding algorithms."""

import copy
from itertools import chain

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["bridge_contraction", "bridges", "has_bridges", "local_bridges"]


@not_implemented_for("directed")
@nx._dispatchable
def bridges(G, root=None):
    """Generate all bridges in a graph.

    A *bridge* in a graph is an edge whose removal causes the number of
    connected components of the graph to increase.  Equivalently, a bridge is an
    edge that does not belong to any cycle. Bridges are also known as cut-edges,
    isthmuses, or cut arcs.

    Parameters
    ----------
    G : undirected graph

    root : node (optional)
       A node in the graph `G`. If specified, only the bridges in the
       connected component containing this node will be returned.

    Yields
    ------
    e : edge
       An edge in the graph whose removal disconnects the graph (or
       causes the number of connected components to increase).

    Raises
    ------
    NodeNotFound
       If `root` is not in the graph `G`.

    NetworkXNotImplemented
        If `G` is a directed graph.

    Examples
    --------
    The barbell graph with parameter zero has a single bridge:

    >>> G = nx.barbell_graph(10, 0)
    >>> list(nx.bridges(G))
    [(9, 10)]

    Notes
    -----
    This is an implementation of the algorithm described in [1]_.  An edge is a
    bridge if and only if it is not contained in any chain. Chains are found
    using the :func:`networkx.chain_decomposition` function.

    The algorithm described in [1]_ requires a simple graph. If the provided
    graph is a multigraph, we convert it to a simple graph and verify that any
    bridges discovered by the chain decomposition algorithm are not multi-edges.

    Ignoring polylogarithmic factors, the worst-case time complexity is the
    same as the :func:`networkx.chain_decomposition` function,
    $O(m + n)$, where $n$ is the number of nodes in the graph and $m$ is
    the number of edges.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bridge_%28graph_theory%29#Bridge-Finding_with_Chain_Decompositions
    """
    multigraph = G.is_multigraph()
    H = nx.Graph(G) if multigraph else G
    chains = nx.chain_decomposition(H, root=root)
    chain_edges = set(chain.from_iterable(chains))
    if root is not None:
        H = H.subgraph(nx.node_connected_component(H, root)).copy()
    for u, v in H.edges():
        if (u, v) not in chain_edges and (v, u) not in chain_edges:
            if multigraph and len(G[u][v]) > 1:
                continue
            yield u, v


@not_implemented_for("directed")
@nx._dispatchable
def has_bridges(G, root=None):
    """Decide whether a graph has any bridges.

    A *bridge* in a graph is an edge whose removal causes the number of
    connected components of the graph to increase.

    Parameters
    ----------
    G : undirected graph

    root : node (optional)
       A node in the graph `G`. If specified, only the bridges in the
       connected component containing this node will be considered.

    Returns
    -------
    bool
       Whether the graph (or the connected component containing `root`)
       has any bridges.

    Raises
    ------
    NodeNotFound
       If `root` is not in the graph `G`.

    NetworkXNotImplemented
        If `G` is a directed graph.

    Examples
    --------
    The barbell graph with parameter zero has a single bridge::

        >>> G = nx.barbell_graph(10, 0)
        >>> nx.has_bridges(G)
        True

    On the other hand, the cycle graph has no bridges::

        >>> G = nx.cycle_graph(5)
        >>> nx.has_bridges(G)
        False

    Notes
    -----
    This implementation uses the :func:`networkx.bridges` function, so
    it shares its worst-case time complexity, $O(m + n)$, ignoring
    polylogarithmic factors, where $n$ is the number of nodes in the
    graph and $m$ is the number of edges.

    """
    try:
        next(bridges(G, root=root))
    except StopIteration:
        return False
    else:
        return True


@not_implemented_for("multigraph")
@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def local_bridges(G, with_span=True, weight=None):
    """Iterate over local bridges of `G` optionally computing the span

    A *local bridge* is an edge whose endpoints have no common neighbors.
    That is, the edge is not part of a triangle in the graph.

    The *span* of a *local bridge* is the shortest path length between
    the endpoints if the local bridge is removed.

    Parameters
    ----------
    G : undirected graph

    with_span : bool
        If True, yield a 3-tuple `(u, v, span)`

    weight : function, string or None (default: None)
        If function, used to compute edge weights for the span.
        If string, the edge data attribute used in calculating span.
        If None, all edges have weight 1.

    Yields
    ------
    e : edge
        The local bridges as an edge 2-tuple of nodes `(u, v)` or
        as a 3-tuple `(u, v, span)` when `with_span is True`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a directed graph or multigraph.

    Examples
    --------
    A cycle graph has every edge a local bridge with span N-1.

       >>> G = nx.cycle_graph(9)
       >>> (0, 8, 8) in set(nx.local_bridges(G))
       True
    """
    if with_span is not True:
        for u, v in G.edges:
            if not (set(G[u]) & set(G[v])):
                yield u, v
    else:
        wt = nx.weighted._weight_function(G, weight)
        for u, v in G.edges:
            if not (set(G[u]) & set(G[v])):
                enodes = {u, v}

                def hide_edge(n, nbr, d):
                    if n not in enodes or nbr not in enodes:
                        return wt(n, nbr, d)
                    return None

                try:
                    span = nx.shortest_path_length(G, u, v, weight=hide_edge)
                    yield u, v, span
                except nx.NetworkXNoPath:
                    yield u, v, float("inf")


@not_implemented_for("directed")
@nx._dispatchable(returns_graph=True)
def bridge_contraction(G, create_using=None):
    """Returns the bridge contraction of `G`.

    The *bridge contraction* of an undirected graph `G` is the graph
    obtained by contracting each connected component of `G` after all
    bridges have been removed into a single node, and turning each bridge
    of `G` into an edge between the nodes representing the components it
    joins. The bridge contraction is also known as the *bridge tree* (or
    *bridge forest*) of `G`; it is a tree whenever `G` is connected and a
    forest otherwise.

    Each node of the returned graph represents a connected component of
    `G` with all bridges removed. The nodes are labeled by nonnegative
    integers in increasing order according to the first appearance of
    their members in `G.nodes`. Each edge of the returned graph
    represents a bridge of `G`.

    The mapping from each original node to its block is stored in the
    graph attribute ``"mapping"`` of the returned graph. The original
    nodes and non-bridge edges contained in each block are stored as the
    node attributes ``"sources"`` and ``"edges"``, respectively. The
    identity of the original bridge edge corresponding to each block
    edge is stored as the edge attribute ``"source"``.

    Parameters
    ----------
    G : undirected graph
        The graph whose bridge contraction is to be computed.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before
        populated.

    Returns
    -------
    H : NetworkX graph
        The bridge contraction of `G`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed.

    Examples
    --------
    Two triangles joined by a single bridge form two blocks connected by
    one edge::

        >>> G = nx.barbell_graph(3, 0)
        >>> H = nx.bridge_contraction(G)
        >>> H.nodes
        NodeView((0, 1))
        >>> H.edges
        EdgeView([(0, 1)])
        >>> H.nodes[0]["sources"]
        (0, 1, 2)
        >>> H.nodes[1]["sources"]
        (3, 4, 5)
        >>> H.edges[0, 1]["source"]
        (2, 3)

    A pair of parallel edges is not a bridge, so both edges stay in the
    same block::

        >>> G = nx.MultiGraph([(0, 1), (0, 1), (1, 2)])
        >>> H = nx.bridge_contraction(G)
        >>> H.nodes
        NodeView((0, 1))
        >>> H.edges
        EdgeView([(0, 1)])
        >>> H.nodes[0]["edges"]
        ((0, 1, 0), (0, 1, 1))
        >>> H.edges[0, 1]["source"]
        (1, 2, 0)

    An isolated node forms its own block::

        >>> G = nx.Graph([(0, 1)])
        >>> G.add_node(2)
        >>> H = nx.bridge_contraction(G)
        >>> H.nodes
        NodeView((0, 1, 2))
        >>> H.nodes[2]["sources"]
        (2,)

    The node mapping is stored as a graph attribute::

        >>> G = nx.path_graph(4)
        >>> H = nx.bridge_contraction(G)
        >>> H.graph["mapping"]
        {0: 0, 1: 1, 2: 2, 3: 3}

    Notes
    -----
    This function uses :func:`bridges` to find the bridges of `G`, so
    parallel edges in a multigraph are never mistaken for bridges.
    Self-loops are not bridges and remain within their block.

    All attributes copied from `G` onto the returned graph are
    independent copies; mutating the attributes of the returned graph
    does not affect `G` and vice versa.

    See Also
    --------
    bridges : find all bridges in a graph
    bridge_components : generator of 2-edge-connected components
    quotient_graph : quotient graph under an equivalence relation on nodes
    contracted_nodes : contract two nodes of a graph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bridge_%28graph_theory%29#Bridge_tree
    """
    if create_using is None:
        H = nx.Graph()
    else:
        H = nx.empty_graph(0, create_using)

    multigraph = G.is_multigraph()

    # Find all bridges. `nx.bridges` handles multigraphs correctly:
    # parallel edges are never bridges.
    bridge_set = set()
    bridge_edges = []
    for u, v in nx.bridges(G):
        if multigraph:
            # A bridge in a multigraph has no parallel twin, so there is
            # exactly one edge between u and v.
            key = next(iter(G[u][v]))
        else:
            key = None
        bridge_edges.append((u, v, key))
        bridge_set.add((u, v))
        bridge_set.add((v, u))

    # Build the blocks as connected components of G with all bridges
    # removed, using a union-find over the non-bridge edges.
    uf = nx.utils.UnionFind()
    nonbridge_edges = []
    if multigraph:
        all_edges = G.edges(keys=True)
    else:
        all_edges = ((u, v, None) for u, v in G.edges())
    for u, v, key in all_edges:
        if (u, v) not in bridge_set and (v, u) not in bridge_set:
            uf.union(u, v)
            nonbridge_edges.append((u, v, key))

    # Order the blocks by first appearance in G.nodes.
    block_of = {}
    blocks = []
    block_index = {}
    for n in G.nodes:
        root = uf[n]
        if root not in block_index:
            block_index[root] = len(blocks)
            blocks.append([])
        block_of[n] = block_index[root]
        blocks[block_index[root]].append(n)

    # Record the non-bridge edges within each block, in G.edges order.
    block_edges = [[] for _ in blocks]
    for u, v, key in nonbridge_edges:
        b = block_of[u]
        if multigraph:
            block_edges[b].append((u, v, key))
        else:
            block_edges[b].append((u, v))

    # Build the output graph.
    H.add_nodes_from(range(len(blocks)))
    for i, sources in enumerate(blocks):
        H.nodes[i]["sources"] = tuple(sources)
        H.nodes[i]["edges"] = tuple(block_edges[i])

    for u, v, key in bridge_edges:
        b, c = block_of[u], block_of[v]
        H.add_edge(b, c)
        if H.is_multigraph():
            ekey = next(iter(H[b][c]))
            edge_view = H.edges[b, c, ekey]
        else:
            edge_view = H.edges[b, c]
        if multigraph:
            edge_view["source"] = (u, v, key)
            edge_data = G.get_edge_data(u, v, key)
        else:
            edge_view["source"] = (u, v)
            edge_data = G.get_edge_data(u, v)
        for attr, value in edge_data.items():
            edge_view[attr] = copy.deepcopy(value)

    # Graph-level mapping and attributes.
    H.graph["mapping"] = dict(block_of)
    for attr, value in G.graph.items():
        if attr != "mapping":
            H.graph[attr] = copy.deepcopy(value)

    return H
