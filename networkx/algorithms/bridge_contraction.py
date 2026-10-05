"""Bridge contraction view of an undirected graph.

Contracting every bridge-free connected component (a *block*, i.e. a
2-edge-connected component) to a single node turns any undirected graph
into a forest: the *bridge tree* (or bridge forest) of the graph.  This
module provides :func:`bridge_contraction`, which computes that
contraction while keeping a full mapping between the original nodes and
edges and the contracted blocks and inter-block edges.
"""

from copy import deepcopy

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["bridge_contraction", "BridgeContraction"]


class BridgeContraction:
    """Read-only view of a graph with all bridge-free blocks contracted.

    Instances are created by :func:`bridge_contraction`; do not construct
    them directly.

    Attributes
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        The contracted graph.  Nodes are integer block ids ``0 .. k-1``
        assigned in the order the blocks' member nodes first appear in
        the input graph.  Each block node carries a ``members`` attribute
        (a frozenset of the original nodes).  Each edge corresponds to a
        bridge of the original graph and carries a ``bridge`` attribute
        with the original edge identity -- ``(u, v)`` for simple graphs
        and ``(u, v, key)`` for multigraphs -- together with an
        independent (deep) copy of the original edge's attributes.  The
        graph-level attributes are an independent copy of the input
        graph's attributes.
    """

    def __init__(self, G, graph, block_of, blocks, bridges):
        self._source = G
        self.graph = graph
        self._block_of = block_of
        self._blocks = blocks
        self._bridges = bridges
        self._bridge_set = frozenset(
            (u, v) if key is None else (u, v, key) for u, v, key in bridges
        )

    def __len__(self):
        """Number of blocks."""
        return len(self._blocks)

    def __iter__(self):
        """Iterate over block ids in their fixed order."""
        return iter(range(len(self._blocks)))

    def blocks(self):
        """Return the list of blocks; each block is a frozenset of original nodes.

        Blocks are ordered by the first appearance of their member nodes
        in the input graph's node iteration order.
        """
        return list(self._blocks)

    def block_of(self, node):
        """Return the id of the block containing ``node``.

        Raises
        ------
        NetworkXError
            If ``node`` is not in the original graph.
        """
        try:
            return self._block_of[node]
        except KeyError:
            raise nx.NetworkXError(f"Node {node!r} is not in the graph.") from None

    def members(self, block):
        """Return the original nodes (a frozenset) contracted into ``block``."""
        return self._blocks[block]

    def bridges(self):
        """Return the bridges of the original graph as identity tuples.

        Each bridge is ``(u, v)`` for simple graphs and ``(u, v, key)``
        for multigraphs, in the fixed block/edge order of the contracted
        graph.
        """
        return [
            (u, v) if key is None else (u, v, key) for u, v, key in self._bridges
        ]

    def is_bridge(self, u, v, key=None):
        """Return True if the edge ``(u, v[, key])`` is a bridge of the source graph."""
        if self._source.is_multigraph():
            return (u, v, key) in self._bridge_set
        return (u, v) in self._bridge_set or (v, u) in self._bridge_set

    def bridge_between(self, block_u, block_v):
        """Return the original identity of the bridge joining two blocks.

        Returns ``(u, v)`` for simple graphs, ``(u, v, key)`` for
        multigraphs, or None if the blocks are not joined by a bridge.
        """
        if self.graph.is_multigraph():
            data = self.graph.get_edge_data(block_u, block_v)
            if not data:
                return None
            return data[next(iter(data))]["bridge"]
        data = self.graph.get_edge_data(block_u, block_v)
        return None if data is None else data["bridge"]


@not_implemented_for("directed")
@nx._dispatchable(graphs="G", preserve_all_attrs=True, returns_graph=False)
def bridge_contraction(G):
    """Contract the bridge-free connected components of ``G``.

    Removing every bridge from an undirected graph splits it into
    2-edge-connected components (*blocks*).  This function contracts each
    block to a single node; the bridges of ``G`` become the edges between
    block nodes, so the result is a forest (the *bridge tree* when ``G``
    is connected).

    Parameters
    ----------
    G : undirected graph
        A NetworkX Graph or MultiGraph.  It is never modified.

    Returns
    -------
    BridgeContraction
        A read-only view exposing the contracted graph (``.graph``), the
        block membership queries :meth:`BridgeContraction.block_of` and
        :meth:`BridgeContraction.members`, and the bridge identity
        queries :meth:`BridgeContraction.bridges`,
        :meth:`BridgeContraction.is_bridge` and
        :meth:`BridgeContraction.bridge_between`.

    Notes
    -----
    * Parallel edges of a multigraph are never bridges, so endpoints of
      parallel edges stay in the same block.
    * Self-loops are never bridges; they stay inside their block and do
      not appear in the contracted graph.
    * Isolated nodes form singleton blocks.
    * Block ids are assigned in the order the blocks' member nodes first
      appear in ``G``.
    * All attributes stored on the contracted graph (graph, block-node
      and bridge-edge attributes) are independent deep copies; mutating
      them does not affect ``G``, and the view never mutates ``G``.

    Examples
    --------
    Two triangles joined by a single bridge contract to two blocks:

    >>> G = nx.Graph()
    >>> G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # triangle, block 0
    >>> G.add_edges_from([(3, 4), (4, 5), (5, 3)])  # triangle, block 1
    >>> G.add_edge(2, 3, color="red")  # the bridge
    >>> bc = nx.bridge_contraction(G)
    >>> bc.blocks()
    [frozenset({0, 1, 2}), frozenset({3, 4, 5})]
    >>> bc.block_of(4)
    1
    >>> bc.bridges()
    [(2, 3)]
    >>> bc.is_bridge(2, 3)
    True
    >>> sorted(bc.graph.edges(data=True))
    [(0, 1, {'bridge': (2, 3), 'color': 'red'})]

    Parallel edges keep their endpoints in one block, and isolated nodes
    form singleton blocks:

    >>> M = nx.MultiGraph()
    >>> _ = M.add_edges_from([(0, 1), (0, 1)])  # parallel edges: no bridge
    >>> M.add_node(7)  # isolated node
    >>> bm = nx.bridge_contraction(M)
    >>> bm.blocks()
    [frozenset({0, 1}), frozenset({7})]
    """
    multigraph = G.is_multigraph()

    # Bridge identities, with edge keys for multigraphs.  A bridge of a
    # multigraph is the sole edge between its endpoints, so its key is
    # the only key in G[u][v].
    bridges = []
    for u, v in nx.bridges(G):
        key = next(iter(G[u][v])) if multigraph else None
        bridges.append((u, v, key))
    bridge_pairs = {(u, v) for u, v, _ in bridges}

    # Union-find over non-bridge edges.  Iterating nodes in G's order and
    # assigning block ids on first sight fixes the block order.
    uf = nx.utils.UnionFind()
    for u, v in G.edges():
        if u == v:
            continue  # self-loops stay inside their block
        if (u, v) in bridge_pairs or (v, u) in bridge_pairs:
            continue
        uf.union(u, v)

    block_of = {}
    blocks = []
    roots = {}
    for node in G:
        root = uf[node]
        bid = roots.get(root)
        if bid is None:
            bid = len(blocks)
            roots[root] = bid
            blocks.append(set())
        blocks[bid].add(node)
        block_of[node] = bid
    blocks = [frozenset(b) for b in blocks]

    C = nx.MultiGraph() if multigraph else nx.Graph()
    C.graph.update(deepcopy(dict(G.graph)))
    for bid, members in enumerate(blocks):
        C.add_node(bid, members=members)
    for u, v, key in bridges:
        bu, bv = block_of[u], block_of[v]
        if multigraph:
            attrs = deepcopy(dict(G[u][v][key]))
            C.add_edge(bu, bv, key=key, bridge=(u, v, key), **attrs)
        else:
            attrs = deepcopy(dict(G[u][v]))
            C.add_edge(bu, bv, bridge=(u, v), **attrs)

    return BridgeContraction(G, C, block_of, blocks, bridges)
