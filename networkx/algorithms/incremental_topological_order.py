"""Incrementally maintained topological order."""

import heapq
from collections import deque
from itertools import pairwise

import networkx as nx
from networkx.classes.graphviews import generic_graph_view

__all__ = ["IncrementalTopologicalOrder", "IncrementalTopologicalOrderBatchResult"]


class IncrementalTopologicalOrderBatchResult:
    """Result of applying a batch to an incremental topological order.

    Attributes
    ----------
    accepted : bool
        Whether the complete batch was applied.
    cycle : list of tuple pairs or None
        If the batch creates a cycle, a closed walk represented by its actual
        directed edges. Otherwise, this is ``None``.
    """

    __slots__ = ("accepted", "cycle")

    def __init__(self, accepted, cycle=None):
        self.accepted = accepted
        self.cycle = cycle

    def __iter__(self):
        yield self.accepted
        yield self.cycle

    def __eq__(self, other):
        return (
            isinstance(other, IncrementalTopologicalOrderBatchResult)
            and self.accepted == other.accepted
            and self.cycle == other.cycle
        )

    def __repr__(self):
        if self.accepted:
            return "IncrementalTopologicalOrderBatchResult(accepted=True, cycle=None)"
        return f"IncrementalTopologicalOrderBatchResult(accepted=False, cycle={self.cycle!r})"


class IncrementalTopologicalOrder:
    """A topological order maintained while nodes and edges are added in batches.

    Parameters
    ----------
    G : NetworkX DiGraph
        An acyclic directed graph. The index stores a copy, so later changes to
        the input graph do not affect the index.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not a simple directed graph.
    HasACycle
        If `G` is not acyclic.

    Examples
    --------
    >>> G = nx.DiGraph([("A", "C"), ("B", "C")])
    >>> order = nx.IncrementalTopologicalOrder(G)
    >>> order.apply_batch(edges=[("B", "A")]).accepted
    True
    >>> list(order)
    ['B', 'A', 'C']
    >>> result = order.apply_batch(edges=[("C", "B")])
    >>> result.accepted
    False
    >>> result.cycle
    [('B', 'C'), ('C', 'B')]
    >>> list(order)
    ['B', 'A', 'C']

    Notes
    -----
    Ties are resolved by insertion order. Initial nodes use their order in the
    input graph, and nodes added later use their order in the successful batch
    that introduces them. Node values are never compared with one another.

    Initialization takes ``O((V + E) log V)`` time. For a successful batch,
    only the interval spanning the positional endpoints of newly backward edges
    is reordered. If that interval contains ``k`` nodes and ``p`` incoming
    adjacency entries, the restricted Kahn update takes
    ``O(k log k + p)`` time. A rejected batch performs an additional restricted
    search for an actual cycle witness; worst-case this is ``O(b(k + p))`` for
    ``b`` backward edges. A rejected batch leaves the graph and index unchanged.
    """

    def __init__(self, G):
        if not G.is_directed() or G.is_multigraph():
            raise nx.NetworkXNotImplemented(
                "IncrementalTopologicalOrder requires a DiGraph."
            )

        self._graph = nx.DiGraph(G)
        self._rank = {node: position for position, node in enumerate(self._graph)}
        self._order = self._initial_order()
        self._position = {node: i for i, node in enumerate(self._order)}
        self._next_rank = len(self._rank)
        self._graph_view = generic_graph_view(self._graph)

    def __iter__(self):
        return iter(self._order)

    def __len__(self):
        return len(self._order)

    @property
    def graph(self):
        """A frozen read-only view of the graph represented by this index."""
        return self._graph_view

    @property
    def order(self):
        """The current topological order as a tuple."""
        return tuple(self._order)

    def position(self, node):
        """Return the zero-based position of `node` in the topological order.

        Parameters
        ----------
        node : node
            A node in the graph.

        Raises
        ------
        NodeNotFound
            If `node` is not in the graph.
        """
        try:
            return self._position[node]
        except KeyError as err:
            raise nx.NodeNotFound(f"Node {node!r} is not in the graph.") from err

    def apply_batch(self, nodes=(), edges=()):
        """Add declared nodes and edges atomically.

        Parameters
        ----------
        nodes : iterable of nodes, optional
            Nodes to add. Existing nodes and repeated declarations are ignored.
        edges : iterable of node pairs, optional
            Edges to add. Every endpoint must already exist or appear in
            `nodes`. Repeated declarations of an existing edge are ignored.

        Returns
        -------
        IncrementalTopologicalOrderBatchResult
            ``accepted`` is ``True`` only if every node and edge in the batch
            can be added. If an edge creates a cycle, ``cycle`` contains a
            closed walk made from actual directed edges, and the whole batch is
            rolled back.

        Raises
        ------
        NetworkXError
            If an edge endpoint is neither present nor declared in this batch.
        """
        node_list = list(nodes)
        edge_list = self._deduplicate_edges(edges)
        declared_nodes = set(node_list)

        for u, v in edge_list:
            if not (self._graph.has_node(u) or u in declared_nodes):
                raise nx.NetworkXError(f"Node {u!r} must be declared in this batch.")
            if not (self._graph.has_node(v) or v in declared_nodes):
                raise nx.NetworkXError(f"Node {v!r} must be declared in this batch.")

        new_nodes = []
        seen_new_nodes = set()
        for node in node_list:
            if not self._graph.has_node(node) and node not in seen_new_nodes:
                new_nodes.append(node)
                seen_new_nodes.add(node)

        new_edges = [(u, v) for u, v in edge_list if not self._graph.has_edge(u, v)]

        saved_order = self._order.copy()
        saved_position = self._position.copy()
        saved_rank = self._rank.copy()
        saved_next_rank = self._next_rank

        for node in new_nodes:
            self._add_node(node)

        try:
            backward_edges = []
            for u, v in new_edges:
                self._graph.add_edge(u, v)
                if self._position[u] >= self._position[v]:
                    backward_edges.append((u, v))

            if backward_edges:
                left = min(self._position[v] for _, v in backward_edges)
                right = max(self._position[u] for u, _ in backward_edges)
                accepted, affected = self._reorder_interval(left, right)
                if not accepted:
                    cycle = self._cycle_witness(affected, backward_edges)
                    self._rollback(
                        new_nodes,
                        new_edges,
                        saved_order,
                        saved_position,
                        saved_rank,
                        saved_next_rank,
                    )
                    return IncrementalTopologicalOrderBatchResult(False, cycle)
        except Exception:
            self._rollback(
                new_nodes,
                new_edges,
                saved_order,
                saved_position,
                saved_rank,
                saved_next_rank,
            )
            raise

        return IncrementalTopologicalOrderBatchResult(True)

    def _initial_order(self):
        indegree = {node: len(self._graph.pred[node]) for node in self._graph}
        ready = [
            (self._rank[node], node) for node in self._graph if indegree[node] == 0
        ]
        heapq.heapify(ready)
        order = []

        while ready:
            _, node = heapq.heappop(ready)
            order.append(node)
            for child in self._graph.successors(node):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, (self._rank[child], child))

        if len(order) != len(self._graph):
            raise nx.HasACycle("The initial graph must be a directed acyclic graph.")
        return order

    def _add_node(self, node):
        position = len(self._order)
        self._graph.add_node(node)
        self._order.append(node)
        self._position[node] = position
        self._rank[node] = self._next_rank
        self._next_rank += 1

    def _reorder_interval(self, left, right):
        segment = self._order[left : right + 1]
        affected = set(segment)
        children = {node: [] for node in segment}
        indegree = {}
        for node in segment:
            predecessors = list(self._graph.predecessors(node))
            internal_predecessors = [p for p in predecessors if p in affected]
            indegree[node] = len(internal_predecessors)
            for predecessor in internal_predecessors:
                children[predecessor].append(node)

        ready = [(self._rank[node], node) for node in segment if indegree[node] == 0]
        heapq.heapify(ready)
        reordered = []
        while ready:
            _, node = heapq.heappop(ready)
            reordered.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, (self._rank[child], child))

        if len(reordered) != len(segment):
            return False, affected

        self._order[left : right + 1] = reordered
        self._position.update((node, left + i) for i, node in enumerate(reordered))
        return True, affected

    def _cycle_witness(self, affected, backward_edges):
        for source, target in backward_edges:
            if source == target:
                return [(source, target)]

        for source, target in backward_edges:
            parent = {target: None}
            queue = deque([target])
            while queue:
                node = queue.popleft()
                for child in self._graph.successors(node):
                    if child not in affected or child in parent:
                        continue
                    parent[child] = node
                    if child != source:
                        queue.append(child)
                        continue

                    current = source
                    chain = []
                    while current is not None:
                        chain.append(current)
                        current = parent[current]
                    path = list(reversed(chain))
                    path.append(target)
                    return list(pairwise(path))

        raise RuntimeError("Failed to reconstruct the cycle in the affected interval.")

    def _rollback(
        self,
        new_nodes,
        new_edges,
        saved_order,
        saved_position,
        saved_rank,
        saved_next_rank,
    ):
        for u, v in new_edges:
            if self._graph.has_edge(u, v):
                self._graph.remove_edge(u, v)
        for node in new_nodes:
            if self._graph.has_node(node):
                self._graph.remove_node(node)
        self._order = saved_order
        self._position = saved_position
        self._rank = saved_rank
        self._next_rank = saved_next_rank

    @staticmethod
    def _deduplicate_edges(edges):
        result = []
        seen = set()
        for edge in edges:
            try:
                u, v = edge
            except (TypeError, ValueError) as err:
                raise nx.NetworkXError("Each edge must be a pair of nodes.") from err
            if (u, v) not in seen:
                seen.add((u, v))
                result.append((u, v))
        return result
