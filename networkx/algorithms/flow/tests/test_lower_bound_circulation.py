"""Tests for lower_bound_circulation."""

import copy
import itertools

import pytest

import networkx as nx


def edge_flows(G, flow):
    """Yield (u, v, key, f) for every edge of G from a flow dict."""
    if G.is_multigraph():
        for u, v, k in G.edges(keys=True):
            yield u, v, k, flow[u][v][k]
    else:
        for u, v in G.edges:
            yield u, v, None, flow[u][v]


def assert_valid_circulation(G, flow, demand="demand", lower="lower", upper="upper"):
    """Check bounds on every edge and conservation at every node."""
    net = {n: 0 for n in G}
    for u, v, k, f in edge_flows(G, flow):
        data = G[u][v][k] if G.is_multigraph() else G[u][v]
        lo = data.get(lower, 0)
        hi = data.get(upper, None)
        assert isinstance(f, int)
        assert lo <= f
        if hi is not None:
            assert f <= hi
        net[u] -= f
        net[v] += f
    for n in G:
        assert net[n] == G.nodes[n].get(demand, 0)


def verify_cut_witness(G, exc, demand="demand", lower="lower", upper="upper"):
    """Independently verify the Hoffman cut certificate on the exception."""
    S = set(exc.cut)
    T = set(G) - S
    upper_ST = 0
    lower_TS = 0
    for u, v, k, data in (
        G.edges(data=True, keys=True)
        if G.is_multigraph()
        else ((u, v, None, d) for u, v, d in G.edges(data=True))
    ):
        hi = data.get(upper, None)
        lo = data.get(lower, 0)
        if u in S and v in T:
            assert hi is not None, "cut through an uncapacitated edge"
            upper_ST += hi
        if u in T and v in S:
            lower_TS += lo
    demand_T = sum(G.nodes[n].get(demand, 0) for n in T)
    assert demand_T - (upper_ST - lower_TS) == exc.deficit > 0


class TestAcceptance:
    def test_basic_feasible(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", lower=2, upper=5)
        G.nodes["A"]["demand"] = -3
        G.nodes["B"]["demand"] = 3
        flow = nx.lower_bound_circulation(G)
        assert flow["A"]["B"] == 3
        assert_valid_circulation(G, flow)

    def test_basic_infeasible_deficit(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", lower=2, upper=2)
        G.nodes["A"]["demand"] = -3
        G.nodes["B"]["demand"] = 3
        with pytest.raises(nx.InfeasibleCirculationError) as excinfo:
            nx.lower_bound_circulation(G)
        exc = excinfo.value
        assert exc.deficit == 1
        assert exc.cut == frozenset({"A"})
        assert isinstance(exc, nx.NetworkXUnfeasible)
        verify_cut_witness(G, exc)


class TestConservationRecheck:
    """Restored flows are re-checked against conservation and bounds."""

    def test_diamond(self):
        G = nx.DiGraph()
        G.add_edge("s", "a", lower=1, upper=4)
        G.add_edge("s", "b", lower=0, upper=4)
        G.add_edge("a", "t", lower=0, upper=4)
        G.add_edge("b", "t", lower=2, upper=4)
        G.add_edge("a", "b", lower=0, upper=1)
        G.nodes["s"]["demand"] = -5
        G.nodes["t"]["demand"] = 5
        flow = nx.lower_bound_circulation(G)
        assert_valid_circulation(G, flow)
        assert flow["s"]["a"] + flow["s"]["b"] == 5

    def test_self_loop_gets_lower_bound(self):
        G = nx.DiGraph()
        G.add_edge("a", "a", lower=2, upper=4)
        G.add_edge("a", "b", lower=1, upper=3)
        G.nodes["a"]["demand"] = -1
        G.nodes["b"]["demand"] = 1
        flow = nx.lower_bound_circulation(G)
        assert flow["a"]["a"] == 2
        assert_valid_circulation(G, flow)

    def test_disconnected_components(self):
        G = nx.DiGraph()
        G.add_edge("a", "b", lower=1, upper=3)
        G.nodes["a"]["demand"] = -2
        G.nodes["b"]["demand"] = 2
        G.add_edge("x", "y", lower=0, upper=10)
        G.nodes["x"]["demand"] = -4
        G.nodes["y"]["demand"] = 4
        flow = nx.lower_bound_circulation(G)
        assert flow["a"]["b"] == 2
        assert flow["x"]["y"] == 4
        assert_valid_circulation(G, flow)

    def test_disconnected_infeasible_component(self):
        G = nx.DiGraph()
        G.add_edge("a", "b", lower=0, upper=5)
        G.nodes["a"]["demand"] = -1
        G.nodes["b"]["demand"] = 1
        # Isolated demand nodes in a separate component cannot be served.
        G.add_nodes_from([("x", {"demand": -2}), ("y", {"demand": 2})])
        with pytest.raises(nx.InfeasibleCirculationError) as excinfo:
            nx.lower_bound_circulation(G)
        assert excinfo.value.deficit == 2
        verify_cut_witness(G, excinfo.value)

    def test_zero_demands(self):
        G = nx.DiGraph()
        G.add_edge("a", "b", lower=1, upper=3)
        G.add_edge("b", "a", lower=1, upper=3)
        flow = nx.lower_bound_circulation(G)
        assert flow == {"a": {"b": 1}, "b": {"a": 1}}
        assert_valid_circulation(G, flow)

    def test_empty_graph(self):
        assert nx.lower_bound_circulation(nx.DiGraph()) == {}
        assert nx.lower_bound_circulation(nx.MultiDiGraph()) == {}

    def test_nodes_without_edges(self):
        G = nx.DiGraph()
        G.add_nodes_from(["a", "b", "c"])
        assert nx.lower_bound_circulation(G) == {"a": {}, "b": {}, "c": {}}


class TestMultiDiGraph:
    def test_parallel_edges_keyed(self):
        M = nx.MultiDiGraph()
        M.add_edge("x", "y", lower=1, upper=2)
        M.add_edge("x", "y", lower=0, upper=2)
        M.nodes["x"]["demand"] = -2
        M.nodes["y"]["demand"] = 2
        flow = nx.lower_bound_circulation(M)
        assert set(flow["x"]["y"]) == {0, 1}
        assert_valid_circulation(M, flow)

    def test_parallel_bounds_respected_per_key(self):
        M = nx.MultiDiGraph()
        M.add_edge("x", "y", lower=0, upper=1)
        M.add_edge("x", "y", lower=0, upper=1)
        M.nodes["x"]["demand"] = -2
        M.nodes["y"]["demand"] = 2
        flow = nx.lower_bound_circulation(M)
        assert flow["x"]["y"] == {0: 1, 1: 1}
        assert_valid_circulation(M, flow)

    def test_parallel_infeasible(self):
        M = nx.MultiDiGraph()
        M.add_edge("x", "y", lower=0, upper=1)
        M.add_edge("x", "y", lower=0, upper=1)
        M.nodes["x"]["demand"] = -3
        M.nodes["y"]["demand"] = 3
        with pytest.raises(nx.InfeasibleCirculationError) as excinfo:
            nx.lower_bound_circulation(M)
        assert excinfo.value.deficit == 1
        verify_cut_witness(M, excinfo.value)

    def test_multigraph_self_loop(self):
        M = nx.MultiDiGraph()
        M.add_edge("a", "a", lower=3, upper=5)
        M.add_edge("a", "b", lower=1, upper=1)
        M.nodes["a"]["demand"] = -1
        M.nodes["b"]["demand"] = 1
        flow = nx.lower_bound_circulation(M)
        assert flow["a"]["a"] == {0: 3}
        assert flow["a"]["b"] == {0: 1}
        assert_valid_circulation(M, flow)


class TestInputValidation:
    def test_lower_exceeds_upper(self):
        G = nx.DiGraph()
        G.add_edge("a", "b", lower=3, upper=2)
        with pytest.raises(nx.NetworkXError, match="lower bound"):
            nx.lower_bound_circulation(G)

    @pytest.mark.parametrize("bad", [1.5, "2", 2.0, True, None])
    def test_non_integer_rejected(self, bad):
        G = nx.DiGraph()
        G.add_edge("a", "b", lower=0, upper=5)
        G.nodes["a"]["demand"] = bad
        G.nodes["b"]["demand"] = 0
        with pytest.raises(nx.NetworkXError, match="integer"):
            nx.lower_bound_circulation(G)
        H = nx.DiGraph()
        H.add_edge("a", "b", lower=bad, upper=5)
        with pytest.raises(nx.NetworkXError):
            nx.lower_bound_circulation(H)
        K = nx.DiGraph()
        K.add_edge("a", "b", lower=0, upper=bad)
        if bad is None:
            # None upper means unbounded and is accepted.
            nx.lower_bound_circulation(K)
        else:
            with pytest.raises(nx.NetworkXError, match="integer"):
                nx.lower_bound_circulation(K)

    def test_unbalanced_demands(self):
        G = nx.DiGraph()
        G.add_edge("a", "b", lower=0, upper=5)
        G.nodes["a"]["demand"] = -3
        G.nodes["b"]["demand"] = 2
        with pytest.raises(nx.NetworkXUnfeasible, match="unbalanced"):
            nx.lower_bound_circulation(G)

    def test_undirected_rejected(self):
        G = nx.Graph()
        G.add_edge("a", "b", lower=0, upper=1)
        with pytest.raises(nx.NetworkXError, match="directed"):
            nx.lower_bound_circulation(G)

    def test_input_not_modified(self):
        G = nx.MultiDiGraph()
        G.add_edge("a", "b", lower=1, upper=3)
        G.add_edge("a", "b", lower=0, upper=2)
        G.add_edge("b", "a", lower=0, upper=4)
        G.nodes["a"]["demand"] = -2
        G.nodes["b"]["demand"] = 2
        snapshot = copy.deepcopy(G)
        nx.lower_bound_circulation(G)
        assert nx.utils.graphs_equal(G, snapshot)


def brute_force_feasible(nodes, edges, demands):
    """edges: list of (u, v, lo, hi). Exhaustive flow enumeration."""
    ranges = [range(lo, hi + 1) for _, _, lo, hi in edges]
    for combo in itertools.product(*ranges):
        net = dict.fromkeys(nodes, 0)
        for (u, v, _, _), f in zip(edges, combo):
            net[u] -= f
            net[v] += f
        if all(net[n] == demands.get(n, 0) for n in nodes):
            return True
    return False


class TestExhaustiveSmallGraphs:
    """Exhaustive comparison against brute-force enumeration on tiny graphs."""

    def test_digraph_exhaustive(self):
        nodes = [0, 1, 2]
        pool = [(0, 1), (1, 2), (0, 2), (1, 0)]
        bound_options = [(0, 0), (0, 1), (1, 1), (1, 2)]
        demand_options = [
            d for d in itertools.product((-1, 0, 1), repeat=3) if sum(d) == 0
        ]
        checked = 0
        for r in range(len(pool) + 1):
            for subset in itertools.combinations(pool, r):
                for bounds in itertools.product(bound_options, repeat=r):
                    edges = [(u, v, lo, hi) for (u, v), (lo, hi) in zip(subset, bounds)]
                    for dem in demand_options:
                        checked += 1
                        G = nx.DiGraph()
                        G.add_nodes_from(nodes)
                        for u, v, lo, hi in edges:
                            G.add_edge(u, v, lower=lo, upper=hi)
                        for n, d in zip(nodes, dem):
                            G.nodes[n]["demand"] = d
                        demands = dict(zip(nodes, dem))
                        feasible = brute_force_feasible(nodes, edges, demands)
                        if feasible:
                            flow = nx.lower_bound_circulation(G)
                            assert_valid_circulation(G, flow)
                        else:
                            with pytest.raises(
                                nx.InfeasibleCirculationError
                            ) as excinfo:
                                nx.lower_bound_circulation(G)
                            verify_cut_witness(G, excinfo.value)
        assert checked > 4000

    def test_multidigraph_randomized(self):
        rng = __import__("random").Random(20260925)
        nodes = [0, 1, 2]
        for _ in range(300):
            G = nx.MultiDiGraph()
            G.add_nodes_from(nodes)
            edges = []
            for _ in range(rng.randint(0, 5)):
                u = rng.choice(nodes)
                v = rng.choice(nodes)
                lo = rng.randint(0, 2)
                hi = lo + rng.randint(0, 2)
                G.add_edge(u, v, lower=lo, upper=hi)
                edges.append((u, v, lo, hi))
            dem = [rng.randint(-2, 2) for _ in nodes]
            dem[-1] = -sum(dem[:-1])  # balance the demands
            if abs(dem[-1]) > 4:
                continue
            for n, d in zip(nodes, dem):
                G.nodes[n]["demand"] = d
            demands = dict(zip(nodes, dem))
            feasible = brute_force_feasible(nodes, edges, demands)
            if feasible:
                flow = nx.lower_bound_circulation(G)
                assert_valid_circulation(G, flow)
            else:
                with pytest.raises(nx.InfeasibleCirculationError) as excinfo:
                    nx.lower_bound_circulation(G)
                verify_cut_witness(G, excinfo.value)
