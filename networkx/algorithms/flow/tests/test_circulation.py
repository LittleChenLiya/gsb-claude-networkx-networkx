"""Tests for networkx.algorithms.flow.circulation."""

import itertools
import random

import pytest

import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.flow.circulation import (
    InfeasibleCirculation,
    lower_bound_circulation,
)


def edge_flows(G, flow):
    """Yield (u, v, key, flow) for every edge of G from a flow dict."""
    if G.is_multigraph():
        for u, v, k in G.edges(keys=True):
            yield u, v, k, flow[u][v][k]
    else:
        for u, v in G.edges():
            yield u, v, None, flow[u][v]


def check_valid_flow(G, flow, demand="demand", lower="lower", upper="upper"):
    """Assert bounds and conservation for a recovered flow dict."""
    for u, v, k, f in edge_flows(G, flow):
        data = G[u][v][k] if G.is_multigraph() else G[u][v]
        lo = data.get(lower, 0)
        hi = data.get(upper, None)
        assert isinstance(f, int)
        assert f >= lo
        if hi is not None:
            assert f <= hi
    balance = dict.fromkeys(G, 0)
    for u, v, k, f in edge_flows(G, flow):
        balance[u] -= f
        balance[v] += f
    for v in G:
        assert balance[v] == G.nodes[v].get(demand, 0)


def adjusted_demands(G, demand="demand", lower="lower"):
    """demand(v) - in_lower(v) + out_lower(v) for each node."""
    adjusted = {v: G.nodes[v].get(demand, 0) for v in G}
    if G.is_multigraph():
        edge_view = ((u, v, d) for u, v, k, d in G.edges(keys=True, data=True))
    else:
        edge_view = G.edges(data=True)
    for u, v, data in edge_view:
        if u == v:
            continue
        lo = data.get(lower, 0)
        adjusted[u] += lo
        adjusted[v] -= lo
    return adjusted


def check_cut_witness(G, exc, demand="demand", lower="lower", upper="upper"):
    """Verify the Gale-Hoffman cut certificate carried by the exception."""
    S, T = exc.cut
    assert isinstance(exc.deficit, int) and exc.deficit > 0
    assert S | T == frozenset(G.nodes)
    assert not (S & T)
    adjusted = adjusted_demands(G, demand, lower)
    if G.is_multigraph():
        edge_view = ((u, v, d) for u, v, k, d in G.edges(keys=True, data=True))
    else:
        edge_view = G.edges(data=True)
    cap = 0
    for u, v, data in edge_view:
        if u in S and v in T:
            hi = data.get(upper, None)
            assert hi is not None  # unbounded cut edge would be feasible
            cap += hi - data.get(lower, 0)
    assert sum(adjusted[v] for v in T) - cap == exc.deficit


def brute_force_feasible(G, demand="demand", lower="lower", upper="upper", bound=8):
    """Exhaustively search for a feasible integer flow."""
    edges = []
    if G.is_multigraph():
        edge_view = ((u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True))
    else:
        edge_view = ((u, v, None, d) for u, v, d in G.edges(data=True))
    for u, v, k, data in edge_view:
        lo = data.get(lower, 0)
        hi = data.get(upper, None)
        hi = bound if hi is None else hi
        edges.append((u, v, lo, hi))
    demands = {v: G.nodes[v].get(demand, 0) for v in G}
    ranges = [range(lo, hi + 1) for u, v, lo, hi in edges]
    for combo in itertools.product(*ranges):
        balance = dict.fromkeys(G, 0)
        for (u, v, lo, hi), f in zip(edges, combo):
            balance[u] -= f
            balance[v] += f
        if all(balance[v] == demands[v] for v in G):
            return True
    return False


class TestAcceptanceExample:
    def test_feasible(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", lower=2, upper=5)
        G.nodes["A"]["demand"] = -3
        G.nodes["B"]["demand"] = 3
        flow = lower_bound_circulation(G)
        assert flow["A"]["B"] == 3
        check_valid_flow(G, flow)

    def test_infeasible_deficit_one(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", lower=2, upper=2)
        G.nodes["A"]["demand"] = -3
        G.nodes["B"]["demand"] = 3
        with pytest.raises(InfeasibleCirculation) as excinfo:
            lower_bound_circulation(G)
        exc = excinfo.value
        assert isinstance(exc, nx.NetworkXUnfeasible)
        assert exc.deficit == 1
        assert exc.cut == (frozenset({"A"}), frozenset({"B"}))
        check_cut_witness(G, exc)


class TestStructure:
    def test_empty_graph(self):
        assert lower_bound_circulation(nx.DiGraph()) == {}
        assert lower_bound_circulation(nx.MultiDiGraph()) == {}

    def test_zero_demands_no_edges(self):
        G = nx.DiGraph([(1, 2), (2, 3)])
        G.add_node(4)  # isolated node
        flow = lower_bound_circulation(G)
        for _, _, _, f in edge_flows(G, flow):
            assert f == 0
        check_valid_flow(G, flow)

    def test_self_loops(self):
        G = nx.DiGraph()
        G.add_edge(1, 1, lower=2, upper=4)
        G.add_edge(1, 2, lower=1, upper=3)
        G.nodes[1]["demand"] = -2
        G.nodes[2]["demand"] = 2
        flow = lower_bound_circulation(G)
        assert flow[1][1] == 2  # self-loop carries its lower bound
        check_valid_flow(G, flow)

    def test_parallel_edges(self):
        G = nx.MultiDiGraph()
        G.add_edge("s", "t", key="a", lower=1, upper=2)
        G.add_edge("s", "t", key="b", lower=0, upper=5)
        G.add_edge("t", "s", key="c", lower=0, upper=1)
        G.nodes["s"]["demand"] = -4
        G.nodes["t"]["demand"] = 4
        flow = lower_bound_circulation(G)
        total = flow["s"]["t"]["a"] + flow["s"]["t"]["b"] - flow["t"]["s"]["c"]
        assert total == 4
        check_valid_flow(G, flow)

    def test_disconnected_components(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, lower=1, upper=3)
        G.nodes[1]["demand"] = -2
        G.nodes[2]["demand"] = 2
        G.add_edge("x", "y", lower=0, upper=1)
        G.nodes["x"]["demand"] = -1
        G.nodes["y"]["demand"] = 1
        flow = lower_bound_circulation(G)
        check_valid_flow(G, flow)

    def test_disconnected_infeasible_component(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, lower=0, upper=5)
        G.nodes[1]["demand"] = -1
        G.nodes[2]["demand"] = 1
        G.add_node("isolated_sink")
        G.nodes["isolated_sink"]["demand"] = 3
        G.add_edge("src1", "src2", lower=0, upper=9)
        G.nodes["src1"]["demand"] = -3
        with pytest.raises(InfeasibleCirculation) as excinfo:
            lower_bound_circulation(G)
        exc = excinfo.value
        assert exc.deficit == 3
        assert exc.cut[1] == frozenset({1, 2, "isolated_sink"})
        check_cut_witness(G, exc)

    def test_unbounded_upper(self):
        G = nx.DiGraph()
        G.add_edge("s", "t", lower=3)  # no upper attribute
        G.nodes["s"]["demand"] = -7
        G.nodes["t"]["demand"] = 7
        flow = lower_bound_circulation(G)
        assert flow["s"]["t"] == 7
        check_valid_flow(G, flow)

    def test_custom_attribute_names(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, lo=1, cap=4)
        G.nodes[0]["d"] = -2
        G.nodes[1]["d"] = 2
        flow = lower_bound_circulation(G, demand="d", lower="lo", upper="cap")
        assert flow[0][1] == 2
        check_valid_flow(G, flow, demand="d", lower="lo", upper="cap")

    def test_flow_func_parameter(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", lower=2, upper=5)
        G.nodes["A"]["demand"] = -3
        G.nodes["B"]["demand"] = 3
        flow = lower_bound_circulation(G, flow_func=edmonds_karp)
        assert flow["A"]["B"] == 3

    def test_input_not_modified(self):
        G = nx.MultiDiGraph()
        G.add_edge(1, 2, lower=1, upper=2, weight=5)
        G.add_edge(2, 3, lower=0, upper=2)
        G.nodes[1]["demand"] = -1
        G.nodes[3]["demand"] = 1
        snapshot = (
            dict(G.nodes(data=True)),
            list(G.edges(keys=True, data=True)),
        )
        lower_bound_circulation(G)
        assert dict(G.nodes(data=True)) == snapshot[0]
        assert list(G.edges(keys=True, data=True)) == snapshot[1]


class TestInvalidInput:
    def test_lower_greater_than_upper(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, lower=3, upper=2)
        with pytest.raises(nx.NetworkXError, match="greater than upper"):
            lower_bound_circulation(G)

    @pytest.mark.parametrize("bad", [1.5, "2", None, 2.0])
    def test_non_integer_bounds(self, bad):
        G = nx.DiGraph()
        G.add_edge(1, 2, lower=bad, upper=5)
        with pytest.raises(nx.NetworkXError, match="integer"):
            lower_bound_circulation(G)
        # A None upper bound is interpreted as unbounded, not invalid.
        if bad is None:
            return
        G = nx.DiGraph()
        G.add_edge(1, 2, lower=0, upper=bad)
        with pytest.raises(nx.NetworkXError, match="integer"):
            lower_bound_circulation(G)

    def test_non_integer_demand(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, lower=0, upper=5)
        G.nodes[1]["demand"] = 2.5
        G.nodes[2]["demand"] = -2.5
        with pytest.raises(nx.NetworkXError, match="integer"):
            lower_bound_circulation(G)

    def test_unbalanced_demands(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, lower=0, upper=5)
        G.nodes[1]["demand"] = -1
        G.nodes[2]["demand"] = 2
        with pytest.raises(nx.NetworkXError, match="must be zero"):
            lower_bound_circulation(G)

    def test_undirected_not_implemented(self):
        G = nx.Graph()
        G.add_edge(1, 2, lower=0, upper=1)
        with pytest.raises(nx.NetworkXNotImplemented):
            lower_bound_circulation(G)


class TestExhaustiveSmall:
    """Compare against brute-force enumeration on small integer graphs."""

    def check_instance(self, G):
        feasible = brute_force_feasible(G)
        if feasible:
            flow = lower_bound_circulation(G)
            check_valid_flow(G, flow)
        else:
            with pytest.raises(InfeasibleCirculation) as excinfo:
                lower_bound_circulation(G)
            check_cut_witness(G, excinfo.value)

    def test_two_node_digraphs_exhaustive(self):
        bounds = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        for has_ab, has_ba in itertools.product([False, True], repeat=2):
            edge_bounds = [bounds] * (has_ab + has_ba)
            for combo in itertools.product(*edge_bounds):
                for demand_a in range(-2, 3):
                    G = nx.DiGraph()
                    G.add_nodes_from(["a", "b"])
                    it = iter(combo)
                    if has_ab:
                        lo, hi = next(it)
                        G.add_edge("a", "b", lower=lo, upper=hi)
                    if has_ba:
                        lo, hi = next(it)
                        G.add_edge("b", "a", lower=lo, upper=hi)
                    G.nodes["a"]["demand"] = demand_a
                    G.nodes["b"]["demand"] = -demand_a
                    self.check_instance(G)

    def test_random_multidigraphs(self):
        rng = random.Random(42)
        for _ in range(300):
            n = rng.randint(1, 4)
            G = nx.MultiDiGraph()
            G.add_nodes_from(range(n))
            for _ in range(rng.randint(0, 5)):
                u = rng.randrange(n)
                v = rng.randrange(n)  # may be a self-loop
                lo = rng.randint(0, 1)
                hi = lo + rng.randint(0, 2)
                G.add_edge(u, v, lower=lo, upper=hi)
            demands = [rng.randint(-2, 2) for _ in range(n)]
            demands[-1] = -sum(demands[:-1])  # balance
            for v, d in zip(G.nodes, demands):
                G.nodes[v]["demand"] = d
            self.check_instance(G)
