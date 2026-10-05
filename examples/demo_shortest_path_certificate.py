"""Demo: independent shortest-path certificate checking in NetworkX.

A shortest-path *certificate* is a claimed distance table plus a predecessor
table.  ``nx.verify_shortest_path_certificate`` checks the certificate against
the graph -- it never reruns Dijkstra to manufacture an answer -- by verifying:

1. every edge obeys the distance constraint ``dist[v] <= dist[u] + w``;
2. every predecessor edge really exists (with an explicit key on multigraphs);
3. every predecessor edge is tight: ``dist[v] == dist[u] + w``;
4. predecessor chains are acyclic and end at a source.

Unreachable nodes are simply omitted from ``dist`` (they are never silently
mixed with distance 0), and negative edge weights are explicitly rejected.

Run the demo::

    python examples/demo_shortest_path_certificate.py

Run the accompanying local, scoped pytest suite (this repository's conftest
imports a matplotlib build that is broken in some environments, so conftest
loading and global plugin autoload are disabled -- the tests need neither)::

    set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    python -m pytest --noconftest -q ^
        networkx/algorithms/shortest_paths/tests/test_certificate.py

On POSIX shells the same command is a single line::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -q \\
        networkx/algorithms/shortest_paths/tests/test_certificate.py
"""

import os
import sys

# When executed directly from the repository, prefer the in-tree networkx.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from networkx.algorithms.shortest_paths.certificate import (
    CertificateError,
    NegativeEdgeError,
)


def show(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def try_verify(G, sources, dist, pred, **kwargs):
    try:
        nx.verify_shortest_path_certificate(G, sources, dist, pred, **kwargs)
        print("ACCEPTED: the certificate is valid.")
        return True
    except CertificateError as err:
        print(f"REJECTED ({len(err.violations)} finding(s)):")
        for v in err.violations:
            print(f"  - [{v.kind}] {v}")
        return False
    except NegativeEdgeError as err:
        print(f"REJECTED (unsupported input): {err}")
        return False


# 1. Correct three-node certificate -----------------------------------------
show("1. Correct certificate on the three-node path 0 - 1 - 2")
G = nx.path_graph(3)
good_dist = {0: 0, 1: 1, 2: 2}
good_pred = {1: 0, 2: 1}
print("dist =", good_dist, " pred =", good_pred)
assert try_verify(G, 0, good_dist, good_pred)

# 2. Terminal distance made too small ---------------------------------------
show("2. Bad certificate: terminal distance shrunk to 0")
bad_dist = {0: 0, 1: 1, 2: 0}
print("dist =", bad_dist, " pred =", good_pred)
assert not try_verify(G, 0, bad_dist, good_pred)
print("-> the finding names the exact edge (1, 2): dist[2]=0 would need to")
print("   equal dist[1]+1=2 on its claimed predecessor edge.")

# 3. Terminal distance made too large (pure edge-constraint failure) --------
show("3. Bad certificate: terminal distance inflated to 9")
big_dist = {0: 0, 1: 1, 2: 9}
print("dist =", big_dist, " pred =", good_pred)
assert not try_verify(G, 0, big_dist, good_pred)
print("-> edge (1, 2) violates dist[2] <= dist[1] + 1.")

# 4. Predecessor self-loop -> chain diagnosis -------------------------------
show("4. Bad certificate: predecessor self-loop pred[2] = 2")
loop_pred = {1: 0, 2: 2}
print("dist =", good_dist, " pred =", loop_pred)
assert not try_verify(G, 0, good_dist, loop_pred)
print("-> the predecessor chain is diagnosed as the cycle 2 -> 2.")

# 5. Parallel edges require an explicit edge identity -----------------------
show("5. Parallel edges: edge identity must be explicit (multigraph)")
M = nx.MultiDiGraph()
M.add_edge(0, 1, key="a", weight=5)
M.add_edge(0, 1, key="b", weight=1)
M.add_edge(1, 2, key="c", weight=2)
print("edges: (0,1,key='a',w=5), (0,1,key='b',w=1), (1,2,key='c',w=2)")
print("-- bare predecessor node, ambiguous between the two parallel edges:")
assert not try_verify(M, 0, {0: 0, 1: 1, 2: 3}, {1: 0, 2: 1})
print("-- explicit (node, key) pair selects the cheap edge 'b':")
assert try_verify(M, 0, {0: 0, 1: 1, 2: 3}, {1: (0, "b"), 2: 1})

# 6. Unreachable nodes are omitted, not set to 0 ----------------------------
show("6. Disconnected node is unreachable (omitted), never distance 0")
H = nx.path_graph(3)
H.add_edge(8, 9)
print("component 8-9 is disconnected from source 0")
assert try_verify(H, 0, good_dist, good_pred)  # 8 and 9 absent: fine
assert not try_verify(H, 0, {**good_dist, 9: 0}, good_pred)

# 7. Negative weights are explicitly unsupported ----------------------------
show("7. Negative edge weights are explicitly unsupported")
N = nx.Graph()
N.add_edge(0, 1, weight=-1)
assert not try_verify(N, 0, {0: 0}, None)

# 8. The original graph is untouched ----------------------------------------
show("8. The graph is never modified")
edges_before = sorted(G.edges())
try_verify(G, 0, bad_dist, good_pred)
assert sorted(G.edges()) == edges_before
print("edges after a rejected verification:", sorted(G.edges()), "(unchanged)")

print("\nAll demo checks behaved as expected.")

