"""
===================
DAG scheduling demo
===================

Terminal demo of ``nx.dag_schedule``: builds a small task graph with
durations, release times and edge lags, prints the resulting schedule,
and shows the error raised when the deadline is infeasible.

Run with:  python examples/dag_schedule_demo.py
"""

import networkx as nx

G = nx.DiGraph()
G.add_node("A", duration=2)
G.add_node("B", duration=4, release=1)
G.add_node("C", duration=3)
G.add_edge("A", "C")
G.add_edge("B", "C", lag=2)

print("Input graph:")
for node, data in G.nodes(data=True):
    print(
        f"  node {node}: duration={data.get('duration', 1)}, release={data.get('release', 0)}"
    )
for u, v, data in G.edges(data=True):
    print(f"  edge {u} -> {v}: lag={data.get('lag', 0)}")

sched = nx.dag_schedule(G, deadline=12)
print("\nSchedule (deadline=12):")
print(f"  makespan: {sched['makespan']}")
for node in G:
    print(
        f"  {node}: earliest_start={sched['earliest_start'][node]}, "
        f"latest_start={sched['latest_start'][node]}, "
        f"total_float={sched['total_float'][node]}"
    )
print(f"  critical_path: {sched['critical_path']}")

print("\nInfeasible deadline (deadline=9):")
try:
    nx.dag_schedule(G, deadline=9)
except nx.NetworkXDeadlineExceeded as exc:
    print(f"  error type: {type(exc).__name__}")
    print(f"  deadline: {exc.deadline}")
    print(f"  minimum_makespan: {exc.minimum_makespan}")
    print(f"  message: {exc}")
