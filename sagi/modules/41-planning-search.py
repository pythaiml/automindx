# sagi/modules/41-planning-search.py — tier-5 · "search thy paths"
#
# GAP CLOSED: goal-graph.next_module returns ONE next step; nothing does multi-step lookahead over
# the whole graph to order a build campaign by expected payoff. This adds weighted planning search:
# it topologically orders the goal graph, then ranks ready frontiers by outcome-learning weight, so
# the individual pursues the highest-value buildable path — not just the next satisfiable node.
#
# GROUNDED REUSE: goal-graph.graph (nodes+edges), outcome-learning.rank/weights (value signal).
from __future__ import annotations

MODULE_ID = "planning-search"
DEPS = ["goal-graph", "outcome-learning"]
MOTTO = "search thy paths"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _graph():
        gg = _sib(host, "goal-graph")
        try:
            return gg["graph"]() if gg else {"nodes": [], "edges": []}
        except Exception:
            return {"nodes": [], "edges": []}

    def frontier(built=None):
        """Nodes whose deps are all satisfied and which aren't built yet — the buildable set now."""
        g = _graph()
        done = set(built or [n["id"] for n in g["nodes"] if n.get("status") == "built"])
        out = []
        for n in g["nodes"]:
            if n["id"] in ("goal",) or n["id"] in done:
                continue
            if all(d in done or d == "goal" for d in n.get("deps", [])):
                out.append(n["id"])
        return out

    def search(depth=5, built=None):
        """Greedy weighted lookahead: repeatedly take the highest-value ready node up to `depth`."""
        ol = _sib(host, "outcome-learning")
        done = set(built or [n["id"] for n in _graph()["nodes"] if n.get("status") == "built"])
        path = []
        for _ in range(max(1, int(depth))):
            f = frontier(done)
            if not f:
                break
            ranked = ol["rank"](f) if ol else f
            pick = ranked[0]
            path.append(pick)
            done.add(pick)
        return {"path": path, "converged": not frontier(done)}

    def best_path():
        """The full recommended build order for the current goal (value-ranked topological plan)."""
        return search(depth=len(_graph()["nodes"]) + 1)

    host.log("module", step="tier5-41", id=MODULE_ID)
    return {"search": search, "best_path": best_path, "frontier": frontier}
