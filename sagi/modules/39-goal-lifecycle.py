# sagi/modules/39-goal-lifecycle.py — tier-4 · "steer thy purpose"
#
# GAP CLOSED: there is one standing goal (goal.txt) but no lifecycle — goals can't be proposed,
# activated, tracked to completion, or retired, and nothing escalates a stuck goal to the operator.
# This manages that lifecycle: propose a goal, activate it (which re-steers the whole individual via
# the human-interface → goal.txt → self-prompt-auditor chain), track convergence via goal-graph, and
# retire a completed one. Ranked by outcome-learning so the individual pursues what has paid off.
#
# GROUNDED REUSE: goal-graph.decompose/next_module (convergence signal), human-interface.steer_goal
# (activation writes goal.txt + refreshes the prompt), outcome-learning.rank (prioritise). STDLIB only.
from __future__ import annotations

import time

MODULE_ID = "goal-lifecycle"
DEPS = ["goal-graph", "outcome-learning", "human-interface"]
MOTTO = "steer thy purpose"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _goals():
        return host.store.read_json("goals.json", default=[]) or []

    def _save(g):
        try:
            host.store.write_json("goals.json", g)
        except Exception:
            pass

    def propose(text):
        """Add a candidate goal (not yet active)."""
        goals = _goals()
        gid = f"g{len(goals) + 1}"
        goals.append({"id": gid, "text": text.strip(), "status": "proposed", "ts": int(time.time())})
        _save(goals)
        return {"id": gid, "text": text.strip()}

    def activate_goal(gid):
        """Make a proposed goal the standing goal — re-steers the individual (goal.txt + prompt)."""
        goals = _goals()
        g = next((x for x in goals if x["id"] == gid), None)
        if not g:
            return {"error": f"no goal {gid}"}
        for x in goals:
            if x["status"] == "active":
                x["status"] = "superseded"
        g["status"] = "active"
        _save(goals)
        hi = _sib(host, "human-interface")
        if hi and "steer_goal" in hi:
            hi["steer_goal"](g["text"])
        else:
            host.store.write("goal.txt", g["text"] + "\n")
        host.log("goal_activate", id=gid, text=g["text"][:60])
        return {"id": gid, "text": g["text"], "status": "active"}

    def status():
        """Active goal + whether the build graph has converged toward it (the completion signal)."""
        goals = _goals()
        active = next((g for g in goals if g["status"] == "active"), None)
        gg = _sib(host, "goal-graph")
        converged = None
        if gg:
            try:
                converged = gg["next_module"]() is None
            except Exception:
                converged = None
        ranked = None
        ol = _sib(host, "outcome-learning")
        if ol:
            try:
                ranked = [g["id"] for g in ol["rank"]([{"id": x["id"]} for x in goals if x["status"] == "proposed"])]
            except Exception:
                ranked = None
        return {"active": active, "converged": converged, "proposed_ranked": ranked,
                "goal_txt": (host.store.read("goal.txt") or "").strip()}

    def retire(gid):
        """Mark a goal completed/retired."""
        goals = _goals()
        for g in goals:
            if g["id"] == gid:
                g["status"] = "retired"
        _save(goals)
        return {"id": gid, "status": "retired"}

    host.log("module", step="tier4-39", id=MODULE_ID)
    return {"propose": propose, "activate_goal": activate_goal, "status": status, "retire": retire}
