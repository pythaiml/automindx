# sagi/modules/09-curriculum.py — expand-9 · "grow beyond thyself"
#
# The 26 specs include an Autonomous-Curriculum engine (ACGSAE) but only as prose. This makes it
# a real driver: it reads the goal-graph, topologically orders it into a curriculum, and advances
# one step at a time — but only through a gate. Each advance verifies the just-built module
# (module-verifier) and confirms the scorecard did not regress (evaluator) before allowing the
# next. It mirrors the build→verify→gate discipline already in sagi_build.build, but grounded in
# the individual's own dependency structure rather than a flat step count. It never fabricates a
# build; a real module comes from the model-driven loop (sagi_build) or an explicit builder.
from __future__ import annotations

MODULE_ID = "curriculum"
DEPS = ["goal-graph", "evaluator"]
MOTTO = "grow beyond thyself"


def _sib(host, mod_id):
    reg = getattr(host, "registry", None)
    return (reg["get"](mod_id) or {}).get("handle") if reg else None


def _toposort(nodes):
    """Order graph nodes so deps precede dependants (ignores 'goal' root, skips cycles)."""
    by_id = {n["id"]: n for n in nodes if n["id"] != "goal"}
    order, temp, done = [], set(), set()

    def visit(nid):
        if nid in done or nid not in by_id:
            return
        if nid in temp:
            return                                   # tolerate a cycle rather than crash
        temp.add(nid)
        for d in by_id[nid].get("deps", []):
            visit(d)
        temp.discard(nid)
        done.add(nid)
        order.append(nid)

    for nid in list(by_id):
        visit(nid)
    return [by_id[i] for i in order]


def activate(host):
    def plan_curriculum(goal=None):
        """Ordered list of curriculum steps from the goal-graph (deps precede dependants)."""
        gg = _sib(host, "goal-graph")
        graph = gg["decompose"](goal) if goal else (gg["graph"]() if gg else {"nodes": []})
        steps = _toposort(graph.get("nodes", []))
        host.store.write_json("curriculum.json",
                              {"goal": graph.get("goal"), "steps": [s["id"] for s in steps]})
        return [{"id": s["id"], "title": s.get("title", s["id"]), "deps": s.get("deps", []),
                 "status": s.get("status", "planned")} for s in steps]

    def advance(builder=None):
        """Advance the curriculum by one GATED step.

        builder: optional callable(next_node)->filename that actually grows the module (e.g. the
        model-driven sagi_build path). If omitted, advance reports the next step and the current
        gate state without fabricating a build.
        """
        gg = _sib(host, "goal-graph")
        verifier = _sib(host, "module-verifier")
        evaluator = _sib(host, "evaluator")

        before = evaluator["evaluate"]()["metrics"] if evaluator else {}
        nxt = gg["next_module"]() if gg else None
        if nxt is None:
            return {"converged": True, "score": before, "next": None}

        built_file = None
        if callable(builder):
            built_file = builder(nxt)               # the model actually grows the module here

        gate = {"verified": None, "regressed": None}
        if built_file and verifier:
            gate["verified"] = verifier["verify"](built_file)["ok"]
        if evaluator:
            after = evaluator["evaluate"]()["metrics"]
            gate["regressed"] = after.get("verified_fraction", 1) < before.get("verified_fraction", 1)
            before = after

        host.log("curriculum_advance", next=nxt["id"], built=built_file,
                 verified=gate["verified"], regressed=gate["regressed"])
        return {"converged": False, "next": nxt, "built": built_file, "gate": gate, "score": before}

    def status():
        """Where the curriculum stands: built vs remaining, and the current score."""
        gg = _sib(host, "goal-graph")
        evaluator = _sib(host, "evaluator")
        graph = gg["graph"]() if gg else {"nodes": []}
        built = [n["id"] for n in graph.get("nodes", []) if n.get("status") == "built"]
        remaining = [n["id"] for n in graph.get("nodes", [])
                     if n["id"] != "goal" and n.get("status") != "built"]
        return {"built": built, "remaining": remaining,
                "score": evaluator["evaluate"]()["metrics"] if evaluator else {}}

    host.log("module", step="expand-9", id=MODULE_ID)
    return {"plan_curriculum": plan_curriculum, "advance": advance, "status": status}
