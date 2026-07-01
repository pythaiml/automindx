# sagi/modules/05-goal-graph.py — expand-5 · "decompose thyself"
#
# sagi_build.build_step already asks a model for "the NEXT single module given what's built",
# steered by goal.txt — but the result is one throwaway title, never a structure. This module
# turns the goal (goal.txt = "expand sAGI") into a persisted, inspectable, dependency-ordered
# graph of buildable modules and picks the next node whose deps are satisfied. When a model is
# wired the decomposition is model-proposed (recall-conditioned via memory-recall, partitioned
# proof/conjecture via epistemic-calibration); offline it is DERIVED from the live modules'
# declared DEPS — a true dependency graph of what has actually been grown. Persisted to
# goal_graph.json inside the package (do no harm — Store-confined).
from __future__ import annotations

import json
import re

MODULE_ID = "goal-graph"
DEPS = ["memory-recall", "epistemic-calibration"]
MOTTO = "decompose thyself"

_GRAPH_FILE = "goal_graph.json"


def _sib(host, mod_id):
    reg = getattr(host, "registry", None)
    return (reg["get"](mod_id) or {}).get("handle") if reg else None


def activate(host):
    def _live_loaders():
        loader = _sib(host, "module-loader")
        return loader["build_loaders"]() if loader else {}

    def _grounded_graph(goal):
        """A dependency graph of the modules actually grown — nodes + DEPS edges (always available)."""
        loaders = _live_loaders()
        nodes = [{"id": "goal", "title": goal, "deps": [], "status": "root"}]
        edges = []
        for mid, spec in loaders.items():
            nodes.append({"id": mid, "title": spec.get("file", mid),
                          "deps": spec.get("deps", []), "status": "built",
                          "motto": spec.get("motto", "")})
            for d in spec.get("deps", []):
                edges.append({"from": d, "to": mid})
            if not spec.get("deps"):
                edges.append({"from": "goal", "to": mid})
        return {"goal": goal, "nodes": nodes, "edges": edges, "source": "grounded"}

    def _model_graph(goal):
        """Ask a model for a JSON decomposition, recall-conditioned + calibrated. None on failure."""
        recall = _sib(host, "memory-recall")
        calib = _sib(host, "epistemic-calibration")
        context = recall["context_for"](goal) if recall else ""
        prompt = (
            f"{context}\n\nOverarching goal: {goal}.\n"
            "Decompose it into the next buildable modules. Reply ONLY JSON: "
            '{\"nodes\":[{\"id\":\"kebab-id\",\"title\":\"...\",\"deps\":[\"id\",...]}]}. '
            "ids are kebab-case; deps reference other ids or existing modules."
        )
        try:
            raw = calib["calibrated_call"](prompt)["answer"] if calib else host.call_model(prompt)
        except Exception:
            return None
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:
            return None
        nodes = data.get("nodes") or []
        if not nodes:
            return None
        for n in nodes:
            n.setdefault("deps", [])
            n.setdefault("status", "proposed")
        edges = [{"from": d, "to": n["id"]} for n in nodes for d in n.get("deps", [])]
        return {"goal": goal, "nodes": nodes, "edges": edges, "source": "model"}

    def decompose(goal=None):
        """Decompose the goal into {goal, nodes, edges, source}; persist to goal_graph.json."""
        goal = goal or (host.store.read("goal.txt") or "expand sAGI").strip()
        graph = _model_graph(goal) or _grounded_graph(goal)
        host.store.write_json(_GRAPH_FILE, graph)
        host.log("goal_graph", goal=goal, nodes=len(graph["nodes"]), source=graph["source"])
        return graph

    def graph():
        """The persisted goal graph (decomposing first if none exists)."""
        return host.store.read_json(_GRAPH_FILE, default=None) or decompose()

    def next_module(built=None):
        """The next node whose deps are all satisfied and which isn't built yet, or None."""
        g = graph()
        if built is None:
            built = {n["id"] for n in g["nodes"] if n.get("status") == "built"}
        built = set(built)
        for n in g["nodes"]:
            if n["id"] in ("goal",) or n["id"] in built:
                continue
            if n.get("status") == "built":
                continue
            if all(d in built or d == "goal" for d in n.get("deps", [])):
                return {"title": n.get("title", n["id"]), "id": n["id"], "deps": n.get("deps", [])}
        return None

    host.log("module", step="expand-5", id=MODULE_ID)
    return {"decompose": decompose, "graph": graph, "next_module": next_module}
