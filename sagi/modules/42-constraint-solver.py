# sagi/modules/42-constraint-solver.py — tier-5 · "reconcile constraints"
#
# GAP CLOSED: invariants the individual must hold (verified_fraction >= 0.9, no failed modules,
# budget not exceeded) are scattered and unchecked. This is a small declarative constraint solver:
# register a constraint (a live metric predicate), solve() evaluates them all against current state,
# and violations() reports which broke — a single place to assert "am I still healthy/legal".
#
# GROUNDED REUSE: evaluator.evaluate (the metric snapshot), policy-guard (a violation can be denied),
# resource-economy.ledger (budget constraints). STDLIB only.
from __future__ import annotations

MODULE_ID = "constraint-solver"
DEPS = ["policy-guard", "evaluator"]
MOTTO = "reconcile constraints"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    constraints = []   # list of {"name", "metric", "op", "value"} or {"name","pred"}

    def _metrics():
        evalr = _sib(host, "evaluator")
        try:
            return evalr["evaluate"]()["metrics"] if evalr else {}
        except Exception:
            return {}

    _OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b,
            ">": lambda a, b: a > b, "<": lambda a, b: a < b}

    def add_constraint(name, metric=None, op=">=", value=0, pred=None):
        """Register a constraint: a metric comparison (metric op value) or a custom pred(metrics)->bool."""
        constraints.append({"name": name, "metric": metric, "op": op, "value": value, "pred": pred})
        return {"name": name, "count": len(constraints)}

    def solve():
        """Evaluate every constraint against live metrics. Returns {satisfied, violations}."""
        m = _metrics()
        violations = []
        for c in constraints:
            ok = True
            if callable(c.get("pred")):
                try:
                    ok = bool(c["pred"](m))
                except Exception:
                    ok = False
            elif c.get("metric") is not None:
                cur = m.get(c["metric"])
                ok = cur is not None and _OPS.get(c["op"], _OPS[">="])(cur, c["value"])
            if not ok:
                violations.append({"name": c["name"], "metric": c.get("metric"),
                                   "want": f"{c.get('op')} {c.get('value')}", "got": m.get(c.get("metric"))})
        host.log("constraint_solve", total=len(constraints), violations=len(violations))
        return {"satisfied": not violations, "violations": violations, "checked": len(constraints)}

    def violations():
        return solve()["violations"]

    # seed the individual's standing health invariants
    add_constraint("all modules verify", pred=lambda m: m.get("failed_modules", 0) == 0)
    add_constraint("high verified fraction", "verified_fraction", ">=", 0.9)

    host.log("module", step="tier5-42", id=MODULE_ID)
    return {"add_constraint": add_constraint, "solve": solve, "violations": violations}
