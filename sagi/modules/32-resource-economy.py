# sagi/modules/32-resource-economy.py — tier-4 · "spend wisely"
#
# GAP CLOSED: growth consumes resources (model calls, build steps, tool invocations) but nothing
# accounts for or bounds them, so an autonomous loop could run unbounded. This is a simple economy:
# set a budget per resource key, spend() against it (policy-gated — an over-budget spend is denied),
# and read the ledger. It mirrors the Ataraxia circuit-breaker idea (sagi_build) but as a live,
# per-key budget any module can respect. Persisted to economy.json.
#
# GROUNDED REUSE: telemetry.counters (observed usage seeds balances), policy-guard.check (deny
# over-budget), outcome-learning (spend efficiency can be rewarded). STDLIB only.
from __future__ import annotations

MODULE_ID = "resource-economy"
DEPS = ["telemetry", "outcome-learning", "policy-guard"]
MOTTO = "spend wisely"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    econ = host.store.read_json("economy.json", default={"budgets": {}, "spent": {}}) or {"budgets": {}, "spent": {}}

    def _persist():
        try:
            host.store.write_json("economy.json", econ)
        except Exception:
            pass

    def budget(key, limit):
        """Set a spending limit for a resource key (e.g. 'model_calls', 'build_steps')."""
        econ["budgets"][key] = float(limit)
        econ["spent"].setdefault(key, 0.0)
        _persist()
        return {"key": key, "limit": float(limit), "remaining": float(limit) - econ["spent"][key]}

    def spend(key, amount=1.0):
        """Charge `amount` to a key. Denied (policy) if it would exceed the budget."""
        amount = float(amount)
        spent = econ["spent"].get(key, 0.0)
        limit = econ["budgets"].get(key)
        would = spent + amount
        if limit is not None and would > limit:
            guard = _sib(host, "policy-guard")
            reason = f"budget exceeded for {key}: {would} > {limit}"
            if guard:
                try: guard["check"]("spend", {"key": key, "would": would, "limit": limit})
                except Exception: pass
            host.log("spend_denied", key=key, would=would, limit=limit)
            return {"ok": False, "reason": reason, "remaining": max(0.0, limit - spent)}
        econ["spent"][key] = would
        _persist()
        return {"ok": True, "spent": would, "remaining": (limit - would) if limit is not None else None}

    def ledger():
        """Current balances: {key: {limit, spent, remaining}}."""
        out = {}
        for key in set(list(econ["budgets"]) + list(econ["spent"])):
            limit = econ["budgets"].get(key)
            spent = econ["spent"].get(key, 0.0)
            out[key] = {"limit": limit, "spent": spent,
                        "remaining": (limit - spent) if limit is not None else None}
        return out

    # seed balances from observed telemetry so the ledger reflects real activity
    tel = _sib(host, "telemetry")
    if tel:
        try:
            for ev, n in (tel["counters"]() or {}).items():
                econ["spent"].setdefault(ev, float(n))
            _persist()
        except Exception:
            pass

    host.log("module", step="tier4-32", id=MODULE_ID)
    return {"budget": budget, "spend": spend, "ledger": ledger}
