# sagi/modules/67-resource-scheduler.py — tier-7 · "allocate fairly"
#
# GAP CLOSED: resource-economy tracks budgets and scheduler runs jobs, but nothing ALLOCATES bounded
# resources across competing work — the Self-Optimizing Resource Allocation the specs describe. This
# joins them: allocate() reserves budget for a work kind, schedule_work() registers a budgeted
# periodic job, and utilization() reports how much of each budget is consumed — with outcome-learning
# steering more budget to work that pays off.
#
# GROUNDED REUSE: resource-economy.budget/spend/ledger, scheduler.schedule/tick, outcome-learning.weights.
from __future__ import annotations

MODULE_ID = "resource-scheduler"
DEPS = ["resource-economy", "scheduler", "outcome-learning"]
MOTTO = "allocate fairly"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def allocate(kind, amount):
        """Reserve a budget for a work kind (delegates to resource-economy)."""
        econ = _sib(host, "resource-economy")
        return econ["budget"](f"work:{kind}", amount) if econ else {"error": "resource-economy not live"}

    def schedule_work(kind, every=1, cost=1.0):
        """Register a budgeted periodic job: it runs only while its work-budget has room."""
        sched = _sib(host, "scheduler")
        econ = _sib(host, "resource-economy")
        # ensure a budget exists so the job is bounded
        if econ and f"work:{kind}" not in (econ["ledger"]() or {}):
            econ["budget"](f"work:{kind}", every * cost * 10)
        job = sched["schedule"](kind if kind in ("autobuild", "benchmark", "consolidate") else "benchmark", every) if sched else None
        return {"kind": kind, "job": job, "cost": cost}

    def utilization():
        """How much of each work-budget is consumed, and which work outcome-learning favours."""
        econ = _sib(host, "resource-economy")
        ledger = econ["ledger"]() if econ else {}
        work = {k: v for k, v in ledger.items() if k.startswith("work:")}
        ol = _sib(host, "outcome-learning")
        weights = (ol["weights"]() if ol else {}).get("weights", {})
        return {"budgets": work,
                "favoured": sorted(weights, key=lambda k: -weights.get(k, 0))[:5]}

    host.log("module", step="tier7-67", id=MODULE_ID)
    return {"allocate": allocate, "schedule_work": schedule_work, "utilization": utilization}
