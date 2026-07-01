# sagi/modules/58-metrics-dashboard.py — tier-6 · "show thy vitals"
#
# GAP CLOSED: evaluator gives a point-in-time scorecard and telemetry gives raw counters, but there
# is no aggregated, UI-ready feed of the individual's vitals over time. This assembles one: snapshot()
# unifies the live metrics, series() appends timestamped snapshots to a rolling history, and widgets()
# emits UI-ready blocks (gauge/counter/list) the console can render.
#
# GROUNDED REUSE: evaluator.evaluate, telemetry.counters, introspection.map/health. STDLIB only.
from __future__ import annotations

import time

MODULE_ID = "metrics-dashboard"
DEPS = ["evaluator", "telemetry", "introspection"]
MOTTO = "show thy vitals"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def snapshot():
        """Unified vitals: scorecard + event counters + module/health counts, timestamped."""
        evalr = _sib(host, "evaluator"); tel = _sib(host, "telemetry"); intro = _sib(host, "introspection")
        metrics = (evalr["evaluate"]()["metrics"] if evalr else {})
        counters = (tel["counters"]() if tel else {})
        health = (intro["health"]() if intro else {"verified": [], "failed": []})
        return {"ts": int(time.time()), "metrics": metrics,
                "events": sum(counters.values()) if counters else 0,
                "verified": len(health.get("verified", [])), "failed": len(health.get("failed", []))}

    def series(append=True, limit=200):
        """Append the current snapshot to a rolling time series and return it."""
        hist = host.store.read_json("dashboard.json", default=[]) or []
        if append:
            hist.append(snapshot())
            hist = hist[-limit:]
            host.store.write_json("dashboard.json", hist)
        return hist

    def widgets():
        """UI-ready widget blocks the console can render directly."""
        s = snapshot()
        m = s["metrics"]
        return [
            {"type": "gauge", "label": "verified", "value": m.get("verified_fraction", 0), "max": 1.0},
            {"type": "counter", "label": "live modules", "value": m.get("live_modules", 0)},
            {"type": "counter", "label": "API surface", "value": m.get("api_surface", 0)},
            {"type": "counter", "label": "memory moments", "value": m.get("memory_moments", 0)},
            {"type": "counter", "label": "events", "value": s["events"]},
            {"type": "status", "label": "health", "value": "ok" if s["failed"] == 0 else f"{s['failed']} failing"},
        ]

    host.log("module", step="tier6-58", id=MODULE_ID)
    return {"snapshot": snapshot, "series": series, "widgets": widgets}
