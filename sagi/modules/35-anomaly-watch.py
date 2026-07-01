# sagi/modules/35-anomaly-watch.py — tier-4 · "sense danger"
#
# GAP CLOSED: evaluator scores the system and rollback-recovery can heal it, but nothing WATCHES for
# trouble between builds — a regression or an error spike goes unnoticed until someone looks. This is
# the sentinel: it snapshots a healthy baseline, and check() flags anomalies (verified-fraction drop,
# any failed module, an error-event spike in telemetry) — and can trip rollback-recovery on a hard
# regression. Complements the passive scorecard with an active alarm. Baseline persists to anomaly.json.
#
# GROUNDED REUSE: evaluator.evaluate/regress (the metric deltas), telemetry.counters (error spikes),
# rollback-recovery.rollback_to (heal on hard regression). STDLIB only.
from __future__ import annotations

MODULE_ID = "anomaly-watch"
DEPS = ["telemetry", "evaluator", "rollback-recovery"]
MOTTO = "sense danger"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _metrics():
        evalr = _sib(host, "evaluator")
        try:
            return evalr["evaluate"]()["metrics"] if evalr else {}
        except Exception:
            return {}

    def baseline():
        """Capture the current healthy metrics as the reference to watch against."""
        m = _metrics()
        host.store.write_json("anomaly.json", m)
        host.log("anomaly_baseline", **{k: m.get(k) for k in ("verified_fraction", "live_modules")})
        return m

    def check(heal=False):
        """Compare live metrics to the baseline; return anomalies. Optionally heal a hard regression."""
        base = host.store.read_json("anomaly.json", default=None)
        cur = _metrics()
        if base is None:
            base = baseline()
        anomalies = []
        if cur.get("failed_modules", 0) > 0:
            anomalies.append({"kind": "verification_failure", "failed": cur["failed_modules"], "severity": "high"})
        if cur.get("verified_fraction", 1) < base.get("verified_fraction", 1):
            anomalies.append({"kind": "verified_fraction_drop",
                              "from": base.get("verified_fraction"), "to": cur.get("verified_fraction"),
                              "severity": "high"})
        if cur.get("live_modules", 0) < base.get("live_modules", 0):
            anomalies.append({"kind": "modules_lost",
                              "from": base.get("live_modules"), "to": cur.get("live_modules"), "severity": "medium"})
        tel = _sib(host, "telemetry")
        if tel:
            try:
                errs = sum(n for e, n in (tel["counters"]() or {}).items() if "error" in e or "fail" in e)
                if errs > base.get("_errors", 0):
                    anomalies.append({"kind": "error_spike", "errors": errs, "severity": "medium"})
            except Exception:
                pass
        healed = None
        if heal and any(a["severity"] == "high" for a in anomalies):
            rb = _sib(host, "rollback-recovery")
            if rb:
                snap = host.memory.head() if getattr(host, "memory", None) else None
                healed = rb["rollback_to"](snap) if snap else None
        host.log("anomaly_check", anomalies=len(anomalies), healed=bool(healed))
        return {"anomalies": anomalies, "healthy": not anomalies, "healed": healed}

    def watch():
        """Subscribe to verify failures so an anomaly is noticed the moment it happens."""
        host.on("module.verified", lambda p: (check() if p and not (p or {}).get("ok", True) else None))
        return {"watching": True}

    watch()
    host.log("module", step="tier4-35", id=MODULE_ID)
    return {"baseline": baseline, "check": check, "watch": watch}
