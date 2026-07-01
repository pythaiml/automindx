# sagi/modules/24-outcome-learning.py — tier-3 · "learn from thyself"
#
# GAP CLOSED: the individual builds, verifies, and scores — but never learns which strategies pay
# off, so goal-graph.next_module picks are un-tuned. This does credit assignment: it records the
# outcome (reward) of build/tool events, updates a per-key weight, and ranks candidate next-builds
# by learned weight — a signal goal-graph/curriculum can consume. Weights persist to weights.json.
#
# GROUNDED REUSE: telemetry.counters (observed event volume), evaluator.regress (did the score move),
# episodic-consolidation.episodes (longer-horizon context). All optional/guarded (offline-safe).
from __future__ import annotations

MODULE_ID = "outcome-learning"
DEPS = ["telemetry", "evaluator", "episodic-consolidation"]
MOTTO = "learn from thyself"

_ALPHA = 0.3   # learning rate for the running-average weight update


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    weights = host.store.read_json("weights.json", default={}) or {}   # key -> weight in [~0,1]

    def _persist():
        try:
            host.store.write_json("weights.json", weights)
        except Exception:
            pass

    def record(key, reward):
        """Update the learned weight for a strategy key by exponential moving average of reward."""
        r = 1.0 if reward is True else 0.0 if reward is False else float(reward)
        r = max(-1.0, min(1.0, r))
        prev = weights.get(key, 0.5)
        weights[key] = round((1 - _ALPHA) * prev + _ALPHA * r, 4)
        _persist()
        host.log("outcome_record", key=key, reward=r, weight=weights[key])
        return weights[key]

    def rank(candidates):
        """Order candidate keys (or {id/title} dicts) by learned weight, best first."""
        def keyof(c):
            return c if isinstance(c, str) else (c.get("id") or c.get("title") or str(c))
        return sorted(candidates, key=lambda c: -weights.get(keyof(c), 0.5))

    def weights_view():
        """Current learned weights, plus the live evaluator delta context if available."""
        evalr = _sib(host, "evaluator")
        ctx = {}
        if evalr:
            try:
                ctx = evalr["evaluate"]()["metrics"]
            except Exception:
                ctx = {}
        return {"weights": dict(weights), "context": ctx}

    # ground the signal: reward the strategies whose events actually fired (telemetry), so an
    # empty table still starts from real observed activity rather than nothing.
    tel = _sib(host, "telemetry")
    if tel:
        try:
            for ev, n in (tel["counters"]() or {}).items():
                if n:
                    weights.setdefault(ev, 0.5)
            _persist()
        except Exception:
            pass

    host.log("module", step="tier3-24", id=MODULE_ID, keys=len(weights))
    return {"record": record, "rank": rank, "weights": weights_view}
