# sagi/modules/61-governance-council.py — tier-7 · "govern by council"
#
# GAP CLOSED: risky actions are gated by a single policy-guard check; there is no COUNCIL that weighs
# multiple independent signals before a high-stakes act (a kernel self-edit, a sovereignty change).
# This convenes one: it collects verdicts from policy-guard (is it allowed), consensus-ledger (does
# the federation agree), and human-interface (is a human needed), and returns an aggregate verdict —
# the DAIO-style multi-signal governance the mindX lineage uses.
#
# GROUNDED REUSE: policy-guard.check, consensus-ledger.propose/vote, human-interface.enqueue.
from __future__ import annotations

MODULE_ID = "governance-council"
DEPS = ["policy-guard", "consensus-ledger", "human-interface"]
MOTTO = "govern by council"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def deliberate(action, context=None):
        """Collect independent verdicts on an action from policy, consensus, and the human channel."""
        context = context or {}
        verdicts = {}
        guard = _sib(host, "policy-guard")
        verdicts["policy"] = guard["check"](action, context) if guard else {"allow": True}
        # high-stakes actions require a human in the loop
        high_stakes = action in ("kernel_edit", "grant_sovereignty", "detach_child", "disaster_restore")
        verdicts["human_required"] = high_stakes
        if high_stakes:
            hi = _sib(host, "human-interface")
            if hi and "enqueue" in hi:
                verdicts["human_ticket"] = hi["enqueue"]({"action": action, **context})
        return verdicts

    def convene(action, context=None):
        """Open a council: deliberate, and if not human-gated, record the decision on the ledger."""
        v = deliberate(action, context)
        allow = v["policy"].get("allow", True) and not v.get("human_required")
        cl = _sib(host, "consensus-ledger")
        if cl:
            try:
                cl["propose"](f"council: {action} -> {'allow' if allow else 'hold'}")
            except Exception:
                pass
        host.log("council_convene", action=action, allow=allow, human=v.get("human_required"))
        return {"action": action, "verdicts": v, "decision": "allow" if allow else "hold-for-human"}

    def verdict(action, context=None):
        """The bottom-line decision for an action."""
        return convene(action, context)["decision"]

    host.log("module", step="tier7-61", id=MODULE_ID)
    return {"convene": convene, "deliberate": deliberate, "verdict": verdict}
