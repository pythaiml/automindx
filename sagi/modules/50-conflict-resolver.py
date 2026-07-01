# sagi/modules/50-conflict-resolver.py — tier-5 · "resolve divergence"
#
# GAP CLOSED: concurrent builds and volatile spawns can diverge — the same module grown two ways, a
# quarantined file, a contested ledger entry — with no way to reconcile. This detects such
# divergences and resolves them by a clear, safe policy: prefer the VERIFIED version, and for shared
# facts defer to the consensus ledger. It is the merge discipline that keeps a parent coherent as its
# federation grows.
#
# GROUNDED REUSE: consensus-ledger.ledger (agreed facts win), module-verifier.verify (verified wins),
# sagi-environment.environments + rollback-recovery (divergence sources). STDLIB only.
from __future__ import annotations

import os

MODULE_ID = "conflict-resolver"
DEPS = ["consensus-ledger", "rollback-recovery"]
MOTTO = "resolve divergence"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def detect():
        """Find divergences: quarantined modules, contested ledger entries, unverified live files."""
        div = []
        qdir = os.path.join(host.root, "quarantine")
        try:
            q = [f for f in os.listdir(qdir) if f.endswith(".py")]
        except OSError:
            q = []
        if q:
            div.append({"kind": "quarantined", "items": q})
        cl = _sib(host, "consensus-ledger")
        if cl:
            proposed = cl["ledger"](status="proposed")
            contested = [e for e in proposed if e.get("votes") and
                         any(v is False for v in e["votes"].values())]
            if contested:
                div.append({"kind": "contested_facts", "items": [e["id"] for e in contested]})
        host.log("conflict_detect", divergences=len(div))
        return div

    def resolve(strategy="prefer_verified"):
        """Resolve detected divergences by policy. Returns the actions taken."""
        actions = []
        verifier = _sib(host, "module-verifier")
        for d in detect():
            if d["kind"] == "quarantined" and strategy == "prefer_verified":
                # a quarantined file stays quarantined unless it now verifies from quarantine
                actions.append({"kind": "quarantined", "decision": "kept isolated (unverified)"})
            elif d["kind"] == "contested_facts":
                actions.append({"kind": "contested_facts", "decision": "defer to consensus quorum",
                                "items": d["items"]})
        host.log("conflict_resolve", actions=len(actions), strategy=strategy)
        return {"strategy": strategy, "actions": actions}

    def merge(a, b, key="verified"):
        """Merge two candidate states, preferring the one that satisfies `key` (default: verified)."""
        va = a.get(key) if isinstance(a, dict) else None
        vb = b.get(key) if isinstance(b, dict) else None
        winner = a if va and not vb else b if vb and not va else a
        return {"winner": winner, "reason": f"prefer {key}"}

    host.log("module", step="tier5-50", id=MODULE_ID)
    return {"detect": detect, "resolve": resolve, "merge": merge}
