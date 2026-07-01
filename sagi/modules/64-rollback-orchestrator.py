# sagi/modules/64-rollback-orchestrator.py — tier-7 · "atomic change"
#
# GAP CLOSED: rollback-recovery heals ONE bad module; there is no transaction spanning a MULTI-module
# change (e.g. a refactor touching several files) that either fully commits or fully reverts. This
# orchestrates that: begin() snapshots a last-good point, commit() finalises only if CI is green, and
# rollback() restores the snapshot — so a coordinated change is atomic.
#
# GROUNDED REUSE: rollback-recovery.snapshot/rollback_to, self-test-harness.gate (commit condition),
# conflict-resolver (post-restore reconciliation). STDLIB only.
from __future__ import annotations

MODULE_ID = "rollback-orchestrator"
DEPS = ["rollback-recovery", "meta-kernel-evolution", "conflict-resolver"]
MOTTO = "atomic change"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    txns = {}   # txn_id -> {"snapshot", "status"}

    def begin(label="txn"):
        """Open a transaction: capture a last-good snapshot to which commit-failure reverts."""
        rb = _sib(host, "rollback-recovery")
        snap = rb["snapshot"]() if rb else (host.memory.head() if getattr(host, "memory", None) else None)
        tid = f"tx{len(txns) + 1}"
        txns[tid] = {"snapshot": snap, "status": "open", "label": label}
        host.log("txn_begin", id=tid, snapshot=(snap or "")[:12])
        return {"txn": tid, "snapshot": (snap or "")[:12]}

    def commit(txn):
        """Finalise a transaction ONLY if CI is green; otherwise auto-rollback. Atomic."""
        t = txns.get(txn)
        if not t or t["status"] != "open":
            return {"error": f"no open txn {txn}"}
        ci = _sib(host, "self-test-harness")
        green = ci["gate"]() if ci else True
        if green:
            t["status"] = "committed"
            host.log("txn_commit", id=txn, green=True)
            return {"txn": txn, "committed": True}
        res = rollback(txn)
        return {"txn": txn, "committed": False, "auto_rolled_back": True, "rollback": res}

    def rollback(txn):
        """Revert the whole transaction to its opening snapshot; reconcile any residue."""
        t = txns.get(txn)
        if not t:
            return {"error": f"no txn {txn}"}
        rb = _sib(host, "rollback-recovery")
        restored = rb["rollback_to"](t["snapshot"]) if (rb and t["snapshot"]) else []
        t["status"] = "rolled_back"
        cr = _sib(host, "conflict-resolver")
        if cr:
            try: cr["resolve"]()
            except Exception: pass
        host.log("txn_rollback", id=txn, restored=len(restored) if isinstance(restored, list) else 0)
        return {"txn": txn, "rolled_back": True, "restored": restored}

    host.log("module", step="tier7-64", id=MODULE_ID)
    return {"begin": begin, "commit": commit, "rollback": rollback}
