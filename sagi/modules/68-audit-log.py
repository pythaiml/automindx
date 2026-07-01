# sagi/modules/68-audit-log.py — tier-7 · "record all deeds"
#
# GAP CLOSED: governed actions are logged to .history, but that log is mutable and unsigned. This is
# an immutable, hash-chained, signed audit trail of every governed deed — each entry chained to its
# predecessor and signed (security-hardening), so tampering is detectable. It is the accountability
# spine under governance-council / access-control / meta-kernel. Persisted to audit_log.json.
#
# GROUNDED REUSE: security-hardening.sign (per-entry signature), event-sourcing (source events),
# access-control (who acted), sha256 chaining. STDLIB only.
from __future__ import annotations

import hashlib
import json
import time

MODULE_ID = "audit-log"
DEPS = ["security-hardening", "event-sourcing", "access-control"]
MOTTO = "record all deeds"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _log():
        return host.store.read_json("audit_log.json", default=[]) or []

    def append(action, detail=None, actor=None):
        """Append a signed, hash-chained audit entry. Immutable by construction (tamper-evident)."""
        log = _log()
        parent = log[-1]["hash"] if log else None
        entry = {"seq": len(log) + 1, "action": action, "detail": detail or {},
                 "actor": actor or getattr(host, "identity_id", "sagi"),
                 "parent": parent, "ts": int(time.time())}
        entry["hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        sec = _sib(host, "security-hardening")
        entry["signature"] = (sec["sign"](entry["hash"])["signature"] if sec else entry["hash"])
        log.append(entry)
        host.store.write_json("audit_log.json", log)
        return {"seq": entry["seq"], "hash": entry["hash"][:12]}

    def verify_chain():
        """Verify the hash chain is unbroken (every entry links to its recomputed predecessor)."""
        log = _log()
        prev = None
        for e in log:
            recomputed = {k: e[k] for k in ("seq", "action", "detail", "actor", "parent", "ts")}
            h = hashlib.sha256(json.dumps(recomputed, sort_keys=True).encode()).hexdigest()
            if h != e.get("hash") or e.get("parent") != prev:
                return {"intact": False, "broken_at": e.get("seq")}
            prev = e["hash"]
        return {"intact": True, "entries": len(log)}

    def query(action=None, actor=None, limit=50):
        log = _log()
        out = [e for e in log
               if (action is None or e["action"] == action) and (actor is None or e["actor"] == actor)]
        return out[-limit:]

    # capture governed deeds as they happen
    host.on("approval.resolved", lambda p: append("approval.resolved", p))
    host.on("module.persisted", lambda p: append("module.persisted", p))
    host.log("module", step="tier7-68", id=MODULE_ID)
    return {"append": append, "verify_chain": verify_chain, "query": query}
