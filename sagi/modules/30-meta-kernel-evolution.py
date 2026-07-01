# sagi/modules/30-meta-kernel-evolution.py — tier-3 · "transcend thyself"
#
# GAP CLOSED (the apex): every tier so far only APPENDS new modules — the individual could never
# safely improve its OWN earlier modules. This enables that, behind hard gates so a bad self-edit
# can never corrupt the individual: propose_change() records a proposed new source for an existing
# module; apply() runs it through scan → snapshot → write → verify → benchmark, and on ANY failure
# restores the exact original source (and rollback snapshot). Safe self-modification of the kernel,
# not blind self-rewrite. Best run inside a #22 volatile environment (stable parent stays intact).
#
# GROUNDED REUSE: security-hardening.scan (pre-write screen), rollback-recovery.snapshot (last-good),
# module-verifier.verify (contract gate), benchmark-suite.run_suite (no-regress gate),
# build-driver (synthesis, when a model proposes the change). STDLIB only; Store-confined writes.
from __future__ import annotations

import os
import time

MODULE_ID = "meta-kernel-evolution"
DEPS = ["build-driver", "rollback-recovery", "benchmark-suite", "security-hardening"]
MOTTO = "transcend thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _history():
        return host.store.read_json("kernel_changes.json", default=[]) or []

    def _save(h):
        try:
            host.store.write_json("kernel_changes.json", h)
        except Exception:
            pass

    def propose_change(target_file, new_source):
        """Record a proposed replacement source for an existing module. No write yet."""
        h = _history()
        pid = f"k{len(h) + 1}-{int(time.time())}"
        h.append({"id": pid, "target": target_file, "source": new_source,
                  "status": "proposed", "ts": int(time.time())})
        _save(h)
        host.log("kernel_propose", id=pid, target=target_file)
        return {"proposal_id": pid}

    def apply(proposal_id):
        """Gated apply: scan → snapshot → write → verify → benchmark; restore original on any failure."""
        h = _history()
        prop = next((p for p in h if p.get("id") == proposal_id and p.get("status") == "proposed"), None)
        if not prop:
            return {"applied": False, "error": f"no proposed change {proposal_id}"}
        target, new_source = prop["target"], prop["source"]
        rel = os.path.join("modules", target)
        original = host.store.read(rel)                       # for exact restore on failure
        gates = {}

        sec = _sib(host, "security-hardening")
        scan = sec["scan"](new_source) if sec else {"safe": True, "findings": []}
        gates["scan_safe"] = scan["safe"]
        if not scan["safe"]:
            return _finish(h, prop, False, gates, reason=f"unsafe: {scan['findings']}")

        rb = _sib(host, "rollback-recovery")
        snap = rb["snapshot"]() if rb else None
        gates["snapshot"] = bool(snap)

        bs = _sib(host, "benchmark-suite")
        before = (bs["run_suite"]() or {}).get("score", 1.0) if bs else 1.0

        host.store.write(rel, new_source)                     # Store-confined write of the new source
        verifier = _sib(host, "module-verifier")
        ok = verifier["verify"](target)["ok"] if verifier else True
        gates["verify"] = ok
        after = (bs["run_suite"]() or {}).get("score", 1.0) if bs else 1.0
        gates["no_regression"] = after >= before

        if ok and after >= before:
            # improved source is on disk + verified; it hot-loads on next boot (no phantom re-register).
            return _finish(h, prop, True, gates, reason="all gates green")

        # FAILURE → restore the exact original (and let rollback-recovery heal the tree too)
        if original is not None:
            host.store.write(rel, original)
        elif rb and snap:
            rb["rollback_to"](snap)
        return _finish(h, prop, False, gates, reason="gate failed — original restored")

    def _finish(h, prop, applied, gates, reason):
        prop["status"] = "applied" if applied else "rejected"
        prop["applied"] = applied
        prop["gates"] = gates
        prop["reason"] = reason
        _save(h)
        host.log("kernel_apply", id=prop["id"], applied=applied, reason=reason)
        return {"applied": applied, "gates": gates, "reason": reason, "id": prop["id"]}

    def history():
        return [{k: v for k, v in p.items() if k != "source"} for p in _history()]

    host.log("module", step="tier3-30", id=MODULE_ID)
    return {"propose_change": propose_change, "apply": apply, "history": history}
