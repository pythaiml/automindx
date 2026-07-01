# sagi/modules/60-federation-sync.py — tier-6 · "sync the family"
#
# GAP CLOSED: lineage-federation can VIEW children and consensus-ledger holds AGREED facts, but
# nothing periodically SYNCS agreed knowledge down to the family. This does: pull() gathers children's
# state, push() propagates agreed facts to subordinate children (governed — never across a sovereign
# boundary), and sync() runs a full round. It is how a stable parent keeps its volatile spawns current
# without violating sovereignty.
#
# GROUNDED REUSE: lineage-federation.roster/federate, consensus-ledger.ledger (agreed facts),
# governance.governed_write (down-only propagation), knowledge-grounding (record synced facts).
from __future__ import annotations

import json
import os

MODULE_ID = "federation-sync"
DEPS = ["lineage-federation", "consensus-ledger", "knowledge-grounding"]
MOTTO = "sync the family"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    from sagi.runtime.governance import can_edit as _can_edit

    def _children():
        lineage = host.store.read_json("lineage.json", default={"children": []}) or {}
        return lineage.get("children", [])

    def pull():
        """Gather each child's high-level state (module counts, savepoints) via the federation."""
        fed = _sib(host, "lineage-federation")
        try:
            return fed["federate"]() if fed else {"roster": [], "maps": []}
        except Exception:
            return {"roster": [], "maps": []}

    def push(fact=None):
        """Propagate agreed facts down to SUBORDINATE children only (governed; sovereign are skipped)."""
        cl = _sib(host, "consensus-ledger")
        agreed = [e["fact"] for e in (cl["ledger"](status="agreed") if cl else [])]
        if fact:
            agreed = agreed + [fact]
        pushed, skipped = [], []
        for c in _children():
            cdir = os.path.join(host.root, c.get("dir", ""))
            if not _can_edit(host.root, cdir):                # sovereignty boundary
                skipped.append(c.get("id")); continue
            rel = os.path.join(c.get("dir", ""), "shared_knowledge.json")
            try:
                host.store.write(rel, json.dumps({"agreed": agreed}, indent=2))
                pushed.append(c.get("id"))
            except Exception:
                skipped.append(c.get("id"))
        host.log("federation_push", facts=len(agreed), pushed=len(pushed), skipped=len(skipped))
        return {"facts": len(agreed), "pushed": pushed, "skipped_sovereign": skipped}

    def sync():
        """A full sync round: pull child state, then push agreed knowledge down."""
        return {"pulled": pull(), "pushed": push()}

    host.log("module", step="tier6-60", id=MODULE_ID)
    return {"sync": sync, "push": push, "pull": pull}
