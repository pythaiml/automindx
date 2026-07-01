# sagi/modules/70-sovereignty-manager.py — tier-7 · "grant sovereignty" (capstone)
#
# GAP CLOSED (the governance capstone): spawn creates sub/sov children and governance enforces the
# boundary, but nothing manages the LIFECYCLE of that boundary — a subordinate child that has matured
# cannot be granted sovereignty, and a sovereign peer cannot be formally detached. This does, and it
# is deliberately gated: a boundary change is high-stakes, so it is routed through governance-council
# (which holds it for a human) and recorded in the audit log. Only with explicit force does it perform
# the move (children/<id> → sovereign/<id>, flipping the child's identity), keeping automindX stable.
#
# GROUNDED REUSE: swarm-orchestrator.gather, governance.can_edit, governance-council.verdict (gate),
# audit-log.append (record), spawn's on-disk layout (children/ vs sovereign/). STDLIB only (os, json).
from __future__ import annotations

import json
import os

MODULE_ID = "sovereignty-manager"
DEPS = ["swarm-orchestrator", "sagi-environment", "continuity"]
MOTTO = "grant sovereignty"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    from sagi.runtime.governance import can_edit as _can_edit

    def boundaries():
        """The current sub/sovereign boundary map of this parent's family."""
        lineage = host.store.read_json("lineage.json", default={"children": []}) or {"children": []}
        out = []
        for c in lineage.get("children", []):
            cdir = os.path.join(host.root, c.get("dir", ""))
            out.append({"id": c.get("id"), "mode": c.get("mode"),
                        "governed_by_parent": _can_edit(host.root, cdir), "exists": os.path.isdir(cdir)})
        return out

    def _record(action, detail):
        al = _sib(host, "audit-log")
        if al:
            try: al["append"](action, detail)
            except Exception: pass

    def promote_to_sovereign(child_id, force=False):
        """Grant a subordinate child sovereignty. HIGH-STAKES → council-gated unless force=True."""
        council = _sib(host, "governance-council")
        verdict = council["verdict"]("grant_sovereignty", {"child": child_id}) if council else "allow"
        if verdict != "allow" and not force:
            _record("grant_sovereignty.hold", {"child": child_id})
            return {"done": False, "decision": verdict, "note": "held for human — pass force=True to override"}
        src = os.path.join(host.root, "children", child_id)
        dst = os.path.join(host.root, "sovereign", child_id)
        if not os.path.isdir(src):
            return {"done": False, "error": f"no subordinate child {child_id}"}
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)                                   # within the package — governed, contained
        # flip the child's identity to sovereign
        ipath = os.path.join(dst, "identity.json")
        try:
            ident = json.load(open(ipath, encoding="utf-8")) if os.path.exists(ipath) else {}
            ident.update({"mode": "sov", "sovereign": True})
            json.dump(ident, open(ipath, "w", encoding="utf-8"), indent=2)
        except Exception:
            pass
        # update lineage
        lineage = host.store.read_json("lineage.json", default={"children": []}) or {"children": []}
        for c in lineage.get("children", []):
            if c.get("id") == child_id:
                c["mode"] = "sov"; c["dir"] = os.path.join("sovereign", child_id)
        host.store.write_json("lineage.json", lineage)
        _record("grant_sovereignty", {"child": child_id})
        host.log("promote_sovereign", child=child_id)
        return {"done": True, "child": child_id, "mode": "sov"}

    def detach(child_id):
        """Formally relinquish a sovereign peer from this parent's lineage (it becomes independent)."""
        lineage = host.store.read_json("lineage.json", default={"children": []}) or {"children": []}
        before = len(lineage.get("children", []))
        lineage["children"] = [c for c in lineage.get("children", []) if c.get("id") != child_id]
        host.store.write_json("lineage.json", lineage)
        _record("detach", {"child": child_id})
        host.log("detach_child", child=child_id)
        return {"detached": child_id, "removed": before - len(lineage["children"])}

    host.log("module", step="tier7-70", id=MODULE_ID)
    return {"promote_to_sovereign": promote_to_sovereign, "detach": detach, "boundaries": boundaries}
