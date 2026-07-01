# sagi/modules/10-swarm-orchestrator.py — expand-10 · "multiply thyself"
#
# The individual can already spawn children (spawn.py: sub/sov), govern edits across the tree
# (governance.py), and snapshot itself (savepoint.py) — all real and already exercised (lineage
# shows sovereign Athena → sub Aegis). What was missing is an orchestrator that uses them to
# expand HORIZONTALLY: split a curriculum into branches, delegate each to a child (respecting the
# governance boundary), and gather the branches back by reading each child's lineage + savepoint.
# This is the top of the expand-sAGI stack: growth that scales out, not just up.
from __future__ import annotations

import os

MODULE_ID = "swarm-orchestrator"
DEPS = ["curriculum", "tool-registry"]
MOTTO = "multiply thyself"


def _sib(host, mod_id):
    reg = getattr(host, "registry", None)
    return (reg["get"](mod_id) or {}).get("handle") if reg else None


def activate(host):
    from sagi.runtime.spawn import spawn as _spawn_fn
    from sagi.runtime.governance import can_edit as _can_edit

    def delegate(branch, mode="sub", individual=""):
        """Spawn a child sAGI to own a curriculum branch. Governance-checked for 'sub' children."""
        res = _spawn_fn(host.root, branch, mode=mode, individual=individual, boot_child=True)
        # a subordinate child must be editable by this parent (sovereign children are not)
        if mode == "sub" and not _can_edit(host.root, res["dir"]):
            host.log("delegate_governance_warn", child=res["child_id"])
        host.log("delegate", branch=branch, child=res["child_id"], mode=mode)
        host.emit("swarm.delegated", {"branch": branch, "child": res["child_id"], "mode": mode})
        return res

    def gather():
        """Collect every child branch: identity, mode, module count, latest savepoint (if any)."""
        lineage = host.store.read_json("lineage.json", default={"children": []}) or {}
        out = []
        for c in lineage.get("children", []):
            cdir = os.path.join(host.root, c.get("dir", ""))
            ident = host.store.read_json(os.path.join(c.get("dir", ""), "identity.json"), default={}) or {}
            mods = []
            try:
                mods = [f for f in os.listdir(os.path.join(cdir, "modules")) if f.endswith((".md", ".py"))]
            except OSError:
                pass
            saves = []
            try:
                saves = sorted(f for f in os.listdir(os.path.join(cdir, "savepoints")) if f.endswith(".md"))
            except OSError:
                pass
            out.append({"id": c.get("id"), "mode": c.get("mode"),
                        "sovereign": ident.get("sovereign", c.get("mode") == "sov"),
                        "modules": len(mods), "latest_savepoint": saves[-1] if saves else None,
                        "editable_by_parent": _can_edit(host.root, cdir)})
        return out

    def map_curriculum():
        """Assign curriculum branches to (proposed) children — a plan, without spawning."""
        cur = _sib(host, "curriculum")
        status = cur["status"]() if cur else {"remaining": [], "built": []}
        branches = status.get("remaining") or status.get("built", [])
        return {"branches": [{"branch": b, "proposed_child": f"branch-{b}", "mode": "sub"}
                             for b in branches],
                "note": "call delegate(branch) to actually spawn a child for a branch"}

    host.log("module", step="expand-10", id=MODULE_ID)
    return {"delegate": delegate, "gather": gather, "map_curriculum": map_curriculum}
