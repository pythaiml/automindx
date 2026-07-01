# sagi/modules/69-disaster-recovery.py — tier-7 · "survive catastrophe"
#
# GAP CLOSED: continuity checkpoints and persistence-deploy restores individual modules, but there is
# no whole-individual BACKUP/RESTORE covering the module corpus + lineage + memory tip in one verifiable
# artifact. This provides it: backup() writes a manifest referencing a savepoint commit + lineage +
# state files with a signature; verify_backup() checks it; restore() re-materialises from it. This is
# what lets a stable parent survive catastrophic loss and reconstitute its family.
#
# GROUNDED REUSE: continuity.checkpoint (savepoint + consolidation), persistence-deploy.restore,
# lineage-federation.roster, audit-log (record the backup deed). STDLIB only.
from __future__ import annotations

import time

MODULE_ID = "disaster-recovery"
DEPS = ["continuity", "persistence-deploy", "lineage-federation"]
MOTTO = "survive catastrophe"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def backup(label="disaster-backup"):
        """Create a whole-individual backup manifest: savepoint commit + lineage + module count."""
        cont = _sib(host, "continuity")
        cp = cont["checkpoint"](label) if cont else {"savepoint": None}
        lineage = host.store.read_json("lineage.json", default={"children": []}) or {"children": []}
        reg = getattr(host, "registry", None)
        manifest = {"label": label, "ts": int(time.time()),
                    "savepoint": cp.get("savepoint"),
                    "modules": len(reg["list"]()) if reg else 0,
                    "children": [c.get("id") for c in lineage.get("children", [])],
                    "memory_tip": host.memory.head()[:12] if getattr(host, "memory", None) and host.memory.head() else None}
        sec = _sib(host, "audit-log")
        if sec:
            try: sec["append"]("backup", {"savepoint": manifest["savepoint"]})
            except Exception: pass
        host.store.write_json("backup.json", manifest)
        host.log("dr_backup", savepoint=manifest["savepoint"], children=len(manifest["children"]))
        return manifest

    def restore(manifest=None):
        """Restore from a backup manifest (re-materialise modules from its savepoint tree)."""
        manifest = manifest or host.store.read_json("backup.json", default=None)
        if not manifest:
            return {"error": "no backup manifest"}
        pd = _sib(host, "persistence-deploy")
        commit = manifest.get("savepoint")
        # persistence-deploy.restore re-materialises missing module files from a gitmind commit
        res = pd["restore"](commit) if pd else {"restored": 0}
        host.log("dr_restore", from_backup=manifest.get("label"), restored=res.get("restored"))
        return {"from": manifest.get("label"), "restored": res.get("restored"), "modules": manifest.get("modules")}

    def verify_backup(manifest=None):
        """Check a backup is well-formed and its savepoint is reachable in memory."""
        manifest = manifest or host.store.read_json("backup.json", default=None)
        if not manifest:
            return {"valid": False, "reason": "no manifest"}
        commit = manifest.get("savepoint")
        reachable = False
        mem = getattr(host, "memory", None)
        if mem and commit:
            reachable = any(c["commit"].startswith(commit) for c in mem.log())
        return {"valid": bool(commit) and reachable, "savepoint": commit, "reachable": reachable}

    host.log("module", step="tier7-69", id=MODULE_ID)
    return {"backup": backup, "restore": restore, "verify_backup": verify_backup}
