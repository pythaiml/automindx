# sagi/modules/40-continuity.py — tier-4 · "endure across time" (capstone)
#
# GAP CLOSED (the capstone): all the durability primitives exist — savepoints, gitmind, lineage,
# consolidation, deploy/restore — but nothing binds them into a single CONTINUITY of self across
# restarts and generations. This is that thread: checkpoint() takes a durable continuity point
# (consolidate memory → savepoint milestone → record lineage), resume() reconstitutes state after a
# restart, and identity_thread() renders the individual's continuous identity across its spawned
# generations. This is what lets a stable parent persist while volatile spawns come and go.
#
# GROUNDED REUSE: persistence-deploy.restore/package, episodic-consolidation.consolidate/milestone,
# lineage-federation.federate/roster, savepoint.save_point. STDLIB only.
from __future__ import annotations

MODULE_ID = "continuity"
DEPS = ["lineage-federation", "persistence-deploy", "episodic-consolidation"]
MOTTO = "endure across time"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def checkpoint(label="continuity"):
        """A durable continuity point: consolidate memory → savepoint milestone → record lineage."""
        ec = _sib(host, "episodic-consolidation")
        consolidated = None
        if ec:
            try:
                consolidated = bool(ec["consolidate"]())
            except Exception:
                consolidated = None
        commit = None
        try:
            from sagi.runtime.savepoint import save_point
            commit = save_point(host.root, label=label, rage=False)["commit"][:12]
        except Exception as e:
            host.log("continuity_savepoint_error", detail=str(e)[:120])
        fed = _sib(host, "lineage-federation")
        roster = []
        try:
            roster = fed["roster"]() if fed else []
        except Exception:
            roster = []
        host.log("continuity_checkpoint", commit=commit, consolidated=bool(consolidated), generations=len(roster))
        return {"label": label, "savepoint": commit, "consolidated": consolidated,
                "generations": len(roster)}

    def resume():
        """Reconstitute state after a restart: restore any missing modules + report memory depth."""
        pd = _sib(host, "persistence-deploy")
        restored = pd["restore"]() if pd else {"restored": 0}
        mem = getattr(host, "memory", None)
        depth = len(mem.log()) if mem is not None else 0
        reg = getattr(host, "registry", None)
        return {"restored": restored.get("restored", 0), "memory_moments": depth,
                "live": len(reg["list"]()) if reg else 0, "resumed": True}

    def identity_thread():
        """The continuous identity across generations: this individual + its lineage of spawns."""
        ident = host.store.read_json("identity.json", default={}) or {}
        fed = _sib(host, "lineage-federation")
        roster = []
        try:
            roster = fed["roster"]() if fed else []
        except Exception:
            roster = []
        mem = getattr(host, "memory", None)
        milestones = len(mem.global_log()) if mem is not None else 0
        return {
            "id": ident.get("id") or getattr(host, "identity_id", "sagi"),
            "grown_from": ident.get("grown_from", "Savante persona (automindX)"),
            "generations": [{"id": c.get("id"), "mode": c.get("mode")} for c in roster],
            "milestones": milestones,     # gitmind GLOBAL chain = the individual's remembered expansions
        }

    host.log("module", step="tier4-40", id=MODULE_ID)
    return {"checkpoint": checkpoint, "resume": resume, "identity_thread": identity_thread}
