# sagi/modules/20-lineage-federation.py — tier-2 · "unite thyselves"
#
# Gap it closes: 10-swarm-orchestrator can spawn and GATHER a lineage (parent → children/peers),
# and every individual can savepoint, introspect (18-wire-gateway /map) and consolidate its memory
# (15) — but nothing unifies the tree into ONE coherent federation view, nor moves knowledge along
# it while honouring the sovereignty boundary. This module federates: it reads the roster, aggregates
# each member's map + latest savepoint, and syncs selected knowledge DOWNWARD to subordinate children
# only — governance refuses to cross a `sovereign/` segment, so a sovereign peer is strictly read-only.
#
# Grounded reuse (nothing reinvented):
#   • swarm-orchestrator.gather()            — the live roster (id, mode, sovereign, modules,
#     latest_savepoint, editable_by_parent) built from lineage.json (10-swarm-orchestrator.py).
#   • runtime/governance.can_edit / governed_write — the authority check + boundary-safe cross-package
#     write; can_edit is False the moment a path crosses a `sovereign/` segment, so sync stays downward.
#   • wire-gateway.handle("GET","/map")      — this individual's own live self-map (18); folded into the
#     federation as the parent's contribution. Degrades to None when the sibling is absent.
#   • episodic-consolidation.episodes()      — the parent's consolidated semantic memory (15), attached
#     to the self map so the federation view carries memory, not just structure.
#   • runtime/spawn lineage.json layout + host.store.read — the on-disk children[] and their savepoints/.
#
# New logic (conjecture, kept small): cross-individual knowledge-merge semantics — constrained to
# parent→subordinate, namespaced under `federated/` in the child so it never clobbers the child's own
# files, and refused (not raised) across any sovereign boundary. Merge policy beyond copy is the caller's.
from __future__ import annotations

import os

MODULE_ID = "lineage-federation"
DEPS = ["swarm-orchestrator", "episodic-consolidation", "wire-gateway"]
MOTTO = "unite thyselves"

_FED_PREFIX = "federated"     # where synced knowledge lands inside a subordinate child's package


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def _child_dir(host, child):
    """Absolute directory of a lineage child from its (parent-relative) 'dir'."""
    return os.path.join(host.root, child.get("dir", ""))


def _read_children(host):
    """The raw lineage children[] (via spawn's lineage.json layout). Never raises."""
    lineage = host.store.read_json("lineage.json", default={"children": []}) or {}
    kids = lineage.get("children", [])
    return kids if isinstance(kids, list) else []


def activate(host):
    # governance is the sovereignty boundary — imported here (never at import time, never raising activate)
    try:
        from sagi.runtime.governance import can_edit as _can_edit, governed_write as _governed_write
    except Exception:                                   # offline probe: keep the module inert-safe
        _can_edit = lambda a, t: os.path.realpath(a) == os.path.realpath(t)  # noqa: E731
        _governed_write = None

    def roster():
        """The lineage roster: one entry per child/peer.

        Prefers swarm-orchestrator.gather() (id, mode, sovereign, modules, latest_savepoint,
        editable_by_parent); falls back to the raw lineage.json children[] when the sibling is
        absent (offline probe). Never raises.
        """
        swarm = _sib(host, "swarm-orchestrator")
        if swarm and callable(swarm.get("gather")):
            try:
                return swarm["gather"]() or []
            except Exception as e:
                host.log("roster_gather_failed", detail=str(e)[:160])
        # fallback: derive a minimal roster straight from lineage.json
        out = []
        for c in _read_children(host):
            out.append({"id": c.get("id"), "mode": c.get("mode"),
                        "sovereign": c.get("mode") == "sov",
                        "editable_by_parent": _can_edit(host.root, _child_dir(host, c))})
        return out

    def _self_map():
        """This individual's own contribution: its live self-map (wire-gateway) + memory (episodic)."""
        wg = _sib(host, "wire-gateway")
        smap = None
        if wg and callable(wg.get("handle")):
            try:
                resp = wg["handle"]("GET", "/map")
                if isinstance(resp, dict):
                    smap = resp.get("json")
            except Exception as e:
                host.log("federate_selfmap_failed", detail=str(e)[:160])
        ep = _sib(host, "episodic-consolidation")
        episodes = []
        if ep and callable(ep.get("episodes")):
            try:
                episodes = ep["episodes"]() or []
            except Exception:
                episodes = []
        return {"id": host.identity_id, "map": smap, "episodes": len(episodes)}

    def _child_savepoint(host, child):
        """Latest savepoint of a child: its filename + Markdown text (read-only, never raises)."""
        cdir = _child_dir(host, child)
        sp_dir = os.path.join(cdir, "savepoints")
        try:
            files = sorted(f for f in os.listdir(sp_dir) if f.endswith(".md"))
        except OSError:
            return {"latest": None, "text": None}
        if not files:
            return {"latest": None, "text": None}
        latest = files[-1]
        try:
            with open(os.path.join(sp_dir, latest), encoding="utf-8") as f:
                text = f.read()
        except OSError:
            text = None
        return {"latest": latest, "text": text}

    def federate():
        """Unify the lineage into one view: {roster, maps, savepoints}.

        - roster: swarm-orchestrator.gather() (or the lineage.json fallback).
        - maps:   {'self': this individual's live map + memory, 'children': {id: light structural map}}.
        - savepoints: {id: {latest, text}} — each child's most recent shareable savepoint from disk.
        Read-only aggregation: it never writes and never crosses a sovereignty boundary. Never raises.
        """
        members = roster()
        children_maps = {}
        savepoints = {}
        for c in members:
            cid = c.get("id")
            if not cid:
                continue
            children_maps[cid] = {
                "mode": c.get("mode"),
                "sovereign": c.get("sovereign", c.get("mode") == "sov"),
                "modules": c.get("modules"),
                "editable_by_parent": c.get("editable_by_parent"),
            }
            # locate the on-disk child record so we can read its savepoints/
            rec = next((k for k in _read_children(host) if k.get("id") == cid), None)
            savepoints[cid] = _child_savepoint(host, rec) if rec else {"latest": None, "text": None}
        view = {"roster": members,
                "maps": {"self": _self_map(), "children": children_maps},
                "savepoints": savepoints}
        host.log("federate", members=len(members),
                 savepoints=sum(1 for v in savepoints.values() if v.get("latest")))
        host.emit("lineage.federated", {"members": len(members)})
        return view

    def sync_knowledge(child_id, keys):
        """Sync selected knowledge DOWNWARD to a SUBORDINATE child — never across a sovereign boundary.

        `keys` are relative paths inside THIS package (e.g. 'episodes.json', 'curriculum.json').
        Each is read from host.store and written into the child under `federated/<key>` via
        governance.governed_write, which enforces both the sovereignty boundary (can_edit) and the
        target-package confinement (do no harm). A sovereign/foreign child is read-only: no write is
        attempted and the reason is reported. Never raises.

        Returns {child_id, subordinate, written:[rel_paths_in_child], skipped:[{key,reason}], reason?}.
        """
        keys = [keys] if isinstance(keys, str) else list(keys or [])
        child = next((c for c in _read_children(host) if c.get("id") == child_id), None)
        if child is None:
            return {"child_id": child_id, "subordinate": False, "written": [],
                    "skipped": [], "reason": "no such child in lineage"}
        cdir = _child_dir(host, child)

        # THE sovereignty gate: governance.can_edit is False across any `sovereign/` segment.
        if not _can_edit(host.root, cdir) or _governed_write is None:
            host.log("sync_refused", child=child_id, reason="sovereign/read-only boundary")
            host.emit("lineage.sync_refused", {"child": child_id})
            return {"child_id": child_id, "subordinate": False, "written": [],
                    "skipped": [{"key": k, "reason": "sovereign boundary (read-only)"} for k in keys],
                    "reason": "child is sovereign/foreign — read-only across the boundary"}

        written, skipped = [], []
        for key in keys:
            content = host.store.read(key)              # confined to this package by Store._safe
            if content is None:
                skipped.append({"key": key, "reason": "not found in this package"})
                continue
            rel = os.path.join(_FED_PREFIX, key)
            try:
                _governed_write(host.root, cdir, rel, content)   # governance-checked cross-package write
                written.append(rel)
            except Exception as e:
                skipped.append({"key": key, "reason": str(e)[:160]})
        host.log("sync_knowledge", child=child_id, written=len(written), skipped=len(skipped))
        host.emit("lineage.synced", {"child": child_id, "written": len(written)})
        return {"child_id": child_id, "subordinate": True, "written": written, "skipped": skipped}

    host.log("module", step="tier2-20", id=MODULE_ID)
    return {"federate": federate, "sync_knowledge": sync_knowledge, "roster": roster}
