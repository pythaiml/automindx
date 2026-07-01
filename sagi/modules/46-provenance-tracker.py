# sagi/modules/46-provenance-tracker.py — tier-5 · "trace thy origins"
#
# GAP CLOSED: .history logs events and gitmind chains snapshots, but there is no single answer to
# "where did THIS come from" for a given module/artifact — its build event, backend, and the deps it
# was composed from. This assembles that provenance from the existing record (.history/build.jsonl +
# the live dep graph) and lets any module attach custom provenance. Read-only over what already exists.
#
# GROUNDED REUSE: .history/build.jsonl (module/build_next events), introspection.describe (deps),
# host.memory.log (snapshot chain). STDLIB only (json, os).
from __future__ import annotations

import json
import os

MODULE_ID = "provenance-tracker"
DEPS = ["introspection", "telemetry"]
MOTTO = "trace thy origins"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _history():
        path = os.path.join(host.root, ".history", "build.jsonl")
        out = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
        except OSError:
            pass
        return out

    def record(artifact, origin):
        """Attach a custom provenance note to an artifact (persisted to provenance.json)."""
        prov = host.store.read_json("provenance.json", default=[]) or []
        prov.append({"artifact": artifact, "origin": origin})
        host.store.write_json("provenance.json", prov)
        return {"artifact": artifact}

    def lineage_of(module_id):
        """Where a module came from: its build events + the deps it was composed from."""
        events = [e for e in _history()
                  if e.get("id") == module_id or e.get("file", "").startswith(module_id)
                  or module_id in str(e.get("node", ""))]
        intro = _sib(host, "introspection")
        deps = intro["describe"](module_id).get("deps", []) if intro else []
        custom = [p for p in (host.store.read_json("provenance.json", default=[]) or [])
                  if p.get("artifact") == module_id]
        return {"module": module_id, "events": events[-10:], "composed_from": deps, "custom": custom}

    def all_provenance():
        """A compact provenance index of every module event on record."""
        idx = {}
        for e in _history():
            if e.get("event") in ("module", "build_next") and (e.get("id") or e.get("node")):
                key = e.get("id") or e.get("node")
                idx.setdefault(key, {"first_ts": e.get("ts"), "backend": e.get("backend")})
        return idx

    host.log("module", step="tier5-46", id=MODULE_ID)
    return {"record": record, "lineage_of": lineage_of, "all": all_provenance}
