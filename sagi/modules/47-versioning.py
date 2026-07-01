# sagi/modules/47-versioning.py — tier-5 · "version thyself"
#
# GAP CLOSED: manifest.json carries only a coarse 0.0.N version and modules have no individual
# version or API-change detection — so a breaking change to a module's public surface is invisible.
# This assigns each module a semantic version derived from its API signature, bumps it when the
# signature changes, and keeps a changelog. Persisted to versions.json.
#
# GROUNDED REUSE: introspection.describe (the api list = the signature), host.store. STDLIB only.
from __future__ import annotations

import hashlib

MODULE_ID = "versioning"
DEPS = ["introspection", "module-verifier"]
MOTTO = "version thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _store():
        return host.store.read_json("versions.json", default={}) or {}

    def _sig(mid):
        intro = _sib(host, "introspection")
        api = intro["describe"](mid).get("api", []) if intro else []
        return hashlib.sha256(",".join(sorted(api)).encode()).hexdigest()[:12], api

    def version(module_id):
        """Current semver of a module (registering v1.0.0 on first sight)."""
        store = _store()
        sig, api = _sig(module_id)
        rec = store.get(module_id)
        if not rec:
            rec = {"version": "1.0.0", "sig": sig, "api": api, "changes": []}
            store[module_id] = rec
            host.store.write_json("versions.json", store)
        return {"module": module_id, "version": rec["version"], "sig": rec["sig"]}

    def bump(module_id, kind="minor"):
        """Bump a module's semver (major|minor|patch) if its API signature changed; record changelog."""
        store = _store()
        sig, api = _sig(module_id)
        rec = store.get(module_id) or {"version": "1.0.0", "sig": None, "api": [], "changes": []}
        if rec["sig"] == sig and rec.get("api") == api:
            return {"module": module_id, "version": rec["version"], "changed": False}
        major, minor, patch = (int(x) for x in rec["version"].split("."))
        if kind == "major": major, minor, patch = major + 1, 0, 0
        elif kind == "patch": patch += 1
        else: minor, patch = minor + 1, 0
        added = sorted(set(api) - set(rec.get("api", [])))
        removed = sorted(set(rec.get("api", [])) - set(api))
        rec.update({"version": f"{major}.{minor}.{patch}", "sig": sig, "api": api})
        rec["changes"].append({"to": rec["version"], "added": added, "removed": removed})
        store[module_id] = rec
        host.store.write_json("versions.json", store)
        host.log("version_bump", module=module_id, version=rec["version"], added=len(added), removed=len(removed))
        return {"module": module_id, "version": rec["version"], "changed": True, "added": added, "removed": removed}

    def changelog(module_id=None):
        store = _store()
        if module_id:
            return store.get(module_id, {}).get("changes", [])
        return {m: r.get("version") for m, r in store.items()}

    host.log("module", step="tier5-47", id=MODULE_ID)
    return {"version": version, "bump": bump, "changelog": changelog}
