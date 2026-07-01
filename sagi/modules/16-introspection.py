# sagi/modules/16-introspection.py — tier-2 · "see thyself"
#
# Gap it closes: modules can reason, verify, and score (evaluator), but nothing offers the
# structured "what am I right now" surface — the live self-map — that every observability / UI /
# wire feature (17-telemetry, 18-wire-gateway) needs. This assembles that map purely from things
# already live in the runtime: the registry's list/get (module ids + their callable handle keys),
# module-loader.build_loaders (dependency edges + mottos), and module-verifier.verify_all (health).
#
# Grounded reuse (nothing reinvented):
#   - registry["list"]/["get"]      -> live ids, handle (API names), registered meta.deps
#   - module-loader.build_loaders   -> {id: {deps, file, motto}} from disk
#   - module-verifier.verify_all    -> {passed, failed} live health
# New logic (small): merge those sources into {modules, edges}; render a Markdown/JSON export.
from __future__ import annotations

MODULE_ID = "introspection"
DEPS = ["evaluator"]
MOTTO = "see thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _loaders():
        """{id: {deps, file, motto}} from module-loader (guarded — probe host has no sibling)."""
        loader = _sib(host, "module-loader")
        try:
            return loader["build_loaders"]() if loader else {}
        except Exception as e:                       # a broken grown file must not break the self-map
            host.log("introspection_loaders_error", detail=str(e)[:160])
            return {}

    def _api_keys(mid):
        """Public API of a live module = the callable keys of its registered handle dict."""
        reg = getattr(host, "registry", None)
        handle = (reg["get"](mid) or {}).get("handle") if reg else None
        if isinstance(handle, dict):
            return sorted(k for k, v in handle.items() if callable(v))
        return []

    def _deps_of(mid, loaders):
        """Deps: prefer the registry meta recorded at activation, fall back to on-disk loader deps."""
        reg = getattr(host, "registry", None)
        meta = (reg["get"](mid) or {}).get("meta", {}) if reg else {}
        return list(meta.get("deps") or (loaders.get(mid, {}).get("deps", []) or []))

    def _map():
        """Live self-map: {modules:[{id,motto,file,deps,api}], edges:[{from,to}]}."""
        reg = getattr(host, "registry", None)
        live = reg["list"]() if reg else []
        loaders = _loaders()
        modules, edges = [], []
        live_set = set(live)
        for mid in live:
            spec = loaders.get(mid, {})
            deps = _deps_of(mid, loaders)
            modules.append({
                "id": mid,
                "motto": spec.get("motto", ""),
                "file": spec.get("file"),
                "deps": deps,
                "api": _api_keys(mid),
            })
            for d in deps:
                edges.append({"from": mid, "to": d, "resolved": d in live_set})
        m = {"modules": modules, "edges": edges}
        _export(m)                                   # persist JSON + Markdown snapshot each map
        host.log("introspection_map", modules=len(modules), edges=len(edges))
        return m

    def describe(mid):
        """One module's face: {id, motto, deps, api:[handle keys]}."""
        loaders = _loaders()
        spec = loaders.get(mid, {})
        return {
            "id": mid,
            "motto": spec.get("motto", ""),
            "deps": _deps_of(mid, loaders),
            "api": _api_keys(mid),
        }

    def health():
        """Live verify-health via module-verifier: {verified:[...], failed:[...]}."""
        verifier = _sib(host, "module-verifier")
        res = verifier["verify_all"]() if verifier else {"passed": [], "failed": []}
        return {"verified": list(res.get("passed", [])), "failed": list(res.get("failed", []))}

    # --- markdown / json export (grounded in host.store confinement) ---
    def _render_md(m):
        lines = ["# sAGI — live self-map", "", f"_{len(m['modules'])} live modules_", ""]
        for mod in m["modules"]:
            motto = f" — *{mod['motto']}*" if mod.get("motto") else ""
            lines.append(f"## {mod['id']}{motto}")
            if mod.get("deps"):
                lines.append(f"- deps: {', '.join(mod['deps'])}")
            if mod.get("api"):
                lines.append(f"- api: {', '.join(mod['api'])}")
            if mod.get("file"):
                lines.append(f"- file: `{mod['file']}`")
            lines.append("")
        return "\n".join(lines) + "\n"

    def _export(m=None):
        """Persist the self-map as introspection.json + introspection.md (Store-confined)."""
        m = m if m is not None else {"modules": [], "edges": []}
        try:
            host.store.write_json("introspection.json", m)
            host.store.write("introspection.md", _render_md(m))
        except Exception as e:
            host.log("introspection_export_error", detail=str(e)[:160])
        return m

    host.log("module", step="tier2-16", id=MODULE_ID)
    return {"map": _map, "describe": describe, "health": health}
