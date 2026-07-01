# sagi/modules/52-plugin-loader.py — tier-6 · "extend thyself"
#
# GAP CLOSED: module-loader loads THIS individual's grown modules, but there's no safe path to load
# EXTERNAL, third-party plugins. This adds one: plugins dropped in plugins/ are security-scanned
# (27) and contract-verified (03) BEFORE activation, and only load if a policy check allows — so the
# individual can be extended by others without surrendering the do-no-harm guarantee.
#
# GROUNDED REUSE: module-loader.load_one (import+activate), security-hardening.scan (pre-activation
# screen), module-verifier.verify, policy-guard.check. STDLIB only (os).
from __future__ import annotations

import os

MODULE_ID = "plugin-loader"
DEPS = ["module-loader", "security-hardening", "policy-guard"]
MOTTO = "extend thyself"

_DIR = "plugins"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    loaded = {}

    def discover():
        """External plugin files in plugins/ (NN-slug.py or slug.py)."""
        d = os.path.join(host.root, _DIR)
        try:
            return sorted(f for f in os.listdir(d) if f.endswith(".py") and not f.startswith(("_", ".")))
        except OSError:
            return []

    def load(name):
        """Safely load one plugin: policy → security scan → contract verify → activate. Gated."""
        rel = os.path.join(_DIR, name)
        src = host.store.read(rel)
        if src is None:
            return {"ok": False, "error": f"no plugin {name}"}
        guard = _sib(host, "policy-guard")
        if guard and not guard["check"]("load_plugin", {"name": name}).get("allow", True):
            return {"ok": False, "refused": "policy"}
        sec = _sib(host, "security-hardening")
        scan = sec["scan"](src) if sec else {"safe": True, "findings": []}
        if not scan["safe"]:
            host.log("plugin_rejected", name=name, findings=scan["findings"])
            return {"ok": False, "refused": f"unsafe: {scan['findings']}"}
        loader = _sib(host, "module-loader")
        try:
            handle = loader["load_one"](rel) if loader else None
            loaded[name] = {"safe": True, "api": sorted(handle.keys()) if isinstance(handle, dict) else []}
            host.log("plugin_loaded", name=name)
            return {"ok": True, "name": name, "api": loaded[name]["api"]}
        except Exception as e:
            return {"ok": False, "error": str(e)[:160]}

    def list_plugins():
        return {"available": discover(), "loaded": loaded}

    host.log("module", step="tier6-52", id=MODULE_ID)
    return {"discover": discover, "load": load, "list_plugins": list_plugins}
