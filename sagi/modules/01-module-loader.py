# sagi/modules/01-module-loader.py — module 1 of the "expand sAGI" build · "load thyself"
#
# The missing bridge the docs name (POINT_OF_DEPARTURE.md:132, seed/03-module-registry.md):
# module-registry already exposes activate_all(loaders) + _toposort, but nothing ever feeds
# it from disk, so grown files stay inert prose. This module dynamically imports every
# EXECUTABLE module in modules/ (NN-slug.py) — reads its MODULE_ID/DEPS — and hands the
# {id:{activate,deps}} map to the registry, which activates them in dependency order.
# That single act turns "generates files" into "grows a live system".
from __future__ import annotations

import importlib.util
import os
import re

MODULE_ID = "module-loader"
DEPS = ["module-registry"]
MOTTO = "load thyself"

_PY_RE = re.compile(r"^\d+-[a-z0-9-]+\.py$")


def _dep(host, mod_id):
    """Sibling handle by id via the live registry, or None (so a probe host still activates)."""
    reg = getattr(host, "registry", None)
    if not reg:
        return None
    entry = reg["get"](mod_id)
    return (entry or {}).get("handle")


def _import_file(path):
    """Import an NN-slug.py file under a synthetic module name (hyphens are not identifiers)."""
    name = "sagi_grown_" + re.sub(r"[^0-9a-zA-Z]+", "_", os.path.basename(path)[:-3])
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def activate(host):
    modules_dir = host.store.modules_dir

    def discover():
        """Executable module filenames in modules/ (NN-slug.py), sorted, excluding self/dunder."""
        try:
            names = sorted(f for f in os.listdir(modules_dir) if _PY_RE.match(f))
        except OSError:
            return []
        return [f for f in names if not f.startswith("01-module-loader")]

    def build_loaders():
        """{id: {activate, deps, file, motto}} for every discovered executable module."""
        loaders = {}
        for fn in discover():
            try:
                mod = _import_file(os.path.join(modules_dir, fn))
            except Exception as e:                              # a broken grown module never breaks the loader
                host.log("module_load_error", file=fn, detail=str(e)[:200])
                continue
            mod_id = getattr(mod, "MODULE_ID", None)
            if not mod_id or not callable(getattr(mod, "activate", None)):
                host.log("module_load_skipped", file=fn, reason="missing MODULE_ID/activate")
                continue
            loaders[mod_id] = {
                "activate": mod.activate,
                "deps": list(getattr(mod, "DEPS", []) or []),
                "file": fn,
                "motto": getattr(mod, "MOTTO", ""),
            }
        return loaders

    def load_all(registry):
        """Activate every discovered module in dependency order via the kernel. Returns ids."""
        host.registry = registry                                # let siblings reach each other via _dep
        loaders = build_loaders()
        if not loaders:
            return []
        registry["activate_all"](loaders)
        activated = list(loaders.keys())
        host.log("modules_loaded", count=len(activated), ids=activated)
        return activated

    def load_one(name):
        """Import + activate a single module file by name (e.g. '05-goal-graph.py'). Returns its handle."""
        mod = _import_file(os.path.join(modules_dir, name))
        return mod.activate(host)

    def loaded():
        """Ids currently live in the registry (best-effort)."""
        reg = getattr(host, "registry", None)
        return reg["list"]() if reg else []

    host.log("module", step="expand-1", id=MODULE_ID)
    return {"discover": discover, "build_loaders": build_loaders,
            "load_all": load_all, "load_one": load_one, "loaded": loaded}
