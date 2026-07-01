# sagi/runtime/boot.py — bring an individual sAGI to life from the seed modules.
#   from sagi.runtime import boot
#   host, handles = boot("/path/to/<individual>")
#
#   python3 -m sagi.runtime.boot --dir ./sagi          # offline demo (no model call)
from __future__ import annotations

import importlib.util
import os
import re
from typing import Callable, Dict, Optional, Tuple

from .host import Host
from .gitmind import GitMind
from .modules import SEED


def _import_file(path: str):
    """Import a grown NN-slug.py module file under a synthetic name (hyphens aren't identifiers)."""
    name = "sagi_grown_" + re.sub(r"[^0-9a-zA-Z]+", "_", os.path.basename(path)[:-3])
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def boot(sagi_dir: str, call_model: Optional[Callable[[str, str], str]] = None,
         rage: bool = False) -> Tuple[Host, Dict[str, dict]]:
    """Activate the three seed modules in ship order and register them in the kernel.

    be thyself (1) -> do no harm (2) -> grow thyself (3). Returns the live host and
    the module handles keyed by id. The host also grows a gitmind memory tree that
    snapshots each persisted moment, so memory is reachable from any .history moment.
    With rage=True, global (expansion / massive-upgrade) commits are also embedded
    into the RAGE store (pgvectorscale, https://rage.pythai.net).
    """
    host = Host(sagi_dir, call_model=call_model)
    host.memory = GitMind(host.root)          # git-like internal memory tree
    host.rage = None
    if rage:
        from .rage_sync import RageSync
        host.rage = RageSync(host.memory)
        host.memory.on_commit = lambda ch, obj: (
            host.rage.save_commit(ch) if obj.get("scope") == "global" else None)
    host.log("run_start", runtime="sagi.runtime")
    handles: Dict[str, dict] = {}
    for mod in SEED:                          # 1 -> 2 -> 3
        handles[mod.MODULE_ID] = mod.activate(host)
    registry = handles["module-registry"]
    for mod in SEED:                          # every seed becomes a live registered module
        registry["register"](mod.MODULE_ID, handles[mod.MODULE_ID], {"deps": mod.DEPS, "motto": mod.MOTTO})
    host.registry = registry                  # siblings reach each other via host.registry["get"](id)
    # grow thyself, for real: if the module-loader has been grown, let it import every
    # executable module in modules/ and activate it through the kernel in deps order.
    loader_file = os.path.join(host.store.modules_dir, "01-module-loader.py")
    if os.path.exists(loader_file):
        loader = _import_file(loader_file).activate(host)
        handles["module-loader"] = loader
        registry["register"]("module-loader", loader, {"deps": ["module-registry"], "motto": "load thyself"})
        for gid in loader["load_all"](registry):
            handles[gid] = registry["get"](gid)["handle"]
    # every persisted moment is a LOCAL snapshot, chained to .history by timestamp.
    host.on("module.persisted", lambda p: host.memory.commit(
        moment={"event": "module.persisted", **(p or {})}, message=(p or {}).get("id", "")))
    # genesis is a GLOBAL milestone (the first expansion of this individual).
    host.memory.global_commit(moment={"event": "boot"}, message="seed")
    host.log("run_end", modules=len(handles))
    return host, handles


def default_call_model(backend: str = "ollama", model: Optional[str] = None) -> Callable[[str, str], str]:
    """Wire host.call_model to the headless builder's backends (ollama/claude-*)."""
    import sagi_build  # repo-root builder; provides ask_model(model, prompt, backend)
    mdl = model or sagi_build.DEFAULT_MODEL
    return lambda system, prompt: sagi_build.ask_model(mdl, f"{system}\n\n{prompt}" if system else prompt, backend)


if __name__ == "__main__":
    import argparse
    import json
    import os

    ap = argparse.ArgumentParser(description="Boot an individual sAGI runtime from its seed modules")
    ap.add_argument("--dir", default=os.environ.get("SAGI_DIR", "./sagi"))
    a = ap.parse_args()
    host, handles = boot(a.dir)
    print("who   :", json.dumps(handles["individuality-core"]["whoami"]()))
    print("live  :", handles["module-registry"]["list"]())
    manifest = handles["self-package-boundary"]["syncManifest"]()
    print("manifest:", manifest.get("version"), [m["file"] for m in manifest["modules"]])
    # demonstrate live registration without a restart
    host.emit("module.persisted", {"id": "04-example", "file": "04-example.md"})
    print("after persist:", handles["module-registry"]["list"]())
