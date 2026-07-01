# sagi/modules/28-persistence-deploy.py — tier-3 · "endure thyself"
#
# GAP CLOSED: the individual can grow and serve (wire-gateway) but has no deploy descriptor and no
# explicit snapshot-restore across restarts — so it can't be packaged to endure. This produces a
# deploy manifest (how to boot + serve this individual, incl. the Tauri path), and a restore() that
# reconstructs module state from the gitmind memory tree / a savepoint. Grounds the "includable
# anywhere / Tauri" promise (sagi/tauri.md) with something executable.
#
# GROUNDED REUSE: introspection.map (what to package), wire-gateway.routes/serve (how to serve),
# host.memory (gitmind tree for restore), savepoint (state maintained). STDLIB only.
from __future__ import annotations

import os

MODULE_ID = "persistence-deploy"
DEPS = ["wire-gateway", "introspection", "sagi-environment"]
MOTTO = "endure thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def package():
        """Emit a deploy manifest describing how to boot + serve this individual. Persisted to deploy.json."""
        intro = _sib(host, "introspection")
        gw = _sib(host, "wire-gateway")
        m = intro["map"]() if intro else {"modules": []}
        manifest = {
            "individual": getattr(host, "identity_id", "sagi"),
            "boot": "python3 -m sagi.runtime.boot --dir $SAGI_DIR",
            "serve": "host.registry['get']('wire-gateway')['handle']['serve'](port)",
            "routes": [r for r in (gw["routes"]() if gw else [])],
            "modules": len(m.get("modules", [])),
            "tauri": "wrap sagi/ per sagi/tauri.md; expose wire-gateway over the Tauri command bridge",
        }
        try:
            host.store.write_json("deploy.json", manifest)
        except Exception:
            pass
        host.log("package", modules=manifest["modules"])
        return manifest

    def restore(commit=None):
        """Reconstruct module state from the gitmind tree (a savepoint commit, or the tip)."""
        mem = getattr(host, "memory", None)
        if mem is None:
            return {"restored": 0, "note": "no memory tree on this host"}
        commit = commit or mem.head()
        tree = mem.tree(commit) if commit else {}
        # state is on disk already after boot; restore confirms integrity by re-materialising any
        # missing module file from the snapshot (Store-confined), never overwriting a newer one.
        restored = 0
        for fn, content in tree.items():
            p = os.path.join("modules", fn)
            if host.store.read(p) is None:
                host.store.write(p, content)
                restored += 1
        host.log("restore", commit=(commit or "")[:12], restored=restored, from_tree=len(tree))
        return {"restored": restored, "from_tree": len(tree), "commit": (commit or "")[:12]}

    def status():
        reg = getattr(host, "registry", None)
        gw = _sib(host, "wire-gateway")
        return {"live": len(reg["list"]()) if reg else 0,
                "servable": bool(gw), "packaged": host.store.read_json("deploy.json", default=None) is not None}

    host.log("module", step="tier3-28", id=MODULE_ID)
    return {"package": package, "restore": restore, "status": status}
