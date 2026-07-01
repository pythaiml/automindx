# sagi/modules/44-dependency-audit.py — tier-5 · "audit thy bonds"
#
# GAP CLOSED: the module graph grows but nothing audits its health — a dependency cycle, an
# unresolved dep, or an orphan (a module nothing composes) goes unnoticed. This audits the live
# graph: detect cycles (DFS), unresolved deps (a dep that isn't live), and orphans, so the
# architecture stays sound as it scales past 40+ modules.
#
# GROUNDED REUSE: introspection.map (nodes + deps + edges), module-loader.build_loaders (on-disk deps).
from __future__ import annotations

MODULE_ID = "dependency-audit"
DEPS = ["introspection", "module-loader"]
MOTTO = "audit thy bonds"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _deps_map():
        intro = _sib(host, "introspection")
        m = intro["map"]() if intro else {"modules": []}
        return {mod["id"]: list(mod.get("deps", [])) for mod in m.get("modules", [])}

    def cycles():
        """All dependency cycles in the live graph (empty = acyclic)."""
        dm = _deps_map()
        found, stack, onstack = [], [], set()

        def dfs(nid, path):
            if nid in onstack:
                i = path.index(nid) if nid in path else 0
                found.append(path[i:] + [nid])
                return
            if nid not in dm:
                return
            onstack.add(nid)
            for d in dm.get(nid, []):
                dfs(d, path + [nid])
            onstack.discard(nid)

        for nid in dm:
            dfs(nid, [])
        # dedupe cycles by their set of nodes
        uniq, seen = [], set()
        for c in found:
            key = frozenset(c)
            if key not in seen:
                seen.add(key); uniq.append(c)
        return uniq

    def audit():
        """Full report: {cycles, unresolved, orphans, modules}."""
        dm = _deps_map()
        live = set(dm)
        unresolved = sorted({d for deps in dm.values() for d in deps if d not in live})
        depended = {d for deps in dm.values() for d in deps}
        seedish = {"individuality-core", "self-package-boundary", "module-registry", "module-loader"}
        orphans = sorted(m for m in live if m not in depended and m not in seedish)
        cyc = cycles()
        report = {"modules": len(live), "cycles": cyc, "unresolved": unresolved, "orphans": orphans,
                  "healthy": not cyc and not unresolved}
        host.log("dependency_audit", cycles=len(cyc), unresolved=len(unresolved), orphans=len(orphans))
        return report

    def orphans():
        return audit()["orphans"]

    host.log("module", step="tier5-44", id=MODULE_ID)
    return {"audit": audit, "cycles": cycles, "orphans": orphans}
