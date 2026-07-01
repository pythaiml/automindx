# sagi/modules/65-capability-discovery.py — tier-7 · "find thy powers"
#
# GAP CLOSED: the individual has 60+ modules but no view of the EMERGENT capabilities their
# composition affords, nor what it could build next to unlock more. This discovers latent
# capabilities by grouping modules into functional clusters, describes composite powers, and suggests
# the next module from the buildable frontier — turning a flat module list into a capability map.
#
# GROUNDED REUSE: introspection.map (modules + edges), skill-library.list_skills, planning-search.frontier.
from __future__ import annotations

MODULE_ID = "capability-discovery"
DEPS = ["introspection", "skill-library", "planning-search"]
MOTTO = "find thy powers"

# functional clusters keyed by motto/name signal — a coarse capability taxonomy
_THEMES = {
    "memory": ("recall", "memory", "consolid", "conversation", "provenance", "event"),
    "reasoning": ("hypothesis", "calibration", "explain", "constraint", "planning"),
    "building": ("build", "curriculum", "goal", "skill", "refactor", "meta-kernel"),
    "safety": ("verif", "security", "policy", "rollback", "anomaly", "ethics", "test", "audit"),
    "society": ("swarm", "federation", "negotiation", "consensus", "sovereign", "lineage"),
    "interface": ("wire", "human", "notification", "dashboard", "api", "multimodal", "prompt"),
}


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _modules():
        intro = _sib(host, "introspection")
        return intro["map"]()["modules"] if intro else []

    def discover():
        """Cluster live modules into capability themes; a theme with modules = a latent power."""
        clusters = {t: [] for t in _THEMES}
        clusters["other"] = []
        for m in _modules():
            placed = False
            for theme, keys in _THEMES.items():
                if any(k in m["id"] for k in keys):
                    clusters[theme].append(m["id"]); placed = True; break
            if not placed:
                clusters["other"].append(m["id"])
        powers = {t: mods for t, mods in clusters.items() if mods}
        host.log("capability_discover", themes=len(powers))
        return {"capabilities": powers,
                "emergent": [t for t, mods in powers.items() if len(mods) >= 3]}

    def compose(a, b):
        """Describe the composite capability of two modules via their shared/combined surface."""
        intro = _sib(host, "introspection")
        da = intro["describe"](a) if intro else {}
        db = intro["describe"](b) if intro else {}
        return {"pair": [a, b], "combined_api": sorted(set(da.get("api", []) + db.get("api", []))),
                "note": f"{a} ({da.get('motto')}) + {b} ({db.get('motto')})"}

    def suggest():
        """What to build next: the buildable frontier, or (if converged) an underpopulated theme."""
        ps = _sib(host, "planning-search")
        frontier = ps["frontier"]() if ps else []
        if frontier:
            return {"suggestion": "build", "candidates": frontier}
        thin = [t for t, mods in discover()["capabilities"].items() if len(mods) < 3 and t != "other"]
        return {"suggestion": "deepen", "underpopulated_themes": thin}

    host.log("module", step="tier7-65", id=MODULE_ID)
    return {"discover": discover, "compose": compose, "suggest": suggest}
