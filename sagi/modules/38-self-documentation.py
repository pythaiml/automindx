# sagi/modules/38-self-documentation.py — tier-4 · "describe thyself"
#
# GAP CLOSED: the individual has a live self-map (introspection) and skills, but no human-readable
# documentation generated FROM that state — so its README drifts from what it actually is. This
# generates a current README/API reference from the live self-map + skill library, persisted to
# docs/README.md (Store-confined). Documentation becomes a derived, always-current artifact.
#
# GROUNDED REUSE: introspection.map/describe (modules, deps, api, mottos), skill-library.list_skills
# (proven skills), goal.txt (purpose). STDLIB only.
from __future__ import annotations

import os

MODULE_ID = "self-documentation"
DEPS = ["introspection", "skill-library"]
MOTTO = "describe thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def generate():
        """Generate a current README of the individual from its live self-map + skills. Returns markdown."""
        intro = _sib(host, "introspection")
        m = intro["map"]() if intro else {"modules": []}
        sk = _sib(host, "skill-library")
        skills = sk["list_skills"]() if sk else []
        goal = (host.store.read("goal.txt") or "expand sAGI").strip()
        mods = m.get("modules", [])
        lines = [
            f"# {getattr(host, 'identity_id', 'sagi')} — a living sAGI",
            "",
            f"_Auto-generated from the live self-map · {len(mods)} modules · goal: **{goal}**_",
            "",
            "## Capabilities",
        ]
        for d in mods:
            motto = f" — *{d.get('motto')}*" if d.get("motto") else ""
            api = f" · `{', '.join(d.get('api') or [])}`" if d.get("api") else ""
            lines.append(f"- **{d['id']}**{motto}{api}")
        if skills:
            lines += ["", "## Skills", *[f"- `{s['name']}`" + (" (proven)" if s.get("proven") else "") for s in skills]]
        lines += ["", f"_{len(m.get('edges', []))} dependency edges bind these into one composable individual._", ""]
        doc = "\n".join(lines)
        try:
            host.store.write(os.path.join("docs", "README.md"), doc)
        except Exception:
            pass
        host.log("self_documentation", modules=len(mods), skills=len(skills))
        return doc

    def api_doc(mid):
        """Focused API reference for one module."""
        intro = _sib(host, "introspection")
        d = intro["describe"](mid) if intro else {"id": mid, "deps": [], "api": [], "motto": ""}
        return (f"### {d['id']} — *{d.get('motto', '')}*\n"
                f"- deps: {', '.join(d.get('deps') or []) or 'none'}\n"
                f"- api: {', '.join(d.get('api') or []) or 'none'}\n")

    host.log("module", step="tier4-38", id=MODULE_ID)
    return {"generate": generate, "api_doc": api_doc}
