# sagi/modules/36-explainability.py — tier-4 · "explain thyself"
#
# GAP CLOSED: the individual can act and self-map, but can't render a human-readable account of WHY
# a module exists or a decision was made — the Explainable-Decision engine the specs describe. This
# builds that from live sources: explain(id) states a module's purpose (motto), what it composes
# (deps), what it exposes (api), and the epistemic status (beliefs it relates to); trace() walks the
# causal dependency chain back to the seed. No new data — pure synthesis over introspection + beliefs.
#
# GROUNDED REUSE: introspection.describe/map (deps + api + motto), hypothesis-engine.beliefs
# (epistemic support), reflection-journal.journal (recorded rationale). STDLIB only.
from __future__ import annotations

MODULE_ID = "explainability"
DEPS = ["introspection", "hypothesis-engine", "reflection-journal"]
MOTTO = "explain thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def explain(subject):
        """A human-readable account of a module: purpose, composition, surface, epistemic status."""
        intro = _sib(host, "introspection")
        desc = intro["describe"](subject) if intro else {"id": subject, "deps": [], "api": [], "motto": ""}
        he = _sib(host, "hypothesis-engine")
        beliefs = [b for b in (he["beliefs"]() if he else []) if subject in str(b.get("claim", ""))]
        lines = [
            f"### Why '{subject}' exists",
            f"- **Purpose:** {desc.get('motto') or '(no motto)'}",
            f"- **Composes:** {', '.join(desc.get('deps') or []) or '(no dependencies — foundational)'}",
            f"- **Provides:** {', '.join(desc.get('api') or []) or '(no public API)'}",
        ]
        if beliefs:
            lines.append(f"- **Evidence:** {len(beliefs)} related belief(s); "
                         f"latest confidence {beliefs[-1].get('confidence')}")
        return "\n".join(lines) + "\n"

    def trace(subject, _seen=None):
        """The causal chain: subject → its deps → their deps … back to the seed (no cycles)."""
        intro = _sib(host, "introspection")
        _seen = _seen if _seen is not None else set()
        if subject in _seen or not intro:
            return {"id": subject, "deps": []}
        _seen.add(subject)
        deps = intro["describe"](subject).get("deps", []) if intro else []
        return {"id": subject, "deps": [trace(d, _seen) for d in deps]}

    def rationale(limit=3):
        """The most recent recorded rationale (reflection notes) — the operator-facing 'why now'."""
        journal = _sib(host, "reflection-journal")
        notes = journal["journal"]() if journal else []
        return [{"file": n.get("file"), "excerpt": (n.get("text") or "")[:200]} for n in notes[-limit:]]

    host.log("module", step="tier4-36", id=MODULE_ID)
    return {"explain": explain, "trace": trace, "rationale": rationale}
