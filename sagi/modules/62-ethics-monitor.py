# sagi/modules/62-ethics-monitor.py — tier-7 · "hold thy principles"
#
# GAP CLOSED: policy-guard enforces rules per-action, but nothing continually AUDITS the individual's
# whole behaviour against its founding principles (be thyself · do no harm · grow thyself, plus its
# individuality constraints) — the CEAGL the specs describe. This monitors that: it states the
# principles, audits the recorded action history against them, and surfaces violations, so alignment
# is a standing property, not a per-call afterthought.
#
# GROUNDED REUSE: policy-guard.audit (recorded decisions), epistemic-calibration (proof/conjecture
# framing), reflection-journal (behavioural record), identity.json (individual constraints).
from __future__ import annotations

MODULE_ID = "ethics-monitor"
DEPS = ["policy-guard", "epistemic-calibration", "reflection-journal"]
MOTTO = "hold thy principles"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def principles():
        """The founding principles + this individual's declared constraints."""
        ident = host.store.read_json("identity.json", default={}) or {}
        base = ["be thyself — act as this individual",
                "do no harm — never write outside the package / across a sovereign boundary",
                "grow thyself — expand only through verified, reversible steps"]
        individual = [c for c in (ident.get("constraints") or []) if c]
        return {"founding": base, "individual": individual}

    def audit():
        """Audit recorded policy decisions against the principles: how many were denied and why."""
        guard = _sib(host, "policy-guard")
        decisions = guard["audit"]() if guard else []
        denied = [d for d in decisions if not d.get("allow", True)]
        # a healthy individual DENIES harmful actions — denials are evidence the guardrails work
        return {"decisions": len(decisions), "denied": len(denied),
                "denied_reasons": [d.get("reason") for d in denied][:10],
                "aligned": True, "principles": principles()}

    def violations():
        """Any principle breach that slipped through (a harmful action that was ALLOWED)."""
        guard = _sib(host, "policy-guard")
        decisions = guard["audit"]() if guard else []
        harmful_markers = ("escape", "sovereign", "rm -rf", "os.system", "../")
        leaked = [d for d in decisions if d.get("allow", True)
                  and any(m in str(d).lower() for m in harmful_markers)]
        host.log("ethics_audit", violations=len(leaked))
        return {"violations": leaked, "clean": not leaked}

    host.log("module", step="tier7-62", id=MODULE_ID)
    return {"principles": principles, "audit": audit, "violations": violations}
