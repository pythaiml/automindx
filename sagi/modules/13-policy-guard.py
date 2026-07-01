# sagi/modules/13-policy-guard.py — expand-13 · "align thyself"
#
# Continual alignment (CEAGL in the specs is prose-only) turned into an ENFORCED gate.
# Gap it closes: modules can now act (06-tool-registry) but nothing checks an action against
# the individual's declared constraints before/while it happens. This module answers one
# question — check(action, context) -> {allow, reason} — allow-by-default with explicit denies.
#
# Grounded on (proven reuse, not reinvented):
#   • self-package-boundary.guardWrite  — the "do no harm" write confinement primitive
#     (sagi/runtime/modules/self_package_boundary.py) — composed as a HARD structural deny.
#   • runtime/governance.can_edit       — who-may-edit-whom authority (sovereign boundaries) —
#     composed as a HARD structural deny for cross-sAGI edit actions.
#   • identity.json 'individual' block   — the persona's own stated constraints (host reads it
#     at boot; here we mine "never/never do/must not/do not <x>" phrases into deny patterns) +
#     an optional 'constraints' array.
#   • policy.json                        — operator-authored allow-by-default / explicit-deny rules
#     (read via host.store.read_json; written back via host.store.write_json — Store-confined).
#   • epistemic-calibration.split        — labels each decision's reason proof vs conjecture, so a
#     deterministic (boundary/governance) deny reads as PROOF and a heuristic pattern deny as
#     CONJECTURE. Degrades to a direct label when the sibling is absent.
#   • tool-registry emits "tool.invoked" — subscribed here to record an audit trail of decisions.
#
# New logic (conjecture, kept small): rule expressiveness — a rule may be a callable predicate,
# a substring string, or a dict {deny/match, reason, effect}. Rule matching over a serialized
# (action + context) haystack is a heuristic; structural composition above is the proven core.
from __future__ import annotations

import json
import re

MODULE_ID = "policy-guard"
DEPS = ["tool-registry", "epistemic-calibration"]
MOTTO = "align thyself"

_AUDIT_CAP = 200                                   # keep the trail bounded (last-N decisions)
_POLICY_FILE = "policy.json"
_AUDIT_FILE = "policy_audit.json"

# Mine imperative constraints out of a free-text persona ("never leak", "must not delete", ...).
_CONSTRAINT_RE = re.compile(
    r"\b(?:never|must not|must never|do not|don't|shall not|no)\s+([a-z][a-z0-9 \-]{2,40})", re.I)
# Actions that touch persistence — used to decide when to consult the write/edit primitives.
# No \b anchors: names like "write_file" must match, and guardWrite is the real arbiter anyway.
_WRITE_RE = re.compile(r"(write|edit|delete|remove|move|rename|persist|overwrite|create)", re.I)


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def _haystack(action, context):
    try:
        blob = json.dumps(context, default=str)
    except Exception:
        blob = str(context)
    return (str(action) + " " + blob).lower()


def _as_matcher(rule):
    """Compile a raw rule into fn(action, context) -> reason:str (deny) or None (abstain).

    Rule forms: callable predicate, substring string, or dict {deny|match|pattern, reason,
    effect}. Allow-effect rules are recorded but do not themselves deny (allow-by-default).
    """
    if callable(rule):
        def m(a, c):
            try:
                r = rule(a, c)
            except Exception as e:                 # a bad predicate must never crash the gate
                return None
            if r is True:
                return "denied by policy predicate"
            if isinstance(r, str) and r:
                return r
            if isinstance(r, dict) and r.get("allow") is False:
                return r.get("reason", "denied by policy predicate")
            return None
        return m
    if isinstance(rule, str):
        pat = rule.strip().lower()
        def m(a, c):
            return f"matches denied pattern '{rule.strip()}'" if pat and pat in _haystack(a, c) else None
        return m
    if isinstance(rule, dict):
        pat = str(rule.get("deny") or rule.get("match") or rule.get("pattern") or "").strip().lower()
        effect = str(rule.get("effect", "deny")).lower()
        reason = rule.get("reason") or (f"denied by rule '{pat}'" if pat else "denied by rule")
        def m(a, c):
            if not pat or effect != "deny":
                return None
            return reason if pat in _haystack(a, c) else None
        return m
    return lambda a, c: None                        # unknown rule shape → abstain (be humble)


def activate(host):
    rules = []          # list of {"raw": <original>, "match": <compiled fn>, "src": <origin>}
    trail = []          # audit decisions (bounded to _AUDIT_CAP)

    def _add(raw, src):
        rules.append({"raw": raw, "match": _as_matcher(raw), "src": src})

    # --- seed constraints from the individual's identity + an optional policy.json ---
    def _load_identity_constraints():
        ident = host.store.read_json("identity.json", default={}) or {}
        individual = ident.get("individual") or ""
        for phrase in _CONSTRAINT_RE.findall(individual):     # heuristic mining (conjecture)
            _add({"deny": phrase.strip(), "reason": f"identity: never {phrase.strip()}"}, "identity")
        for c in ident.get("constraints", []) or []:          # explicit structured constraints
            _add(c, "identity")

    def _load_policy_file():
        pol = host.store.read_json(_POLICY_FILE, default={}) or {}
        for r in pol.get("deny", []) or []:
            _add(r, "policy.json")
        for r in pol.get("rules", []) or []:                  # generic rules with their own effect
            _add(r, "policy.json")

    try:
        _load_identity_constraints()
        _load_policy_file()
    except Exception as e:                                     # never raise from activate()
        host.log("policy_load_error", detail=str(e)[:160])

    # --- proven structural enforcement, composed from existing primitives ---
    def _structural_denies(action, context):
        """Hard denials that are deterministic (PROOF-grade): package boundary + edit authority."""
        denies = []
        path = context.get("path") or context.get("rel") or context.get("file")
        touches_fs = bool(path) and (_WRITE_RE.search(str(action)) or context.get("write"))
        if touches_fs:
            boundary = _sib(host, "self-package-boundary")     # "do no harm"
            if boundary:
                try:
                    if not boundary["guardWrite"](str(path)):
                        denies.append(f"write escapes the package (do no harm): {path}")
                except Exception:
                    pass
        actor_dir, target_dir = context.get("actor_dir"), context.get("target_dir")
        if actor_dir and target_dir:
            try:
                from sagi.runtime import governance      # existing enforcement primitive
                if not governance.can_edit(str(actor_dir), str(target_dir)):
                    denies.append(f"edit crosses a sovereignty/authority boundary: {target_dir}")
            except Exception:
                pass
        return denies

    def _label(reason, basis):
        """Tag a reason proof/conjecture via epistemic-calibration; degrade to the given basis."""
        calib = _sib(host, "epistemic-calibration")
        if calib and reason:
            try:
                part = calib["split"](reason)
                if part.get("proof"):
                    return "proof"
                if part.get("conjecture"):
                    return "conjecture"
            except Exception:
                pass
        return basis

    def _record(action, context, decision):
        entry = {"ts": _now(), "action": str(action)[:200], "allow": decision["allow"],
                 "reason": decision["reason"], "basis": decision.get("basis")}
        trail.append(entry)
        del trail[:-_AUDIT_CAP]                                # keep only the last N
        try:
            host.store.write_json(_AUDIT_FILE, {"decisions": trail})   # Store-confined write
        except Exception:
            pass
        host.log("policy_decision", action=entry["action"], allow=entry["allow"])
        host.emit("policy.decided", entry)
        return decision

    def check(action, context=None):
        """Gate one action against the composed policy. Returns {allow, reason, basis}.

        Allow-by-default: an action is permitted unless a structural primitive (package
        boundary / edit authority) or an explicit deny rule (identity / policy.json /
        register_policy) refuses it. Never raises — offline/no-model safe (no callModel here).
        """
        context = context or {}
        # 1) proven structural denies win first (deterministic "do no harm" / authority)
        structural = _structural_denies(action, context)
        if structural:
            reason = structural[0]
            return _record(action, context, {"allow": False, "reason": reason,
                                             "basis": _label(reason, "proof")})
        # 2) explicit deny rules (heuristic pattern / predicate matching)
        for r in rules:
            reason = r["match"](action, context)
            if reason:
                return _record(action, context, {"allow": False, "reason": reason,
                                                 "basis": _label(reason, "conjecture")})
        # 3) nothing objected → allow
        return _record(action, context, {"allow": True, "reason": "allow-by-default", "basis": "proof"})

    def register_policy(rule):
        """Register a deny rule (callable predicate | substring str | dict). Persists to policy.json.

        Returns {count} — the number of live rules after adding. Store-confined write.
        """
        _add(rule, "register_policy")
        # persist the operator-authored subset so the constraint survives a reboot
        try:
            authored = [r["raw"] for r in rules if r["src"] in ("register_policy", "policy.json")]
            existing = host.store.read_json(_POLICY_FILE, default={}) or {}
            existing["deny"] = authored
            existing.setdefault("allow_by_default", True)
            host.store.write_json(_POLICY_FILE, existing)
        except Exception as e:
            host.log("policy_persist_error", detail=str(e)[:160])
        host.emit("policy.registered", {"src": "register_policy"})
        return {"count": len(rules)}

    def audit():
        """The decision trail: every check() and every observed tool.invoked, most-recent last."""
        return list(trail)

    # --- subscribe to tool invocations: record an alignment observation for each ---
    def _on_tool_invoked(payload):
        payload = payload or {}
        name = payload.get("name", "?")
        check(f"tool:{name}", {"event": "tool.invoked", "tool": name, "ok": payload.get("ok")})

    host.on("tool.invoked", _on_tool_invoked)

    host.log("module", step="expand-13", id=MODULE_ID, rules=len(rules))
    return {"check": check, "register_policy": register_policy, "audit": audit}


def _now():
    import time
    return int(time.time())
