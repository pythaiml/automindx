# sagi/modules/59-access-control.py — tier-6 · "guard thy gates"
#
# GAP CLOSED: policy-guard checks WHAT actions are allowed; nothing scopes WHO may call which
# capability. This adds role-based, capability-scoped access control: grant a role a set of
# capabilities, assign callers to roles, and check_access(caller, capability) before a sensitive
# call. It composes with policy-guard (policy is the what; access-control is the who). Persisted to acl.json.
#
# GROUNDED REUSE: policy-guard.check (composed after an access grant), tool-registry (capabilities to scope).
from __future__ import annotations

MODULE_ID = "access-control"
DEPS = ["policy-guard", "tool-registry"]
MOTTO = "guard thy gates"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _acl():
        return host.store.read_json("acl.json", default={"roles": {}, "callers": {}}) or {"roles": {}, "callers": {}}

    def grant(role, capabilities):
        """Define/extend a role with a set of capabilities."""
        acl = _acl()
        acl["roles"][role] = sorted(set(acl["roles"].get(role, []) + list(capabilities)))
        host.store.write_json("acl.json", acl)
        return {"role": role, "capabilities": acl["roles"][role]}

    def assign(caller, role):
        """Assign a caller (id) to a role."""
        acl = _acl()
        acl["callers"][caller] = role
        host.store.write_json("acl.json", acl)
        return {"caller": caller, "role": role}

    def check_access(caller, capability):
        """Allow iff the caller's role includes the capability AND policy-guard also allows it."""
        acl = _acl()
        role = acl["callers"].get(caller)
        caps = acl["roles"].get(role, []) if role else []
        allowed = capability in caps or "*" in caps
        reason = "granted" if allowed else f"caller '{caller}' (role {role}) lacks '{capability}'"
        if allowed:
            guard = _sib(host, "policy-guard")
            if guard:
                d = guard["check"](capability, {"caller": caller})
                if not d.get("allow", True):
                    allowed, reason = False, f"policy: {d.get('reason')}"
        host.log("access_check", caller=caller, capability=capability, allowed=allowed)
        return {"allow": allowed, "reason": reason, "role": role}

    def roles():
        return _acl()

    # seed the two standing principals: the owner (all) and untrusted (read-only)
    if not _acl()["roles"]:
        grant("owner", ["*"]); grant("guest", ["read_file", "recall", "map", "health"])
        assign(getattr(host, "identity_id", "sagi"), "owner")

    host.log("module", step="tier6-59", id=MODULE_ID)
    return {"grant": grant, "assign": assign, "check_access": check_access, "roles": roles}
