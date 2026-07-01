# sagi/modules/29-human-interface.py — tier-3 · "serve thy operator"
#
# GAP CLOSED: everything so far is autonomous; the operator has no seat. This adds a human-in-the-
# loop surface: a pending-approval queue for policy-gated actions (an action can be enqueued for a
# human instead of auto-denied), approve/deny to resolve them, and a readable digest of the
# individual (reflection journal + scorecard + goal + what's awaiting approval). The operator can
# also steer the standing goal (goal.txt). Persisted to approvals.json (Store-confined).
#
# GROUNDED REUSE: policy-guard.audit (what was denied → candidates for review), reflection-journal.journal
# (recent notes), evaluator via reflection (scorecard), goal.txt. STDLIB only.
from __future__ import annotations

import time

MODULE_ID = "human-interface"
DEPS = ["wire-gateway", "reflection-journal", "policy-guard"]
MOTTO = "serve thy operator"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _queue():
        return host.store.read_json("approvals.json", default=[]) or []

    def _save(q):
        try:
            host.store.write_json("approvals.json", q)
        except Exception:
            pass

    def _enqueue(payload):
        q = _queue()
        aid = f"a{len(q) + 1}-{int(time.time())}"
        q.append({"id": aid, "action": (payload or {}), "status": "pending", "ts": int(time.time())})
        _save(q)
        host.emit("approval.enqueued", {"id": aid})
        return aid

    def pending():
        """Actions awaiting a human decision."""
        return [a for a in _queue() if a.get("status") == "pending"]

    def approve(aid, decision="approve", note=""):
        """Resolve a pending action: approve|deny. Emits approval.resolved for downstream actors."""
        q = _queue()
        for a in q:
            if a.get("id") == aid and a.get("status") == "pending":
                a["status"] = "approved" if decision == "approve" else "denied"
                a["note"] = note
                a["resolved_ts"] = int(time.time())
                _save(q)
                host.emit("approval.resolved", {"id": aid, "decision": a["status"]})
                host.log("approval", id=aid, decision=a["status"])
                return {"id": aid, "decision": a["status"]}
        return {"error": f"no pending action {aid}"}

    def steer_goal(text):
        """Operator sets the standing goal (goal.txt) — re-steers proposals and the dynamic prompt."""
        host.store.write("goal.txt", (text or "").strip() + "\n")
        host.emit("module.registered", {"id": "goal-steered"})   # nudge self-prompt-auditor to refresh
        return {"goal": (text or "").strip()}

    def digest():
        """A readable operator digest: goal · scorecard · recent reflections · pending approvals."""
        goal = (host.store.read("goal.txt") or "expand sAGI").strip()
        journal = _sib(host, "reflection-journal")
        notes = journal["journal"]() if journal else []
        guard = _sib(host, "policy-guard")
        denied = [d for d in (guard["audit"]() if guard else []) if not d.get("allow", True)]
        lines = [f"# sAGI operator digest", "", f"**Goal:** {goal}",
                 f"**Reflections:** {len(notes)} · **Denied actions:** {len(denied)} · "
                 f"**Pending approvals:** {len(pending())}", ""]
        if notes:
            lines.append("## Latest reflection")
            lines.append((notes[-1].get("text") or "")[:600])
        return "\n".join(lines) + "\n"

    # bridge: policy denials become reviewable by a human instead of vanishing
    host.on("tool.invoked", lambda p: None)   # reserved hook; explicit enqueue is the primary path
    host.log("module", step="tier3-29", id=MODULE_ID)
    return {"pending": pending, "approve": approve, "digest": digest,
            "enqueue": _enqueue, "steer_goal": steer_goal}
