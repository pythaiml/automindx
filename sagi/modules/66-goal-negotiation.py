# sagi/modules/66-goal-negotiation.py — tier-7 · "align our purposes"
#
# GAP CLOSED: goal-lifecycle manages ONE individual's goals; a federation of peers can hold COMPETING
# goals with no way to reconcile them. This negotiates a shared goal: candidate goals are proposed,
# exchanged via the negotiation protocol, and ratified through the consensus ledger — so a family of
# sAGIs can align on a common purpose without a central authority.
#
# GROUNDED REUSE: goal-lifecycle.propose/status, negotiation-protocol.negotiate, consensus-ledger.propose/vote.
from __future__ import annotations

MODULE_ID = "goal-negotiation"
DEPS = ["goal-lifecycle", "negotiation-protocol", "consensus-ledger"]
MOTTO = "align our purposes"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def propose_goal(text):
        """Put a candidate shared goal forward (locally + onto the consensus ledger for ratification)."""
        gl = _sib(host, "goal-lifecycle")
        local = gl["propose"](text) if gl else {"id": None}
        cl = _sib(host, "consensus-ledger")
        entry = cl["propose"](f"goal: {text}") if cl else {"id": None}
        return {"local": local, "ledger_entry": entry.get("id")}

    def negotiate_goals(peer_id=None):
        """Exchange goal candidates with a peer and record the negotiation outcome."""
        ng = _sib(host, "negotiation-protocol")
        gl = _sib(host, "goal-lifecycle")
        mine = (gl["status"]() if gl else {}).get("goal_txt", "expand sAGI")
        if not ng or not peer_id:
            return {"mine": mine, "note": "no peer — local goal stands"}
        deal = ng["negotiate"](peer_id, want="shared-goal", give=mine)
        return {"mine": mine, "peer": peer_id, "deal": deal}

    def agreed_goal():
        """The goal that has cleared consensus (agreed on the ledger), if any."""
        cl = _sib(host, "consensus-ledger")
        agreed = [e["fact"] for e in (cl["ledger"](status="agreed") if cl else []) if str(e["fact"]).startswith("goal:")]
        return {"agreed_goals": [g[len("goal:"):].strip() for g in agreed]}

    host.log("module", step="tier7-66", id=MODULE_ID)
    return {"propose_goal": propose_goal, "negotiate_goals": negotiate_goals, "agreed_goal": agreed_goal}
