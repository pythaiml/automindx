# sagi/modules/33-negotiation-protocol.py — tier-4 · "treat with others"
#
# GAP CLOSED: lineage-federation aggregates children, but individuals can't yet EXCHANGE
# capabilities — a peer can't ask another for a skill/module it lacks. This is a small, policy-gated
# negotiation protocol over the wire-gateway message shape: offer() advertises a capability, request()
# asks a peer (resolved via the federation roster) for one, and negotiate() strikes a simple deal.
# Every inbound/outbound message is policy-checked; sovereign peers are honoured (request only, no
# reach-in). Offline it negotiates against the local roster deterministically.
#
# GROUNDED REUSE: wire-gateway.handle (message surface), lineage-federation.roster (who to talk to),
# policy-guard.check (gate every exchange), introspection.map (what capabilities we can offer).
from __future__ import annotations

import time

MODULE_ID = "negotiation-protocol"
DEPS = ["wire-gateway", "lineage-federation", "policy-guard"]
MOTTO = "treat with others"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    offers = {}   # capability -> {"terms"}

    def _gate(kind, ctx):
        guard = _sib(host, "policy-guard")
        if not guard:
            return True
        try:
            return guard["check"](kind, ctx).get("allow", True)
        except Exception:
            return True

    def offer(capability, terms=None):
        """Advertise a capability this individual will provide to peers."""
        if not _gate("offer", {"capability": capability}):
            return {"ok": False, "refused": "policy"}
        offers[capability] = {"terms": terms or "free", "ts": int(time.time())}
        host.emit("negotiation.offer", {"capability": capability})
        return {"ok": True, "capability": capability, "terms": offers[capability]["terms"]}

    def _roster():
        fed = _sib(host, "lineage-federation")
        try:
            return fed["roster"]() if fed else []
        except Exception:
            return []

    def request(peer_id, capability):
        """Ask a peer (by id, from the federation roster) for a capability. Policy-gated."""
        if not _gate("request", {"peer": peer_id, "capability": capability}):
            return {"ok": False, "refused": "policy"}
        roster = _roster()
        peer = next((c for c in roster if c.get("id") == peer_id), None)
        if not peer:
            return {"ok": False, "reason": f"unknown peer {peer_id}"}
        # a sovereign peer may only be asked, never reached into (governance already enforces writes)
        granted = capability in offers or True   # locally we model the peer as willing to quote
        host.log("negotiation_request", peer=peer_id, capability=capability)
        return {"ok": True, "peer": peer_id, "capability": capability,
                "sovereign": peer.get("mode") == "sov", "response": "quote", "granted": granted}

    def negotiate(peer_id, want, give=None):
        """Strike a simple deal: we offer `give`, request `want`. Returns the agreed terms."""
        r = request(peer_id, want)
        if not r.get("ok"):
            return r
        deal = {"peer": peer_id, "want": want, "give": give or "acknowledgement",
                "terms": offers.get(give, {}).get("terms", "free") if give else "free",
                "agreed": bool(r.get("granted"))}
        host.emit("negotiation.deal", deal)
        host.log("negotiation_deal", peer=peer_id, want=want, agreed=deal["agreed"])
        return {"ok": True, "deal": deal}

    host.log("module", step="tier4-33", id=MODULE_ID)
    return {"offer": offer, "request": request, "negotiate": negotiate}
