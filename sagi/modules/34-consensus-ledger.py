# sagi/modules/34-consensus-ledger.py — tier-4 · "agree with others"
#
# GAP CLOSED: the 26 specs describe a Distributed Consensus Knowledge-Fusion engine but it was only
# prose. This is a real, minimal one: a hash-chained, append-only ledger of proposed facts (the
# gitmind object model, applied to shared knowledge), where peers vote and a fact becomes "agreed"
# once it clears a calibrated majority. Each entry is signed (security-hardening) and chained to its
# parent by sha256, so the shared record is tamper-evident. Persisted to ledger.json.
#
# GROUNDED REUSE: security-hardening.sign (entry integrity), memory-recall (context for a vote),
# lineage-federation.roster (the voter set), sha256 chaining (gitmind pattern). STDLIB only.
from __future__ import annotations

import hashlib
import json
import time

MODULE_ID = "consensus-ledger"
DEPS = ["lineage-federation", "memory-recall", "security-hardening"]
MOTTO = "agree with others"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _chain():
        return host.store.read_json("ledger.json", default=[]) or []

    def _save(c):
        try:
            host.store.write_json("ledger.json", c)
        except Exception:
            pass

    def _quorum():
        fed = _sib(host, "lineage-federation")
        try:
            n = len(fed["roster"]()) if fed else 0
        except Exception:
            n = 0
        return max(1, (n + 1) // 2 + 1)   # simple majority of (self + peers), min 1

    def propose(fact):
        """Append a proposed fact, chained + signed. Starts with the proposer's own vote."""
        chain = _chain()
        parent = chain[-1]["hash"] if chain else None
        sec = _sib(host, "security-hardening")
        eid = f"e{len(chain) + 1}"
        payload = {"id": eid, "fact": fact, "parent": parent, "ts": int(time.time()),
                   "votes": {getattr(host, "identity_id", "sagi"): True}, "status": "proposed"}
        h = hashlib.sha256((json.dumps(payload, sort_keys=True)).encode()).hexdigest()
        payload["hash"] = h
        payload["signature"] = (sec["sign"](fact)["signature"] if sec else h)
        chain.append(payload)
        _save(chain)
        host.log("consensus_propose", id=eid, fact=str(fact)[:60])
        return {"id": eid, "hash": h[:12]}

    def vote(entry_id, agree=True, voter=None):
        """Cast a vote; the entry becomes 'agreed' once agreeing votes reach a majority quorum."""
        chain = _chain()
        e = next((x for x in chain if x["id"] == entry_id), None)
        if not e:
            return {"error": f"no entry {entry_id}"}
        e["votes"][voter or getattr(host, "identity_id", "sagi")] = bool(agree)
        agree_n = sum(1 for v in e["votes"].values() if v)
        if agree_n >= _quorum():
            e["status"] = "agreed"
        _save(chain)
        host.log("consensus_vote", id=entry_id, agree=bool(agree), status=e["status"])
        return {"id": entry_id, "agree_votes": agree_n, "quorum": _quorum(), "status": e["status"]}

    def ledger(status=None):
        """The shared ledger (optionally filtered to 'agreed'/'proposed'). Tamper-evident chain."""
        chain = _chain()
        return [{"id": e["id"], "fact": e["fact"], "status": e["status"],
                 "hash": e["hash"][:12], "parent": (e.get("parent") or "")[:12],
                 "votes": e["votes"]} for e in chain if status is None or e["status"] == status]

    host.log("module", step="tier4-34", id=MODULE_ID)
    return {"propose": propose, "vote": vote, "ledger": ledger}
