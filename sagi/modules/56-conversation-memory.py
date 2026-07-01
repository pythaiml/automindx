# sagi/modules/56-conversation-memory.py — tier-6 · "remember our talk"
#
# GAP CLOSED: memory-recall remembers MODULES (what sAGI grew); nothing remembers CONVERSATIONS —
# the dialog turns with an operator or peer. This adds episodic dialog memory: append turns to a
# session, recall relevant past turns, and list sessions — so the individual has continuity of
# conversation, not just of code. Turns persist under conversations/ and feed episodic consolidation.
#
# GROUNDED REUSE: memory-recall.recall (semantic search over turns once committed), episodic-
# consolidation (long-dialog summarisation), host.memory.commit. STDLIB only.
from __future__ import annotations

import time

MODULE_ID = "conversation-memory"
DEPS = ["memory-recall", "episodic-consolidation"]
MOTTO = "remember our talk"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _all():
        return host.store.read_json("conversations.json", default={}) or {}

    def append(session, role, text):
        """Append a dialog turn {role, text, ts} to a session."""
        convos = _all()
        convos.setdefault(session, []).append({"role": role, "text": text, "ts": int(time.time())})
        convos[session] = convos[session][-500:]          # bound each session
        host.store.write_json("conversations.json", convos)
        return {"session": session, "turns": len(convos[session])}

    def recall_dialog(session, query=None, k=10):
        """Recent turns of a session, optionally keyword-filtered."""
        turns = _all().get(session, [])
        if query:
            q = query.lower()
            turns = [t for t in turns if q in t["text"].lower()]
        return turns[-k:]

    def sessions():
        return {s: len(t) for s, t in _all().items()}

    def summarize(session):
        """Consolidate a long session into a semantic episode (via episodic-consolidation)."""
        ec = _sib(host, "episodic-consolidation")
        turns = _all().get(session, [])
        if not ec or not turns:
            return {"summary": None, "turns": len(turns)}
        # feed the dialog into memory so consolidation can pick it up
        if getattr(host, "memory", None) is not None:
            try:
                host.memory.commit(moment={"event": "dialog", "session": session, "turns": len(turns)},
                                   message=f"dialog {session}")
            except Exception:
                pass
        return {"summary": bool(ec["consolidate"]()), "turns": len(turns)}

    host.log("module", step="tier6-56", id=MODULE_ID)
    return {"append": append, "recall_dialog": recall_dialog, "sessions": sessions, "summarize": summarize}
