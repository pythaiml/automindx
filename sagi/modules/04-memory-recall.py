# sagi/modules/04-memory-recall.py — expand-4 · "remember thyself"
#
# The individual already grows a memory (gitmind content-addressed tree, host.memory, attached
# by boot) and can embed it into RAGE (rage_sync.RageSync → pgvectorscale, SQLite fallback).
# But nothing exposes recall as a LIVE capability the reasoning loop can query, so each proposal
# is made blind to what was already grown. This module closes that: semantic recall when RAGE is
# available, keyword recall over the gitmind trees when it is not, plus a compact context block a
# proposal prompt can be conditioned on. Every path degrades — recall always returns something.
from __future__ import annotations

MODULE_ID = "memory-recall"
DEPS = ["module-loader"]
MOTTO = "remember thyself"


def activate(host):
    mem = getattr(host, "memory", None)          # gitmind.GitMind, or None on a probe host
    _rage = {"sync": None, "tried": False}

    def _rage_sync():
        """Lazily build a RageSync (best-effort). None if unavailable (offline / no DB)."""
        if _rage["tried"]:
            return _rage["sync"]
        _rage["tried"] = True
        try:
            from sagi.runtime.rage_sync import RageSync
            _rage["sync"] = RageSync(mem) if mem is not None else None
        except Exception as e:
            host.log("recall_rage_unavailable", detail=str(e)[:120])
            _rage["sync"] = None
        return _rage["sync"]

    def _keyword_recall(query, k):
        """Fallback: substring match over the module forms in the latest gitmind tree."""
        if mem is None:
            return []
        head = mem.head()
        tree = mem.tree(head) if head else {}
        q = (query or "").lower()
        hits = []
        for fn, content in tree.items():
            score = sum(content.lower().count(w) for w in q.split() if w)
            if score:
                hits.append({"file": fn, "score": score, "text": content[:400]})
        hits.sort(key=lambda h: -h["score"])
        return hits[:k]

    def recall(query, k=5):
        """Recall prior grown knowledge: semantic (RAGE) if available, else keyword over gitmind."""
        sync = _rage_sync()
        if sync is not None:
            try:
                sync.save_all()                  # make sure the current trees are embedded
                res = sync.search(query, limit=k)
                if res:
                    return res
            except Exception as e:
                host.log("recall_search_failed", detail=str(e)[:120])
        return _keyword_recall(query, k)

    def context_for(goal, k=5):
        """A compact recall block to prepend to a proposal prompt (conditions the next step)."""
        hits = recall(goal, k)
        if not hits:
            return ""
        lines = ["## Recalled from your own growth (for grounding):"]
        for h in hits:
            tag = h.get("file") or h.get("commit", "")[:8] or "memory"
            snippet = (h.get("text") or "").strip().replace("\n", " ")[:200]
            lines.append(f"- [{tag}] {snippet}")
        return "\n".join(lines)

    def timeline(limit=20, scope=None):
        """The individual's memory timeline, newest first (gitmind commits)."""
        return mem.log(limit=limit, scope=scope) if mem is not None else []

    host.log("module", step="expand-4", id=MODULE_ID)
    return {"recall": recall, "context_for": context_for, "timeline": timeline}
