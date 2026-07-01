# sagi/runtime/rage_sync.py — save the gitmind trees into RAGE.
#
# Persists an individual sAGI's memory tree (sagi/runtime/gitmind.py) into the RAGE
# semantic store — pgvectorscale + all-MiniLM-L6-v2 embeddings, the mindX rage
# pattern (services/rage_memory.py). Each distinct module form becomes an embedded
# row tagged with the commit, timestamp, and scope it belongs to, so memory is both
# time-travellable (via .history moments) and semantically searchable. If pgvector
# isn't configured, services.memory.get_memory() falls back to SQLite keyword
# search, so saving the trees always works.
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


class RageSync:
    def __init__(self, gitmind, memory=None, session: str = "sagi-memory"):
        self.gm = gitmind
        if memory is None:
            from services.memory import get_memory   # RAGE (pgvector) or SQLite fallback
            memory = get_memory()
        self.mem = memory
        self.session = session

    def save_commit(self, commit_hash: str, _seen: Optional[set] = None) -> int:
        """Embed every module form at a commit into RAGE (skipping already-saved forms)."""
        c = self.gm.read_commit(commit_hash)
        if not c:
            return 0
        tree = self.gm.tree(commit_hash)
        n = 0
        for fn, content in tree.items():
            key = hashlib.sha256((fn + "\0" + content).encode()).hexdigest()
            if _seen is not None:
                if key in _seen:
                    continue
                _seen.add(key)
            self.mem.append(self.session, "memory", {
                "text": content,
                "file": fn,
                "commit": commit_hash,
                "ts": c.get("ts"),
                "scope": c.get("scope", "local"),
                "moment": c.get("moment", {}),
            })
            n += 1
        return n

    def save_all(self, scope: Optional[str] = None) -> int:
        """Save every tree in the history (or only local/global) into RAGE."""
        seen: set = set()
        total = 0
        for c in self.gm.log(scope=scope):
            total += self.save_commit(c["commit"], _seen=seen)
        return total

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Semantic (pgvector) or keyword (SQLite) recall over the saved trees."""
        return self.mem.search(query, self.session, limit=limit)


if __name__ == "__main__":
    import argparse
    import json
    import os

    ap = argparse.ArgumentParser(description="Save an individual sAGI's gitmind trees into RAGE")
    ap.add_argument("--dir", default=os.environ.get("SAGI_DIR", "./sagi"))
    ap.add_argument("--search", default=None, help="query the saved trees instead of saving")
    a = ap.parse_args()
    from .gitmind import GitMind
    sync = RageSync(GitMind(os.path.abspath(a.dir)))
    if a.search:
        for hit in sync.search(a.search):
            print(json.dumps({k: hit.get(k) for k in ("file", "commit", "ts", "scope")}), "—", (hit.get("text") or "")[:80])
    else:
        print(f"saved {sync.save_all()} module form(s) into RAGE ({type(sync.mem).__name__})")
