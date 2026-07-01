# sagi/runtime/gitmind.py — memory that grows as a tree.
#
# An internal, content-addressed version store (git-like but server-less): every
# meaningful moment snapshots the individual's modules/ into blobs + a tree + a
# commit chained to its parent, so memory can be revisited from any moment in
# .history. It is the in-package analogue of a self-hosted forge (Forgejo) and of
# mindX's /mindx/gitmind — minus the server. Objects live under
# <SAGI_DIR>/.gitmind/objects/ (deduplicated by sha256); HEAD names the tip commit.
#
# Upstream: https://github.com/Professor-Codephreak/gitmind — this is the sAGI
# in-runtime port. Innovations here that extend gitmind and can flow back upstream:
#   • two commit tiers — LOCAL (per-timestamp timeline, vertical scaling) vs GLOBAL
#     (expansion / massive-upgrade milestones, horizontal scaling), on their own ref;
#   • .history alignment — snapshots are reachable by .history timestamp (at_moment);
#   • RAGE embedding — trees are indexable into pgvectorscale for semantic recall
#     (rage_sync.RageSync; the mindX RAGE substrate, https://rage.pythai.net).
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional


class GitMind:
    # Both the grown spec prose (.md) and the executable modules (.py) that make those
    # capabilities live are memory — a snapshot of the individual must capture both.
    SNAPSHOT_SUFFIXES = (".md", ".py")

    def __init__(self, root: str, source_dir: str = "modules", on_commit=None,
                 suffixes=SNAPSHOT_SUFFIXES):
        self.root = os.path.abspath(root)
        self.gm = os.path.join(self.root, ".gitmind")
        self.objects = os.path.join(self.gm, "objects")
        self.head_path = os.path.join(self.gm, "HEAD")
        self.global_path = os.path.join(self.gm, "GLOBAL")   # tip of the global chain
        self.src = os.path.join(self.root, source_dir)
        self.suffixes = tuple(suffixes)                      # which module forms are memory
        # Optional RAGE sink: called (commit_hash, commit_obj) after each new commit,
        # e.g. to embed the tree into pgvectorscale (see rage_sync.RageSync).
        self.on_commit = on_commit
        os.makedirs(self.objects, exist_ok=True)

    # --- content-addressed object store (dedup by hash) ---
    def _put(self, data: bytes) -> str:
        h = hashlib.sha256(data).hexdigest()
        p = os.path.join(self.objects, h)
        if not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(data)
        return h

    def _get(self, h: str) -> bytes:
        with open(os.path.join(self.objects, h), "rb") as f:
            return f.read()

    def _snapshot_tree(self):
        entries: Dict[str, str] = {}
        try:
            for fn in sorted(os.listdir(self.src)):
                if fn.endswith(self.suffixes) and not fn.startswith((".", "_")):
                    with open(os.path.join(self.src, fn), "rb") as f:
                        entries[fn] = self._put(f.read())      # blob per module (spec + code)
        except OSError:
            pass
        tree_hash = self._put(json.dumps(entries, sort_keys=True).encode())
        return tree_hash, entries

    def _read_ref(self, path: str) -> Optional[str]:
        try:
            return open(path, encoding="utf-8").read().strip() or None
        except OSError:
            return None

    def head(self) -> Optional[str]:
        return self._read_ref(self.head_path)

    def global_head(self) -> Optional[str]:
        return self._read_ref(self.global_path)

    # --- commit a moment ---
    # scope='local'  → the per-timestamp timeline (vertical scaling: this individual
    #                  deepening over time); the default, deduped on no-op.
    # scope='global' → reserved for expansion and massive upgrade moves (horizontal
    #                  scaling: scaling out/up); always recorded, and also advances
    #                  the GLOBAL ref so the milestone chain is walkable on its own.
    def commit(self, moment: Any = None, message: str = "", ts: Optional[int] = None,
               scope: str = "local") -> str:
        tree_hash, entries = self._snapshot_tree()
        parent = self.head()
        if scope == "local" and parent:
            pc = self.read_commit(parent)
            if pc and pc.get("tree") == tree_hash:
                return parent                                  # nothing changed → no empty local commit
        obj = {
            "type": "commit", "tree": tree_hash, "parent": parent,
            "global_parent": self.global_head() if scope == "global" else None,
            "ts": int(ts if ts is not None else time.time()),
            "moment": moment or {}, "message": message, "count": len(entries), "scope": scope,
        }
        ch = self._put(json.dumps(obj, sort_keys=True).encode())
        with open(self.head_path, "w", encoding="utf-8") as f:
            f.write(ch)
        if scope == "global":
            with open(self.global_path, "w", encoding="utf-8") as f:
                f.write(ch)
        if self.on_commit:
            try:
                self.on_commit(ch, obj)                        # e.g. embed the tree into RAGE
            except Exception:
                pass                                            # a RAGE sink must never break a commit
        return ch

    def global_commit(self, moment: Any = None, message: str = "", ts: Optional[int] = None) -> str:
        """Mark an expansion / massive-upgrade milestone (horizontal scaling)."""
        return self.commit(moment=moment, message=message, ts=ts, scope="global")

    def read_commit(self, h: str) -> Optional[Dict[str, Any]]:
        try:
            o = json.loads(self._get(h))
            return o if isinstance(o, dict) and o.get("type") == "commit" else None
        except Exception:
            return None

    # --- navigate ---
    def log(self, limit: int = 1000, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        h = self.head()
        while h and len(out) < limit:
            c = self.read_commit(h)
            if not c:
                break
            if scope is None or c.get("scope", "local") == scope:
                out.append({"commit": h, **c})
            h = c.get("parent")
        return out                                             # newest first

    def global_log(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Walk only the expansion / massive-upgrade milestones (the GLOBAL chain)."""
        out: List[Dict[str, Any]] = []
        h = self.global_head()
        while h and len(out) < limit:
            c = self.read_commit(h)
            if not c:
                break
            out.append({"commit": h, **c})
            h = c.get("global_parent")
        return out

    def tree(self, commit_hash: str) -> Dict[str, str]:
        """The memory state (filename -> content) at a commit."""
        c = self.read_commit(commit_hash)
        if not c:
            return {}
        entries = json.loads(self._get(c["tree"]))
        return {fn: self._get(bh).decode("utf-8", "ignore") for fn, bh in entries.items()}

    def at_moment(self, ts: int) -> Optional[Dict[str, Any]]:
        """The memory snapshot as it was at a .history moment (latest commit ≤ ts)."""
        for c in self.log():
            if c["ts"] <= ts:
                return c
        return None

    def tree_at_moment(self, ts: int) -> Dict[str, str]:
        c = self.at_moment(ts)
        return self.tree(c["commit"]) if c else {}
