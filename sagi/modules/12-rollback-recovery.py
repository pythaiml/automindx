# sagi/modules/12-rollback-recovery.py — expand-12 · "heal thyself" — SG6
#
# Gap: an autobuilt module that fails verify is left in modules/ where it can poison the next
# boot/load — the 26 specs describe self-healing (SFTRVE) but nothing enforces it. This closes it:
# when module-verifier emits {ok:false}, guard quarantines the bad file and restores modules/ to
# the last-good gitmind tree, so a bad autobuild can never corrupt the individual.
#
# Grounded on (proven reuse, not reinvented):
#   • host.memory (gitmind.GitMind): head()/commit() mark a moment; tree(commit) yields the exact
#     {filename -> content} of a prior snapshot — the source of truth we write back.
#   • module-verifier emits "module.verified" {file, ok, errors} after every gate (03-module-verifier).
#   • host.store.write / write_json — Store-confined writes (Store._safe re-checks the package boundary).
# New direction (flagged): gitmind today only READS trees; rollback_to() writes those blobs BACK to
# modules/ (restore), and quarantine() moves a failed file out via os.remove of the in-package source.
from __future__ import annotations

import os

MODULE_ID = "rollback-recovery"
DEPS = ["module-verifier", "memory-recall"]
MOTTO = "heal thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    mem = getattr(host, "memory", None)            # gitmind.GitMind, or None on a probe host
    store = host.store
    # last-good marker: the gitmind commit modules/ is known-safe at. Seeded to the current tip.
    state = {"last_good": mem.head() if mem is not None else None}

    def snapshot():
        """Mark the current modules/ tree as last-good and return its gitmind commit (or None)."""
        if mem is None:
            return None                            # degrade: no memory on an offline probe host
        commit = mem.commit(moment={"event": "rollback.snapshot"}, message="last-good")
        state["last_good"] = commit
        try:
            store.write_json("rollback.json", {"last_good": commit})
        except Exception:
            pass                                   # observability write must never break a snapshot
        host.log("rollback_snapshot", commit=commit)
        return commit

    def quarantine(name):
        """Move a (failed) module file out of modules/ into quarantine/. Returns the new rel path."""
        base = os.path.basename(name or "")
        if not base:
            return None
        src = os.path.join(store.modules_dir, base)
        content = store.read(os.path.join("modules", base))
        if content is None:
            return None                            # nothing to quarantine (already gone)
        dest_rel = os.path.join("quarantine", base)
        store.write(dest_rel, content)             # Store-confined copy under the package
        try:
            os.remove(src)                         # in-package source (rooted at store.modules_dir)
        except OSError:
            pass
        host.log("quarantine", file=base, dest=dest_rel)
        host.emit("module.quarantined", {"file": base, "dest": dest_rel})
        return dest_rel

    def rollback_to(commit):
        """Restore modules/ to a prior gitmind tree by writing its blobs back. Returns [restored]."""
        if mem is None or not commit:
            return []                              # degrade: nothing to restore without memory
        tree = mem.tree(commit)                    # {filename -> content} at that commit
        restored = []
        for fn, text in tree.items():
            base = os.path.basename(fn)
            store.write(os.path.join("modules", base), text)  # write blob back (Store-confined)
            restored.append(base)
        host.log("rollback_to", commit=commit, restored=len(restored))
        host.emit("module.rolledback", {"commit": commit, "restored": restored})
        return restored

    def guard(payload):
        """host.on('module.verified') handler: on ok=false quarantine + rollback; on ok=true snapshot."""
        payload = payload or {}
        ok = payload.get("ok")
        f = payload.get("file")
        if ok is False and f:
            quarantine(f)
            rollback_to(state["last_good"])        # heal: restore the known-good tree
            host.log("rollback_heal", file=os.path.basename(f), last_good=state["last_good"])
        elif ok is True:
            snapshot()                             # a clean verify becomes the new last-good marker

    host.on("module.verified", guard)              # auto-quarantine/heal on every gate result
    host.log("module", step="expand-12", id=MODULE_ID)
    return {"snapshot": snapshot, "quarantine": quarantine, "rollback_to": rollback_to, "guard": guard}
