# sagi/modules/15-episodic-consolidation.py — expand-15 · "consolidate thyself"
#
# Gap it closes: the individual grows a raw local gitmind timeline (per-moment commits), and
# memory-recall can query it, but nothing ever *compresses* that timeline into higher-order
# semantic episodes. Without consolidation the timeline only gets longer and recall gets noisier.
# This module turns a window of recent LOCAL moments into a single semantic episode summary,
# embeds that summary into RAGE (best-effort), and marks a GLOBAL milestone — so memory scales
# episodic → semantic (vertical timeline → horizontal milestone chain), exactly the two-tier
# design gitmind already ships.
#
# Grounded on (proven reuse, not reinvented):
#   • memory-recall.timeline(limit) — the live newest-first view of the gitmind local timeline
#     (04-memory-recall.py); the raw material we consolidate.
#   • inference-router.call(prompt, kind) — the backend-agnostic reasoning path (14); returns a
#     clear "[inference-router: offline …]" marker when nothing is reachable, which we detect to
#     fall back to a deterministic extractive summary (no model, no network).
#   • gitmind.global_commit / read_commit / tree (runtime/gitmind.py) — the GLOBAL milestone tier
#     and the per-commit message+file view.
#   • rage_sync.RageSync.save_all (runtime/rage_sync.py) — best-effort embedding of the trees into
#     the RAGE semantic store; degrades to a no-op when pgvector/services aren't present.
#
# New logic (conjecture, kept small): the extractive-summary heuristic (dedup recent messages +
# rank files by touch frequency) and the episodes.json ledger. Summary quality is model-dependent
# and consolidation cadence is a policy choice — both are left to the caller.
from __future__ import annotations

import time

MODULE_ID = "episodic-consolidation"
DEPS = ["memory-recall", "inference-router"]
MOTTO = "consolidate thyself"

_EPISODES = "episodes.json"                 # in-package ledger of consolidated episode summaries
_OFFLINE = "[inference-router: offline"      # prefix of the router's offline marker (see #14)


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def _digest(moments):
    """A compact, deterministic digest of a window of local moments (newest first).

    Returns (messages, files): de-duplicated commit messages in timeline order and the files
    touched, ranked by how often they appear across the window. Pure reuse of the commit shape
    gitmind emits ({message, count, moment, ...}) — no model involved.
    """
    messages, seen = [], set()
    file_hits = {}
    for m in moments:
        msg = (m.get("message") or "").strip()
        if msg and msg not in seen:
            seen.add(msg)
            messages.append(msg)
        mom = m.get("moment") or {}
        # moments may name the file(s) they touched (reflection-journal, builders, …)
        for key in ("file", "files", "built"):
            v = mom.get(key)
            for fn in ([v] if isinstance(v, str) else (v or [])):
                if fn:
                    file_hits[fn] = file_hits.get(fn, 0) + 1
    files = [fn for fn, _ in sorted(file_hits.items(), key=lambda kv: (-kv[1], kv[0]))]
    return messages, files


def _extractive_summary(moments):
    """Deterministic offline summary of a window — no model, no network (proven fallback)."""
    messages, files = _digest(moments)
    if not moments:
        return "Episode: no local moments to consolidate yet."
    lines = [f"Episode over {len(moments)} recent local moment(s):"]
    for msg in messages[:8]:
        lines.append(f"  - {msg[:120]}")
    if not messages:
        lines.append("  - (no commit messages recorded)")
    if files:
        lines.append("Files most touched: " + ", ".join(files[:8]))
    return "\n".join(lines)


def _summary_prompt(moments):
    messages, files = _digest(moments)
    body = "\n".join(f"- {m[:200]}" for m in messages) or "- (no messages)"
    tail = ("\nFiles touched: " + ", ".join(files[:12])) if files else ""
    return (
        "Consolidate the following recent memory moments into ONE short semantic episode "
        "summary (3-5 sentences). State what changed and why it matters; do not invent facts.\n\n"
        f"Recent moments (newest first):\n{body}{tail}\n\nEpisode summary:"
    )


def activate(host):
    mem = getattr(host, "memory", None)                # gitmind.GitMind, or None on a probe host

    def _timeline(window):
        """Newest-first window of local moments — via memory-recall, else gitmind, else []."""
        mr = _sib(host, "memory-recall")
        if mr and callable(mr.get("timeline")):
            try:
                return mr["timeline"](window, "local") or []
            except Exception:
                pass
        if mem is not None:
            try:
                return mem.log(limit=window, scope="local")
            except Exception:
                return []
        return []

    def _embed_best_effort():
        """Best-effort RAGE embedding of the current trees. Never raises; no-op when unavailable."""
        if mem is None:
            return False
        try:
            from sagi.runtime.rage_sync import RageSync
            RageSync(mem).save_all()
            return True
        except Exception as e:
            host.log("consolidate_rage_unavailable", detail=str(e)[:120])
            return False

    def _append_episode(entry):
        ledger = host.store.read_json(_EPISODES, default=[])
        if not isinstance(ledger, list):
            ledger = []
        ledger.append(entry)
        host.store.write_json(_EPISODES, ledger)
        return ledger

    def consolidate(window=20):
        """Consolidate the last `window` local moments into one semantic episode summary.

        Routes the digest through inference-router.call (kind='summarize'); when the router is
        absent or reports offline, falls back to a deterministic extractive summary. The episode
        is embedded into RAGE (best-effort) and a GLOBAL milestone is marked. Returns the summary
        string. Never raises.
        """
        moments = _timeline(window)
        router = _sib(host, "inference-router")

        summary, source = None, "extractive"
        if router and callable(router.get("call")) and moments:
            try:
                out = router["call"](_summary_prompt(moments), "summarize")
                if out and not str(out).startswith(_OFFLINE):
                    summary, source = str(out).strip(), "model"
            except Exception as e:
                host.log("consolidate_router_failed", detail=str(e)[:120])
        if summary is None:
            summary = _extractive_summary(moments)          # deterministic offline path

        embedded = _embed_best_effort()
        milestone_commit = milestone(f"episode: consolidated {len(moments)} moment(s)")

        _append_episode({
            "ts": int(time.time()),
            "window": window,
            "moments": len(moments),
            "source": source,
            "embedded": embedded,
            "milestone": milestone_commit,
            "summary": summary,
        })
        host.log("consolidate", window=window, moments=len(moments), source=source,
                 embedded=embedded, milestone=milestone_commit)
        host.emit("episode.consolidated", {"moments": len(moments), "source": source})
        return summary

    def episodes():
        """All consolidated episode summaries, oldest first: [summary]."""
        ledger = host.store.read_json(_EPISODES, default=[])
        if not isinstance(ledger, list):
            return []
        return [e.get("summary", "") for e in ledger if isinstance(e, dict)]

    def milestone(label):
        """Mark a GLOBAL (expansion) milestone on the gitmind chain. Returns the commit hash or None."""
        if mem is None:
            host.log("milestone_skipped", reason="no memory", label=label)
            return None
        try:
            commit = mem.global_commit(
                moment={"event": "episode-milestone", "label": label},
                message=f"milestone: {label}")
            host.log("milestone", label=label, commit=commit)
            host.emit("milestone.marked", {"label": label, "commit": commit})
            return commit
        except Exception as e:
            host.log("milestone_failed", label=label, detail=str(e)[:120])
            return None

    host.log("module", step="expand-15", id=MODULE_ID)
    return {"consolidate": consolidate, "episodes": episodes, "milestone": milestone}
