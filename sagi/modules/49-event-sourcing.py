# sagi/modules/49-event-sourcing.py — tier-5 · "replay thyself"
#
# GAP CLOSED: .history/build.jsonl is an append-only event log, but nothing treats it AS an event
# source — the individual can't replay its history to reconstruct a projection of "what happened".
# This does: events() reads the log, project() folds it with a reducer into any derived state, and
# replay() gives a ready-made projection (event counts, modules built, backends used, timeline span).
# Combined with gitmind (state snapshots), this closes the loop from event log → reconstructed state.
#
# GROUNDED REUSE: .history/build.jsonl (the event stream), host.memory (state snapshots). STDLIB only.
from __future__ import annotations

import json
import os

MODULE_ID = "event-sourcing"
DEPS = ["memory-recall"]
MOTTO = "replay thyself"


def activate(host):
    def events(limit=None, kind=None):
        """The raw event stream from .history/build.jsonl (newest last), optionally filtered by kind."""
        path = os.path.join(host.root, ".history", "build.jsonl")
        out = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if kind is None or e.get("event") == kind:
                        out.append(e)
        except OSError:
            pass
        return out[-limit:] if limit else out

    def project(reducer, initial=None):
        """Fold the event stream with reducer(acc, event) -> acc. Pure reconstruction."""
        acc = initial if initial is not None else {}
        for e in events():
            try:
                acc = reducer(acc, e)
            except Exception:
                pass
        return acc

    def replay():
        """A ready projection: event counts, modules built, backends seen, and the timeline span."""
        evs = events()
        counts, modules, backends, ts = {}, [], set(), []
        for e in evs:
            counts[e.get("event", "?")] = counts.get(e.get("event", "?"), 0) + 1
            if e.get("event") in ("module", "build_next") and (e.get("id") or e.get("file")):
                modules.append(e.get("id") or e.get("file"))
            if e.get("backend"):
                backends.add(e["backend"])
            if e.get("ts"):
                ts.append(e["ts"])
        return {"total_events": len(evs), "by_event": counts, "modules_built": len(modules),
                "backends": sorted(backends), "span": [min(ts), max(ts)] if ts else None}

    host.log("module", step="tier5-49", id=MODULE_ID)
    return {"events": events, "project": project, "replay": replay}
