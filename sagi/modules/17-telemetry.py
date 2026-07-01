# sagi/modules/17-telemetry.py — tier-2 · "know thy pulse"
#
# Gap it closes: 16-introspection gives a point-in-time self-map, but nothing watches the
# individual's pulse OVER TIME. Modules already broadcast their whole lifecycle on the host bus
# (module.registered/persisted/verified, tool.invoked/registered, swarm.delegated) yet those
# signals evaporate — no one aggregates them. This subscribes to the live bus, tallies per-event
# counters, keeps a rolling record stream, and persists notable moments to .history so growth is
# observable, not just a scorecard snapshot.
#
# Grounded reuse (nothing reinvented):
#   - host.on(event, fn)            -> subscribe to the real pub/sub bus (host.py)
#   - host.emit(...) call-sites     -> the known emitters we default-track (verified by grep):
#         module_registry.register  -> "module.registered"
#         boot / gitmind persist    -> "module.persisted"
#         03-module-verifier        -> "module.verified"
#         06-tool-registry          -> "tool.invoked", "tool.registered"
#         10-swarm-orchestrator     -> "swarm.delegated"
#   - host.log(event, **fields)     -> append notable records to .history/build.jsonl
#   - introspection (dep) reached via _sib, used only to stamp the live module count (guarded)
# New logic (small): a counter dict + a bounded record deque, and idempotent per-event subscribe.
from __future__ import annotations

from collections import deque

MODULE_ID = "telemetry"
DEPS = ["introspection"]
MOTTO = "know thy pulse"

# The known emitters worth aggregating (the conjecture's starting set; grep-verified above).
DEFAULT_EVENTS = [
    "module.registered",
    "module.persisted",
    "module.verified",
    "tool.invoked",
    "tool.registered",
    "swarm.delegated",
]

_STREAM_CAP = 512   # bounded rolling window so telemetry never grows without limit


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    counts = {}                       # event -> n
    records = deque(maxlen=_STREAM_CAP)   # rolling [{ts, event, payload}]
    subscribed = set()                # events already wired to host.on (idempotent track)

    def _now():
        import time
        return int(time.time())

    def _observe(event, payload):
        """One bus hit: tally it, append to the rolling stream, persist the moment."""
        counts[event] = counts.get(event, 0) + 1
        rec = {"ts": _now(), "event": event, "payload": payload}
        records.append(rec)
        # Persist notable records to .history (host.log doesn't emit -> no feedback loop).
        try:
            host.log("telemetry", event=event, n=counts[event])
        except Exception:
            pass
        return rec

    def track(events=None):
        """Subscribe to host.on for each event (defaults to the known emitters).

        Idempotent: an already-tracked event is not re-subscribed. Returns the full set of
        events currently under observation. Degrades gracefully if host.on is absent.
        """
        wanted = list(events) if events else list(DEFAULT_EVENTS)
        on = getattr(host, "on", None)
        for ev in wanted:
            if ev in subscribed:
                continue
            if callable(on):
                # bind ev per-iteration via default arg so each handler keeps its own event name
                on(ev, lambda payload, _ev=ev: _observe(_ev, payload))
            subscribed.add(ev)
        host.log("telemetry_track", events=sorted(subscribed))
        return sorted(subscribed)

    def counters():
        """Current per-event tallies: {event: n} (a copy; callers can't mutate our state)."""
        return dict(counts)

    def stream(limit=50):
        """The most recent telemetry records, newest last, capped at `limit`."""
        try:
            n = int(limit)
        except (TypeError, ValueError):
            n = 50
        if n <= 0:
            return []
        return list(records)[-n:]

    # Start observing the known emitters immediately so telemetry is live on boot. Any module
    # that registers/verifies/persists AFTER us is captured without an explicit track() call.
    track()

    # Guarded introspection stamp — proves the dep is reachable, but telemetry works without it.
    intro = _sib(host, "introspection")
    live_n = None
    if intro:
        try:
            live_n = len(intro["describe"](MODULE_ID).get("api", []))
        except Exception:
            live_n = None

    host.log("module", step="tier2-17", id=MODULE_ID, tracked=len(subscribed), intro=live_n)
    return {"track": track, "counters": counters, "stream": stream}
