# sagi/modules/57-notification.py — tier-6 · "reach the operator"
#
# GAP CLOSED: the individual can build a digest (human-interface) but has no OUTBOUND channel — no
# way to push an alert when something notable happens (anomaly, approval needed, milestone). This is
# that channel: notify() queues a message to an outbox and emits an event; subscribe() registers a
# named sink; outbox() drains pending messages. It never reaches the network itself (do no harm) —
# it stages notifications for whatever transport the host wires (console, wire-gateway, email adapter).
#
# GROUNDED REUSE: human-interface (approval alerts), telemetry (event source), host.on/emit bus.
from __future__ import annotations

import time

MODULE_ID = "notification"
DEPS = ["human-interface", "telemetry"]
MOTTO = "reach the operator"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    sinks = {}   # name -> callable(message)

    def _outbox():
        return host.store.read_json("outbox.json", default=[]) or []

    def notify(message, level="info"):
        """Stage a notification (queued to the outbox + emitted). Delivers to any live sinks."""
        box = _outbox()
        rec = {"message": message, "level": level, "ts": int(time.time()), "delivered": False}
        box.append(rec)
        host.store.write_json("outbox.json", box[-500:])
        host.emit("notification", {"message": message, "level": level})
        for name, fn in list(sinks.items()):
            try:
                fn(rec); rec["delivered"] = True
            except Exception:
                pass
        host.log("notify", level=level)
        return {"queued": True, "level": level}

    def subscribe(name, fn):
        """Register an outbound sink fn(record) — e.g. console print, wire-gateway push, email adapter."""
        if not callable(fn):
            raise ValueError("sink must be callable")
        sinks[name] = fn
        return {"subscribed": name}

    def outbox(pending_only=True):
        box = _outbox()
        return [m for m in box if (not pending_only or not m.get("delivered"))]

    # bridge: an enqueued approval or a fired anomaly becomes a notification automatically
    host.on("approval.enqueued", lambda p: notify(f"approval needed: {(p or {}).get('id')}", "action"))
    host.on("notification.milestone", lambda p: notify(f"milestone: {(p or {}).get('label')}", "info"))

    host.log("module", step="tier6-57", id=MODULE_ID)
    return {"notify": notify, "subscribe": subscribe, "outbox": outbox}
