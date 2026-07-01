# sagi/modules/31-scheduler.py — tier-4 · "keep thy rhythm"
#
# GAP CLOSED: the individual can autobuild, benchmark, and consolidate — but only when a human
# triggers it. This gives it a RHYTHM: register periodic jobs (autobuild / benchmark / consolidate)
# and advance a logical clock with tick(); due jobs fire and their outcome is fed to outcome-learning
# so the schedule itself is tuned. A host loop (aglm.AutonomousLoop) calls tick(); offline it is a
# pure, testable logical-time stepper (no wall-clock, no threads).
#
# GROUNDED REUSE: build-driver.autobuild, benchmark-suite.run_suite (via a job), curriculum.advance,
# episodic-consolidation.consolidate, outcome-learning.record (tune cadence). STDLIB only.
from __future__ import annotations

MODULE_ID = "scheduler"
DEPS = ["curriculum", "build-driver", "outcome-learning"]
MOTTO = "keep thy rhythm"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    state = host.store.read_json("schedule.json", default={"clock": 0, "jobs": []}) or {"clock": 0, "jobs": []}

    def _persist():
        try:
            host.store.write_json("schedule.json", state)
        except Exception:
            pass

    def _run_job(kind):
        """Fire one job by kind; returns (ok, detail). Every path guarded/offline-safe."""
        if kind == "autobuild":
            bd = _sib(host, "build-driver")
            r = bd["autobuild"](1) if bd else []
            return (bool(r), {"built": len(r)})
        if kind == "benchmark":
            bs = _sib(host, "benchmark-suite")
            s = bs["run_suite"]() if bs else {}
            return (isinstance(s, dict) and not s.get("failed"), {"score": (s or {}).get("score")})
        if kind == "consolidate":
            ec = _sib(host, "episodic-consolidation")
            s = ec["consolidate"]() if ec else None
            return (bool(s), {"summary": bool(s)})
        return (False, {"error": f"unknown job {kind}"})

    def schedule(kind, every=1):
        """Register a periodic job (kind: autobuild|benchmark|consolidate) firing every N ticks."""
        jid = f"j{len(state['jobs']) + 1}"
        state["jobs"].append({"id": jid, "kind": kind, "every": max(1, int(every)), "last": 0, "runs": 0})
        _persist()
        return {"id": jid, "kind": kind, "every": every}

    def tick(n=1):
        """Advance the logical clock n steps; fire every due job once per due step. Returns fired jobs."""
        learn = _sib(host, "outcome-learning")
        fired = []
        for _ in range(max(1, int(n))):
            state["clock"] += 1
            for j in state["jobs"]:
                if state["clock"] - j["last"] >= j["every"]:
                    ok, detail = _run_job(j["kind"])
                    j["last"] = state["clock"]; j["runs"] += 1
                    if learn:
                        try: learn["record"](f"job:{j['kind']}", ok)
                        except Exception: pass
                    fired.append({"id": j["id"], "kind": j["kind"], "ok": ok, **detail})
        _persist()
        host.log("scheduler_tick", clock=state["clock"], fired=len(fired))
        return {"clock": state["clock"], "fired": fired}

    def jobs():
        return {"clock": state["clock"], "jobs": list(state["jobs"])}

    host.log("module", step="tier4-31", id=MODULE_ID)
    return {"schedule": schedule, "tick": tick, "jobs": jobs}
