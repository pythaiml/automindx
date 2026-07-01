# sagi/modules/43-test-synthesis.py — tier-5 · "test what thou build"
#
# GAP CLOSED: benchmark-suite has a fixed set of cases; a freshly grown module gets no test of its
# own. This synthesises smoke tests from a module's live API — call each no-arg-safe method against
# the registry and assert it returns without raising — and registers them into benchmark-suite so
# every grown capability is continuously exercised.
#
# GROUNDED REUSE: introspection.describe (the API to test), benchmark-suite.add_case/run_suite,
# registry.get (invoke the live handle). STDLIB only.
from __future__ import annotations

MODULE_ID = "test-synthesis"
DEPS = ["benchmark-suite", "module-verifier", "introspection"]
MOTTO = "test what thou build"

# methods that are safe AND cheap to call with no args as a smoke test (read-only surfaces).
# Deliberately excludes methods that cascade into verify_all/evaluate (health/map/status/snapshot/
# widgets/report/audit) — those make a full-corpus smoke run O(n^2). CI keeps its own verify_all pass.
_SAFE = {"list_tools", "list_skills", "backends", "counters", "jobs", "ledger", "episodes",
         "sources", "beliefs", "weights", "journal", "roster", "frontier", "rationale",
         "identity_thread", "list_models", "list_prompts", "list_datasets", "list_plugins",
         "principles", "sessions", "outbox", "coverage", "changelog", "all"}


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def synthesize_tests(module_id):
        """Register a smoke-test case per no-arg-safe API method of a live module. Returns case names."""
        reg = getattr(host, "registry", None)
        handle = (reg["get"](module_id) or {}).get("handle") if reg else None
        bs = _sib(host, "benchmark-suite")
        if not (handle and bs):
            return {"added": []}
        added = []
        for name, fn in (handle.items() if isinstance(handle, dict) else []):
            if name in _SAFE and callable(fn):
                case = f"{module_id}.{name}"
                def make(fn=fn):
                    def _case():
                        try:
                            fn()
                            return True
                        except Exception:
                            return False
                    return _case
                try:
                    bs["add_case"](case, make())
                    added.append(case)
                except Exception:
                    pass
        host.log("test_synthesis", module=module_id, added=len(added))
        return {"added": added}

    def run(module_id=None):
        """Synthesise (for a module or all) then run the full benchmark suite."""
        intro = _sib(host, "introspection")
        targets = [module_id] if module_id else [m["id"] for m in (intro["map"]()["modules"] if intro else [])]
        total = 0
        for t in targets:
            total += len(synthesize_tests(t)["added"])
        bs = _sib(host, "benchmark-suite")
        return {"synthesized": total, "suite": bs["run_suite"]() if bs else None}

    def coverage():
        """Fraction of live modules that expose at least one smoke-testable method."""
        intro = _sib(host, "introspection")
        mods = intro["map"]()["modules"] if intro else []
        testable = [m for m in mods if any(a in _SAFE for a in m.get("api", []))]
        return {"testable": len(testable), "total": len(mods),
                "fraction": round(len(testable) / max(1, len(mods)), 3)}

    host.log("module", step="tier5-43", id=MODULE_ID)
    return {"synthesize_tests": synthesize_tests, "run": run, "coverage": coverage}
