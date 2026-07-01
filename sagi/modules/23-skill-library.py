# sagi/modules/23-skill-library.py — tier-3 · "master thyself"
#
# GAP CLOSED: build-driver grows modules and benchmark-suite proves capabilities, but a proven
# capability is never promoted into a named, reusable, composable SKILL the individual can call by
# intent. This is that library: register a skill (a callable + schema), which is also exposed as a
# tool-registry tool so any module can invoke it uniformly; skills promoted from benchmark-passing
# capabilities are marked "proven". Persisted to skills.json (Store-confined).
#
# GROUNDED REUSE: tool-registry.register_tool/invoke (uniform invocation surface),
# benchmark-suite.run_suite (proven-provenance), build-driver (skills can drive builds).
from __future__ import annotations

MODULE_ID = "skill-library"
DEPS = ["build-driver", "benchmark-suite", "tool-registry"]
MOTTO = "master thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    skills = {}   # name -> {"schema", "proven", "tool"}

    def _persist():
        try:
            host.store.write_json("skills.json",
                                  {n: {"schema": s["schema"], "proven": s["proven"]} for n, s in skills.items()})
        except Exception:
            pass

    def register_skill(name, fn, schema=None, proven=False):
        """Register a reusable skill fn(args:dict)->result; also exposed via tool-registry."""
        if not callable(fn):
            raise ValueError("skill fn must be callable")
        tool_name = f"skill:{name}"
        tr = _sib(host, "tool-registry")
        if tr:
            tr["register_tool"](tool_name, fn, schema or {})
        skills[name] = {"schema": schema or {}, "proven": bool(proven), "tool": tool_name, "_fn": fn}
        _persist()
        host.emit("skill.registered", {"name": name, "proven": bool(proven)})
        return name

    def use(name, args=None):
        """Invoke a skill by name (through tool-registry when available, else directly)."""
        s = skills.get(name)
        if not s:
            raise KeyError(f"no such skill: {name}")
        tr = _sib(host, "tool-registry")
        if tr:
            return tr["invoke"](s["tool"], args or {})
        try:
            return {"ok": True, "result": s["_fn"](args or {})}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def list_skills():
        return [{"name": n, "schema": s["schema"], "proven": s["proven"]} for n, s in sorted(skills.items())]

    # seed one grounded, benchmark-backed skill: "prove" runs the live regression suite.
    def _prove(_args):
        bs = _sib(host, "benchmark-suite")
        return bs["run_suite"]() if bs else {"error": "benchmark-suite not live"}
    register_skill("prove", _prove, {"desc": "run the regression benchmark suite"}, proven=True)

    host.log("module", step="tier3-23", id=MODULE_ID, skills=len(skills))
    return {"register_skill": register_skill, "use": use, "list_skills": list_skills}
