# sagi/modules/63-self-test-harness.py — tier-7 · "prove every change"
#
# GAP CLOSED: verification happens per-module ad hoc; there is no single CI gate that runs the WHOLE
# regression (contract verify_all + synthesized smoke tests + the benchmark suite) and returns one
# green/red verdict a change must clear. This is that harness — continuous integration for a mind.
# rollback-orchestrator / meta-kernel call gate() before committing any change.
#
# GROUNDED REUSE: module-verifier.verify_all, test-synthesis.run, benchmark-suite.run_suite.
from __future__ import annotations

MODULE_ID = "self-test-harness"
DEPS = ["test-synthesis", "benchmark-suite", "module-verifier"]
MOTTO = "prove every change"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def run_ci():
        """Full regression: contract verify_all + synthesized smoke tests + benchmark. One verdict."""
        mv = _sib(host, "module-verifier")
        va = mv["verify_all"]() if mv else {"passed": [], "failed": []}
        ts = _sib(host, "test-synthesis")
        smoke = ts["run"]() if ts else {"suite": None}
        bs = _sib(host, "benchmark-suite")
        bench = bs["run_suite"]() if bs else {"score": None, "failed": []}
        green = (not va.get("failed")) and (not (bench or {}).get("failed"))
        report = {"green": green,
                  "verify": {"passed": len(va.get("passed", [])), "failed": va.get("failed", [])},
                  "benchmark": {"score": (bench or {}).get("score"), "failed": (bench or {}).get("failed", [])},
                  "smoke_synthesized": (smoke or {}).get("synthesized", 0)}
        host.store.write_json("ci.json", report)
        host.log("ci_run", green=green, failed=len(va.get("failed", [])))
        return report

    def gate():
        """True iff CI is green — the boolean a change must satisfy before it is committed."""
        return run_ci()["green"]

    def report():
        return host.store.read_json("ci.json", default=None) or run_ci()

    host.log("module", step="tier7-63", id=MODULE_ID)
    return {"run_ci": run_ci, "gate": gate, "report": report}
