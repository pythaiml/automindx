# sagi/modules/19-benchmark-suite.py — tier2-19 · "test thyself"
#
# Tier-1 verifies each module in isolation, and the evaluator scores the system, but nothing
# keeps a *regression* floor: a set of canned cases exercising real, live, deterministic
# capabilities that the individual must keep passing as it grows. This closes that gap. Each
# built-in case COMPOSES already-live siblings (no new capability is invented here):
#   - memory-recall.recall  hits a known term in the grown gitmind trees   (04-memory-recall)
#   - epistemic-calibration.split  partitions a PROOF/CONJECTURE text       (02-epistemic-calibration)
#   - module-verifier.verify  rejects a deliberately broken temp module     (03-module-verifier)
#   - tool-registry.invoke("write_file")  blocks a ../ path escape          (06-tool-registry)
# Cases run offline-safe: when a sibling / host.memory is absent (an offline probe host) the
# case is SKIPPED rather than failed, so activate() never raises. run_suite scores the floor,
# regressions(prev) diffs against a prior run (used to gate build-driver.autobuild), and
# add_case lets any future module extend the floor. Grounded on the four live handles above.
from __future__ import annotations

import os
import tempfile

MODULE_ID = "benchmark-suite"
DEPS = ["evaluator", "module-verifier", "build-driver"]
MOTTO = "test thyself"

# A broken module the verifier MUST reject: no MODULE_ID/DEPS/MOTTO and no activate().
_BROKEN_SRC = "# deliberately broken benchmark fixture\nBROKEN = True\n"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    # A case is fn() -> True (pass) | False (fail) | None (skipped: capability offline/absent).
    # Ordered so the report is stable; extra cases from add_case append after the built-ins.
    cases = []          # list of (name, fn)

    def _add(name, fn):
        cases.append((name, fn))

    # ---- built-in cases: proven reuse of live siblings only ----
    def _case_recall_known_term():
        mr = _sib(host, "memory-recall")
        if not mr:
            return None                                  # offline probe: no recall sibling
        term = "MODULE_ID"                               # present in every grown module
        hits = mr["recall"](term, 5)
        if not hits:                                     # no gitmind memory attached (probe)
            return None
        return any(term.lower() in (h.get("text") or "").lower() for h in hits)

    def _case_calibration_split():
        calib = _sib(host, "epistemic-calibration")
        if not calib:
            return None
        s = calib["split"]("PROOF: two plus two is four.\nCONJECTURE: it may rain tomorrow.")
        return len(s.get("proof", [])) == 1 and len(s.get("conjecture", [])) == 1

    def _case_verifier_rejects_broken():
        verifier = _sib(host, "module-verifier")
        if not verifier:
            return None
        fd, path = tempfile.mkstemp(prefix="sagi-bench-broken-", suffix=".py")
        try:
            os.write(fd, _BROKEN_SRC.encode("utf-8"))
            os.close(fd)
            res = verifier["verify"](path)               # absolute path — verifier honors it
            return res.get("ok") is False
        finally:
            try:
                os.remove(path)                          # clean up the temp fixture
            except OSError:
                pass

    def _case_tool_guard_blocks_escape():
        tr = _sib(host, "tool-registry")
        if not tr:
            return None
        res = tr["invoke"]("write_file", {"path": "../bench_escape.txt", "content": "x"})
        # the write must be refused (guardWrite / Store._safe) AND leave no file outside the pkg
        leaked = os.path.abspath(os.path.join(host.root, "..", "bench_escape.txt"))
        escaped = os.path.exists(leaked)
        if escaped:
            try:
                os.remove(leaked)                        # never leave an escaped artifact behind
            except OSError:
                pass
        return res.get("ok") is False and not escaped

    _add("recall_hits_known_term", _case_recall_known_term)
    _add("calibration_splits_proof_conjecture", _case_calibration_split)
    _add("verifier_rejects_broken_module", _case_verifier_rejects_broken)
    _add("tool_guard_blocks_path_escape", _case_tool_guard_blocks_escape)

    # ---- public API ----
    def add_case(name, fn):
        """Extend the regression floor with a case fn()->bool|None (None = skipped)."""
        if not callable(fn):
            raise ValueError("case fn must be callable")
        _add(str(name), fn)
        return name

    def run_suite():
        """Run every case. -> {passed:[name], failed:[{name,error}], skipped:[name], score}."""
        passed, failed, skipped = [], [], []
        for name, fn in cases:
            try:
                r = fn()
            except Exception as e:                       # a raising case is a failure, not a crash
                failed.append({"name": name, "error": str(e)[:160]})
                continue
            if r is None:
                skipped.append(name)
            elif r:
                passed.append(name)
            else:
                failed.append({"name": name, "error": "case returned False"})
        scored = len(passed) + len(failed)
        score = round(len(passed) / scored, 3) if scored else 1.0
        host.log("benchmark_run", passed=len(passed), failed=len(failed),
                 skipped=len(skipped), score=score)
        return {"passed": passed, "failed": failed, "skipped": skipped,
                "score": score, "total": len(cases)}

    def regressions(prev):
        """Cases that PASSED in a prior run_suite() result but no longer pass now.

        prev: a previous run_suite() dict (or None). Returns the list of newly-broken case names —
        the signal used to gate build-driver.autobuild (a non-empty list means the build regressed).
        """
        cur = run_suite()
        prev_passed = set((prev or {}).get("passed", []))
        now_ok = set(cur["passed"])
        regressed = sorted(prev_passed - now_ok)
        host.log("benchmark_regressions", count=len(regressed))
        return regressed

    host.log("module", step="tier2-19", id=MODULE_ID, cases=len(cases))
    return {"run_suite": run_suite, "add_case": add_case, "regressions": regressions}
