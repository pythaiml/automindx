# sagi/modules/07-evaluator.py — expand-7 · "measure thyself"
#
# The 26 grown specs include a Self-Healing/Verification engine but nothing measures the system
# as it grows, so there is no signal for whether an expansion helped or regressed. This is a
# lightweight, dependency-free-of-external-services evaluation harness: it scores the live system
# (how many capabilities are live, how many verify clean, how large the callable API surface is,
# how deep the memory timeline runs, and a calibration self-consistency probe) and can diff two
# snapshots to prove a build step was progress. Every input is already on-disk/in-registry.
from __future__ import annotations

MODULE_ID = "evaluator"
DEPS = ["module-verifier", "memory-recall"]
MOTTO = "measure thyself"


def _sib(host, mod_id):
    reg = getattr(host, "registry", None)
    return (reg["get"](mod_id) or {}).get("handle") if reg else None


def activate(host):
    def evaluate():
        """Score the live system. Returns {metrics:{...}}."""
        reg = getattr(host, "registry", None)
        live = reg["list"]() if reg else []

        # callable API surface across every live handle
        api_surface = 0
        for mid in live:
            handle = (reg["get"](mid) or {}).get("handle") if reg else None
            if isinstance(handle, dict):
                api_surface += sum(1 for v in handle.values() if callable(v))

        verifier = _sib(host, "module-verifier")
        vres = verifier["verify_all"]() if verifier else {"passed": [], "failed": []}
        total = len(vres["passed"]) + len(vres["failed"])
        verified_fraction = round(len(vres["passed"]) / total, 3) if total else 1.0

        recall = _sib(host, "memory-recall")
        moments = len(recall["timeline"](limit=1000)) if recall else 0

        # calibration self-consistency: a labelled probe should round-trip correctly
        calib = _sib(host, "epistemic-calibration")
        calib_ok = None
        if calib:
            s = calib["split"]("PROOF: two plus two is four.\nCONJECTURE: it may rain tomorrow.")
            calib_ok = (len(s["proof"]) == 1 and len(s["conjecture"]) == 1)

        manifest = host.store.read_json("manifest.json", default={}) or {}
        metrics = {
            "live_modules": len(live),
            "verified_modules": len(vres["passed"]),
            "failed_modules": len(vres["failed"]),
            "verified_fraction": verified_fraction,
            "api_surface": api_surface,
            "memory_moments": moments,
            "manifest_version": manifest.get("version"),
            "calibration_consistent": calib_ok,
        }
        host.log("evaluate", **{k: v for k, v in metrics.items() if k != "manifest_version"})
        return {"metrics": metrics, "failed": vres["failed"]}

    def regress(prev):
        """Delta of current metrics vs a previous metrics dict (proves progress/regression)."""
        cur = evaluate()["metrics"]
        prev = (prev or {}).get("metrics", prev) or {}
        delta = {}
        for k, v in cur.items():
            if isinstance(v, (int, float)) and isinstance(prev.get(k), (int, float)):
                delta[k] = v - prev[k]
        return {"delta": delta, "current": cur}

    def report_md():
        """A human-readable Markdown scorecard."""
        m = evaluate()["metrics"]
        lines = ["# sAGI — expansion scorecard", ""]
        for k, v in m.items():
            lines.append(f"- **{k}**: {v}")
        return "\n".join(lines) + "\n"

    host.log("module", step="expand-7", id=MODULE_ID)
    return {"evaluate": evaluate, "regress": regress, "report_md": report_md}
