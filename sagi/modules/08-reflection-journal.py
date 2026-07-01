# sagi/modules/08-reflection-journal.py — expand-8 · "reflect on thyself"
#
# mindX keeps an improvement journal; sAGI has memory (gitmind) and evaluation (evaluator) but
# never writes down what a step meant. This module closes the loop: after a build step it writes
# a calibrated note — what was built, why, what is PROOF vs CONJECTURE (via epistemic-calibration),
# the current scorecard (via evaluator), and what should come next (via goal-graph) — persists it
# to journal/NN.md inside the package, and commits the moment to the gitmind memory tree so the
# reflection is itself recallable. Grounded on host.store.write + host.memory.commit + host.log.
from __future__ import annotations

import os
import time

MODULE_ID = "reflection-journal"
DEPS = ["memory-recall", "evaluator"]
MOTTO = "reflect on thyself"

_JOURNAL_DIR = "journal"


def _sib(host, mod_id):
    reg = getattr(host, "registry", None)
    return (reg["get"](mod_id) or {}).get("handle") if reg else None


def activate(host):
    def _next_index():
        try:
            existing = [f for f in os.listdir(os.path.join(host.root, _JOURNAL_DIR)) if f.endswith(".md")]
        except OSError:
            existing = []
        return len(existing) + 1

    def reflect(step=None):
        """Write a calibrated improvement note for a build step; persist + commit. Returns the note text."""
        step = step or {}
        if isinstance(step, str):
            step = {"built": step}
        built = step.get("built") or step.get("id") or "(unnamed step)"
        why = step.get("why", "advances the expand-sAGI goal")

        calib = _sib(host, "epistemic-calibration")
        evaluator = _sib(host, "evaluator")
        goalgraph = _sib(host, "goal-graph")

        assessed = calib["assess"](why + ". " + step.get("notes", "")) if calib else {"claims": []}
        proof = [c["claim"] for c in assessed["claims"] if c["kind"] == "proof"]
        conj = [c["claim"] for c in assessed["claims"] if c["kind"] == "conjecture"]
        metrics = evaluator["evaluate"]()["metrics"] if evaluator else {}
        nxt = goalgraph["next_module"]() if goalgraph else None

        proof_lines = [f"- {p}" for p in proof] or ["- (none stated)"]
        conj_lines = [f"- {c}" for c in conj] or ["- (none stated)"]
        idx = _next_index()
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        note = "\n".join([
            f"# Reflection {idx:02d} — {built}",
            f"_{when} UTC_",
            "",
            f"**Built:** {built}",
            f"**Why:** {why}",
            "",
            "**Proof (grounded):**",
            *proof_lines,
            "",
            "**Conjecture (uncertain):**",
            *conj_lines,
            "",
            "**Scorecard:** " + ", ".join(f"{k}={v}" for k, v in metrics.items()),
            "",
            f"**Next:** {nxt['title'] if nxt else '(no unmet node — converged)'}",
            "",
        ])
        rel = os.path.join(_JOURNAL_DIR, f"{idx:02d}-{_slug(built)}.md")
        host.store.write(rel, note)                         # Store-confined write (do no harm)
        if getattr(host, "memory", None) is not None:
            host.memory.commit(moment={"event": "reflection", "step": built, "file": rel},
                               message=f"reflect: {built}")
        host.log("reflection", step=built, file=rel, proof=len(proof), conjecture=len(conj))
        return note

    def journal():
        """All reflection notes, oldest first: [{file, text}]."""
        d = os.path.join(host.root, _JOURNAL_DIR)
        try:
            files = sorted(f for f in os.listdir(d) if f.endswith(".md"))
        except OSError:
            return []
        return [{"file": f, "text": host.store.read(os.path.join(_JOURNAL_DIR, f))} for f in files]

    host.log("module", step="expand-8", id=MODULE_ID)
    return {"reflect": reflect, "journal": journal}


def _slug(s):
    import re
    return (re.sub(r"[^a-z0-9]+", "-", (s or "note").lower()).strip("-")[:40]) or "note"
