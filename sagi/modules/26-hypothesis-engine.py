# sagi/modules/26-hypothesis-engine.py — tier-3 · "question thyself"
#
# GAP CLOSED: the individual reasons and recalls but never forms and TESTS a hypothesis, updating a
# calibrated belief — the scientific loop the Savante spine implies. This closes it: hypothesize()
# proposes an answer (via inference-router, offline a grounded heuristic) tagged proof/conjecture by
# epistemic-calibration; test() checks it against ingested knowledge (knowledge-grounding.cite) and
# updates a calibrated belief; beliefs() lists them. Beliefs persist to beliefs.json — the in-module
# analogue of aglm.beliefs.BeliefSystem (referenced, kept simple/serializable here).
from __future__ import annotations

import time

MODULE_ID = "hypothesis-engine"
DEPS = ["inference-router", "epistemic-calibration", "knowledge-grounding"]
MOTTO = "question thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _beliefs():
        return host.store.read_json("beliefs.json", default=[]) or []

    def _save(bs):
        try:
            host.store.write_json("beliefs.json", bs)
        except Exception:
            pass

    def hypothesize(question):
        """Propose a hypothesis for a question, tagged proof vs conjecture with a confidence."""
        router = _sib(host, "inference-router")
        calib = _sib(host, "epistemic-calibration")
        answer = None
        if router:
            try:
                answer = router["call"](f"Hypothesis for: {question}. State it in one sentence.", kind="reason")
            except Exception:
                answer = None
        if not answer or "(offline" in answer or "no model backend" in answer.lower():
            answer = f"CONJECTURE: {question} — untested hypothesis (no model backend to derive it)."
        parts = calib["split"](answer) if calib else {"proof": [], "conjecture": [answer]}
        confidence = 0.7 if parts["proof"] else 0.3
        return {"question": question, "hypothesis": answer.strip(),
                "proof": parts["proof"], "conjecture": parts["conjecture"], "confidence": confidence}

    def test(hypothesis):
        """Cheap test: is the hypothesis supported by ingested knowledge? Update a calibrated belief."""
        text = hypothesis if isinstance(hypothesis, str) else hypothesis.get("hypothesis", "")
        kg = _sib(host, "knowledge-grounding")
        support = kg["cite"](text, 3) if kg else []
        supported = len(support) > 0
        prior = hypothesis.get("confidence", 0.5) if isinstance(hypothesis, dict) else 0.5
        conf = round(min(0.95, prior + 0.2) if supported else max(0.05, prior - 0.2), 3)
        belief = {"claim": text, "supported": supported, "confidence": conf,
                  "evidence": [s.get("source") or s.get("file") for s in support][:3], "ts": int(time.time())}
        bs = _beliefs()
        bs = [b for b in bs if b.get("claim") != text] + [belief]
        _save(bs)
        host.log("hypothesis_test", supported=supported, confidence=conf)
        return {"result": "supported" if supported else "unsupported", "belief": belief}

    def beliefs():
        return _beliefs()

    host.log("module", step="tier3-26", id=MODULE_ID)
    return {"hypothesize": hypothesize, "test": test, "beliefs": beliefs}
