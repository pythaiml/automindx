# sagi/modules/02-epistemic-calibration.py — expand-2 · "know thy uncertainty"
#
# Makes the Savante spine executable. personas.SAGI_EXPANSION (and the SAVANTE persona it
# grows from) mandates: derive from first principles, SEPARATE PROVEN FROM CONJECTURE, and
# quantify uncertainty. Today that is only a system-prompt instruction. This module turns it
# into code every other module can call: split any text into proof vs conjecture, score claim
# confidence, and run one calibrated host.callModel turn that returns its answer already
# partitioned. Grounded on host.callModel (host.py:96) — degrades to pure-heuristic tagging
# when no model backend is wired (e.g. an offline probe/boot).
from __future__ import annotations

import re

MODULE_ID = "epistemic-calibration"
DEPS = ["module-loader"]
MOTTO = "know thy uncertainty"

# Explicit markers a calibrated model turn is asked to emit, plus natural-language cues.
_PROOF_RE = re.compile(r"^\s*(?:PROOF|PROVEN|FACT|GROUNDED)\s*[:\-]", re.I)
_CONJ_RE = re.compile(r"^\s*(?:CONJECTURE|CONJ|GUESS|ASSUMPTION|SPECULATION|UNCERTAIN)\s*[:\-]", re.I)
_HEDGE = re.compile(r"\b(maybe|might|probably|likely|possibly|assume|assumes|assuming|"
                    r"conjecture|speculat\w+|i think|i believe|perhaps|could be|seems?|"
                    r"unverified|unclear|unknown|guess)\b", re.I)
_CERTAIN = re.compile(r"\b(proven|proof|therefore|because|by definition|is defined|"
                      r"the code shows|grounded in|verified|deterministic\w*|always|exactly)\b", re.I)

_CALIBRATE_SYS = (
    "Answer, then on their own lines label each claim you made either "
    "'PROOF: <claim>' (grounded / derivable / verifiable) or "
    "'CONJECTURE: <claim>' (assumed / uncertain). Separate proven from conjecture."
)


def _claims(text):
    """Split text into candidate claim strings (lines, then sentences)."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip(" \t-*•")
        if not line:
            continue
        # keep marker lines whole; otherwise break long lines into sentences
        if _PROOF_RE.match(line) or _CONJ_RE.match(line):
            out.append(line)
        else:
            out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return out


def _kind_and_conf(claim):
    """Classify one claim → ('proof'|'conjecture', confidence in [0,1])."""
    if _PROOF_RE.match(claim):
        return "proof", 0.9
    if _CONJ_RE.match(claim):
        return "conjecture", 0.3
    hedges = len(_HEDGE.findall(claim))
    certain = len(_CERTAIN.findall(claim))
    if hedges > certain:
        return "conjecture", max(0.15, 0.45 - 0.1 * hedges)
    if certain > 0:
        return "proof", min(0.95, 0.6 + 0.1 * certain)
    return "conjecture", 0.5                      # unmarked, unquantified → treat as conjecture, be humble


def activate(host):
    def assess(text):
        """{claims:[{claim,kind,confidence}]} — every claim tagged and scored."""
        claims = []
        for c in _claims(text):
            kind, conf = _kind_and_conf(c)
            claims.append({"claim": re.sub(r"^\s*\w+\s*[:\-]\s*", "", c).strip(), "kind": kind,
                           "confidence": round(conf, 2)})
        return {"claims": claims}

    def split(text):
        """{proof:[...], conjecture:[...]} — the Savante partition of a body of text."""
        a = assess(text)["claims"]
        return {"proof": [c["claim"] for c in a if c["kind"] == "proof"],
                "conjecture": [c["claim"] for c in a if c["kind"] == "conjecture"]}

    def calibrated_call(prompt):
        """One host.callModel turn whose answer is returned already partitioned. Offline-safe."""
        try:
            answer = host.call_model(prompt, system=_CALIBRATE_SYS)
        except Exception as e:
            answer = f"CONJECTURE: (no model backend: {str(e)[:80]})"
        part = split(answer)
        return {"answer": answer, "proof": part["proof"], "conjecture": part["conjecture"]}

    host.log("module", step="expand-2", id=MODULE_ID)
    return {"assess": assess, "split": split, "calibrated_call": calibrated_call}
