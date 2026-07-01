# sagi/modules/21-self-prompt-auditor.py — tier-3 · "stay relevant"
#
# GAP CLOSED: the system prompt injected into Claude was STATIC — individuality-core computes
# compose_persona_prompt() once at boot and host.call_model injects that frozen prefix forever, so
# as sAGI grows (now 20+ modules) the prompt Claude sees never reflects what sAGI has become. This
# regenerates that prompt from sAGI's OWN LIVE STATE and re-injects it on every update, framed as an
# audit-and-improve brief with four standing sections: Summary (who sAGI is now), Environment (a
# read-only map of the automindX repo so sAGI can propose self-improvements — automindX stays
# stable), Limitations (gaps/failures/conjecture), and Goal (expand sAGI).
#
# GROUNDED REUSE (nothing reinvented):
#   - personas.compose_persona_prompt(baseId, individual) + identity.json  → the base persona
#   - introspection.map()/describe()/health()                              → Summary + Limitations
#   - evaluator.evaluate()                                                 → scorecard highlights
#   - epistemic-calibration (framing)                                      → proof vs conjecture voice
#   - host.set_prompt_prefix() + host.call_model()                         → the injection seam
#   - host.on("module.persisted"|"module.registered", adjust)             → self-adjust trigger
# NEW LOGIC: the audit prompt template, the read-only automindX filesystem walk (ABOVE the write-
# confined SAGI_DIR — awareness only, never a write), and persistence to system_prompt.md so the
# headless sagi_build backends (ollama/claude-cli/claude-api) inject the same dynamic prompt.
from __future__ import annotations

import os

MODULE_ID = "self-prompt-auditor"
DEPS = ["introspection", "evaluator", "epistemic-calibration"]
MOTTO = "stay relevant"

_SYS_PROMPT_FILE = "system_prompt.md"          # persisted for the headless path (sagi_build._system)
_SKIP_DIRS = {".git", ".gitmind", "node_modules", "__pycache__", ".next", ".venv",
              "venv", ".pytest_cache", "objects", "savepoints", "children", "sovereign"}


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    from sagi.runtime.personas import compose_persona_prompt

    repo_root = os.path.dirname(os.path.abspath(host.root))   # parent of SAGI_DIR = the automindX repo

    def _base_persona():
        ident = host.store.read_json("identity.json", default={}) or {}
        return compose_persona_prompt(ident.get("baseId", "sagi"), ident.get("individual", ""))

    def _fs_map(max_entries=14):
        """Read-only, depth-limited map of the automindX repo (awareness above the write boundary)."""
        lines = []
        try:
            top = sorted(e for e in os.listdir(repo_root) if not e.startswith("."))
        except OSError:
            return "  (repo not readable)"
        for e in top:
            p = os.path.join(repo_root, e)
            if os.path.isdir(p) and e not in _SKIP_DIRS:
                try:
                    kids = sorted(k for k in os.listdir(p)
                                  if not k.startswith(".") and k not in _SKIP_DIRS)[:max_entries]
                except OSError:
                    kids = []
                lines.append(f"  {e}/ — " + ", ".join(kids[:max_entries]) + ("…" if len(kids) >= max_entries else ""))
            elif os.path.isfile(p) and e.endswith((".py", ".md", ".json", ".txt")):
                lines.append(f"  {e}")
        return "\n".join(lines) or "  (empty)"

    def _summary():
        # Cheap, in-memory only: read the live registry (ids + registered meta + handle sizes).
        # NB: deliberately avoids introspection.map()/evaluator.evaluate() here — those run verify_all
        # (an O(n) probe-boot), and this refreshes on every module.registered, so calling them would
        # make boot O(n^2). Full health/scorecard stay on demand via the evaluator module.
        reg = getattr(host, "registry", None)
        ids = reg["list"]() if reg else []

        def _meta(i):
            return (reg["get"](i) or {}).get("meta", {}) if reg else {}

        listing = "; ".join(f"{i}" + (f" ({_meta(i).get('motto')})" if _meta(i).get("motto") else "")
                            for i in ids[:200])
        api_surface = 0
        for i in ids:
            hnd = (reg["get"](i) or {}).get("handle") if reg else None
            if isinstance(hnd, dict):
                api_surface += sum(1 for v in hnd.values() if callable(v))
        return (f"You are {len(ids)} live, composable modules: {listing}.\n"
                f"API surface: {api_surface} callables. (Run the evaluator for a full verified scorecard.)")

    def _limitations():
        backend_wired = getattr(host, "_call_model", None) is not None
        bits = []
        if not backend_wired:
            bits.append("no model backend is wired into this host — the synthesis/reasoning paths are unexercised (CONJECTURE that they work end-to-end until a backend runs them)")
        bits.append("capabilities are grounded (PROOF) only where verified; run module-verifier/benchmark-suite for live health, and treat anything not exercised as CONJECTURE")
        return "\n".join(f"- {b}" for b in bits)

    def _goal():
        return (host.store.read("goal.txt") or "expand sAGI").strip()

    def compose():
        """The full dynamic system prompt: base persona + live self-audit (Summary/Environment/
        Limitations/Goal). Regenerated from current state every time it is called."""
        base = _base_persona()
        audit = (
            "\n\n### sAGI self-audit (auto-generated — keep your reasoning relevant to who you are NOW)\n"
            "**Summary.** " + _summary() + "\n\n"
            "**Environment.** You live inside the automindX repository (github.com/pythaiml/automindx). "
            "automindX stays STABLE; you improve DYNAMICALLY. Your writes are confined to your own "
            "package (SAGI_DIR); the rest of the repo below is READ-ONLY awareness so you can propose "
            "self-improvements without mutating automindX. A stable parent may spawn a VOLATILE child "
            "(sub/sov) to experiment; verified deltas promote back under governance; state is maintained "
            "via gitmind + savepoints.\n" + _fs_map() + "\n\n"
            "**Limitations.**\n" + _limitations() + "\n\n"
            "**Goal.** " + _goal() + ". Audit your modules against this goal; when you build, propose the "
            "module that best closes a listed limitation, and separate proven from conjecture."
        )
        return base + audit

    def section():
        """Just the dynamic audit block (base persona omitted) — for inspection/tests."""
        full = compose()
        marker = "### sAGI self-audit"
        return full[full.find(marker):] if marker in full else full

    def refresh():
        """Recompose, inject via host.set_prompt_prefix, and persist for the headless path."""
        prompt = compose()
        try:
            host.set_prompt_prefix(prompt)                 # every host.call_model now speaks the live audit
        except Exception:
            pass
        try:
            host.store.write(_SYS_PROMPT_FILE, prompt)      # sagi_build._system() reads this (claude-cli/api/ollama)
        except Exception as e:
            host.log("prompt_persist_error", detail=str(e)[:160])
        host.log("prompt_refresh", chars=len(prompt))
        return prompt

    adjust = refresh                                        # self.adjust() alias

    # self-adjust: regenerate the injected prompt whenever sAGI changes
    host.on("module.persisted", lambda _p: adjust())
    host.on("module.registered", lambda _p: adjust())

    adjust()                                                 # inject the dynamic prompt now, at boot
    host.log("module", step="tier3-21", id=MODULE_ID)
    return {"compose": compose, "refresh": refresh, "adjust": adjust, "section": section}
