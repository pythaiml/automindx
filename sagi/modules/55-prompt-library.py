# sagi/modules/55-prompt-library.py — tier-6 · "curate thy prompts"
#
# GAP CLOSED: the dynamic system prompt is generated live (self-prompt-auditor), but reusable TASK
# prompts (propose-module, summarize, critique) are hardcoded across modules. This is a versioned
# library of named prompt templates with {slot} interpolation, so prompts become first-class,
# reusable, and improvable artifacts rather than string literals. Persisted to prompts.json.
#
# GROUNDED REUSE: self-prompt-auditor.section (the live audit block can be embedded in a template),
# versioning.bump (track template revisions). STDLIB only (string.Template-style, no eval).
from __future__ import annotations

import re

MODULE_ID = "prompt-library"
DEPS = ["self-prompt-auditor", "versioning"]
MOTTO = "curate thy prompts"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _store():
        return host.store.read_json("prompts.json", default={}) or {}

    def add(name, template):
        """Register/replace a named prompt template with {slot} placeholders."""
        store = _store()
        rev = (store.get(name, {}).get("rev", 0)) + 1
        store[name] = {"template": template, "rev": rev}
        host.store.write_json("prompts.json", store)
        return {"name": name, "rev": rev}

    def render(name, **slots):
        """Render a template, filling {slot} placeholders; {audit} injects the live self-audit block."""
        tpl = _store().get(name, {}).get("template")
        if tpl is None:
            return {"error": f"no prompt {name}"}
        if "{audit}" in tpl:
            auditor = _sib(host, "self-prompt-auditor")
            slots.setdefault("audit", auditor["section"]() if auditor else "")
        def sub(m):
            return str(slots.get(m.group(1), m.group(0)))
        return re.sub(r"\{(\w+)\}", sub, tpl)

    def list_prompts():
        return {n: r["rev"] for n, r in _store().items()}

    # seed grounded, reusable templates the build/critique loops can share
    if not _store():
        add("propose_module", "{audit}\n\nPropose the next module that closes a limitation. Reply with a Title and a spec.")
        add("critique", "Critique the following for correctness and separate proof from conjecture:\n\n{text}")

    host.log("module", step="tier6-55", id=MODULE_ID)
    return {"add": add, "render": render, "list_prompts": list_prompts}
