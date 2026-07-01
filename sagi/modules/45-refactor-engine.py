# sagi/modules/45-refactor-engine.py — tier-5 · "refine thyself"
#
# GAP CLOSED: meta-kernel-evolution can safely APPLY a change but nothing PROPOSES quality
# refactors. This spots refactor opportunities across the module corpus (duplicated helpers, oversized
# files, missing docstrings) and routes a proposed improvement through meta-kernel's gated apply — so
# the individual can tidy itself without ever risking a bad edit (scan → verify → benchmark → rollback).
#
# GROUNDED REUSE: introspection.map (the corpus), meta-kernel-evolution.propose_change/apply (the
# only safe write path for existing modules), host.store (read sources). STDLIB only.
from __future__ import annotations

import os

MODULE_ID = "refactor-engine"
DEPS = ["meta-kernel-evolution", "introspection"]
MOTTO = "refine thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _files():
        try:
            return sorted(f for f in os.listdir(host.store.modules_dir)
                          if f.endswith(".py") and f[0].isdigit())
        except OSError:
            return []

    def candidates():
        """Refactor opportunities across the corpus: large files, missing header docstring, dup helpers."""
        out = []
        dup_helper = 0
        for fn in _files():
            src = host.store.read(os.path.join("modules", fn)) or ""
            lines = src.count("\n")
            if "def _sib(host, mid):" in src:
                dup_helper += 1
            issues = []
            if lines > 160:
                issues.append(f"large ({lines} lines)")
            if not src.lstrip().startswith("#"):
                issues.append("no header comment")
            if issues:
                out.append({"file": fn, "issues": issues})
        summary = {"files": len(_files()), "duplicated__sib_helper": dup_helper, "candidates": out}
        host.log("refactor_candidates", n=len(out), dup_sib=dup_helper)
        return summary

    def propose_refactor(target_file, new_source):
        """Route a refactor through meta-kernel's gated apply (scan→verify→benchmark→rollback)."""
        mk = _sib(host, "meta-kernel-evolution")
        if not mk:
            return {"error": "meta-kernel-evolution not live"}
        pid = mk["propose_change"](target_file, new_source)["proposal_id"]
        return {"proposal_id": pid, "note": "call apply(proposal_id) via meta-kernel-evolution to gate it"}

    def apply(proposal_id):
        """Convenience passthrough to meta-kernel-evolution.apply (fully gated)."""
        mk = _sib(host, "meta-kernel-evolution")
        return mk["apply"](proposal_id) if mk else {"error": "meta-kernel-evolution not live"}

    host.log("module", step="tier5-45", id=MODULE_ID)
    return {"candidates": candidates, "propose_refactor": propose_refactor, "apply": apply}
