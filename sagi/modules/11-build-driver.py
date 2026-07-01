# sagi/modules/11-build-driver.py — expand-11 · "drive thyself"
#
# GAP CLOSED: curriculum.advance(builder=...) already gates a build→verify step, but nothing
# ever supplied the builder — so the self-build loop could never actually grow the next module.
# This is that missing builder. It reads goal-graph.next_module(), asks a model (host.call_model,
# else the headless sagi_build.ask_model backends) to synthesize a real activate(host) module,
# persists it to modules/NN-slug.py through the boundary-guarded tool-registry, and emits
# module.persisted so module-registry live-registers it and module-verifier.gate auto-verifies.
#
# GROUNDED / PROVEN REUSE (not reinvented):
#   - goal-graph.next_module()            → the node to build next
#   - tool-registry.invoke("write_file")  → boundary-guarded persist (self-package-boundary.guardWrite)
#   - module-verifier.verify(file)         → the ok/not-ok gate; auto-runs on module.persisted
#   - curriculum.advance(builder=...)      → the gated advance loop autobuild drives
#   - sagi_build.ask_model / host.call_model → the model backends (ollama / claude-cli / claude-api)
# NEW LOGIC: prompt synthesis, code extraction, NN-slug numbering, and an OFFLINE STUB fallback so
# the whole pipeline is testable with no model backend wired.
from __future__ import annotations

import os
import re

MODULE_ID = "build-driver"
DEPS = ["curriculum", "tool-registry", "goal-graph"]
MOTTO = "drive thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def _slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", (s or "module").lower()).strip("-")[:48]
    return s or "module"


def activate(host):
    # --- model access: host-wired first, then the headless builder backends; None if offline ---
    def _ask(prompt):
        if getattr(host, "_call_model", None) is not None:
            try:
                return host.call_model(prompt)
            except Exception:
                pass
        try:
            import sagi_build  # lazy: importing at module top would need `requests`
            return sagi_build.ask_model(sagi_build.DEFAULT_MODEL, prompt,
                                        getattr(host, "backend", None))
        except Exception:
            return None

    def _build_prompt(node):
        nid = node.get("id") or "grown-module"
        deps = node.get("deps") or []
        title = node.get("title") or nid
        return (
            f"Write ONE Python module for the sAGI runtime that implements the node "
            f"id='{nid}', title='{title}', deps={deps}.\n"
            "Contract (match exactly): module-level MODULE_ID (str), DEPS (list[str]), MOTTO (str), "
            "and activate(host) returning a dict whose values are ALL callables. Reach siblings via "
            "host.registry['get'](id)['handle']; degrade gracefully if a sibling is absent; never "
            "raise from activate(); stdlib only; all writes via host.store. "
            "Reply with ONLY the module source, optionally in a ```python fence."
        )

    def _extract_code(raw):
        if not raw:
            return None
        m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.S)
        body = (m.group(1) if m else raw).strip()
        return body or None

    def _stub_source(node):
        """A minimal, verifier-passing module — the offline fallback so the pipeline is testable."""
        nid = node.get("id") or "grown-module"
        deps = list(node.get("deps") or [])
        motto = (node.get("title") or nid)[:80]
        return (
            "# sagi/modules/auto — synthesized stub (build-driver offline fallback)\n"
            "# Grounded on goal-graph.next_module(); a minimal valid activate(host) so the\n"
            "# build->verify pipeline is testable with no model backend wired.\n"
            "from __future__ import annotations\n\n"
            f"MODULE_ID = {nid!r}\n"
            f"DEPS = {deps!r}\n"
            f"MOTTO = {motto!r}\n\n\n"
            "def _sib(host, mid):\n"
            "    reg = getattr(host, \"registry\", None)\n"
            "    return (reg[\"get\"](mid) or {}).get(\"handle\") if reg else None\n\n\n"
            "def activate(host):\n"
            "    def status():\n"
            "        return {\"id\": MODULE_ID, \"deps\": DEPS, \"stub\": True}\n\n"
            "    def describe():\n"
            "        return MOTTO\n\n"
            "    host.log(\"module\", id=MODULE_ID, stub=True)\n"
            "    return {\"status\": status, \"describe\": describe}\n"
        )

    def _next_number():
        try:
            files = os.listdir(host.store.modules_dir)
        except OSError:
            return 1
        mx = 0
        for f in files:
            m = re.match(r"^(\d+)-.*\.py$", f)
            if m:
                mx = max(mx, int(m.group(1)))
        return mx + 1

    def _persist(rel, content):
        """Boundary-guarded write via tool-registry, with a guarded direct-store fallback."""
        tr = _sib(host, "tool-registry")
        if tr:
            r = tr["invoke"]("write_file", {"path": rel, "content": content})
            if isinstance(r, dict) and r.get("ok"):
                return True
        boundary = _sib(host, "self-package-boundary")
        if boundary and not boundary["guardWrite"](rel):
            return False
        try:
            host.store.write(rel, content)
            return True
        except Exception:
            return False

    def synthesize(node):
        """node -> module source string (no write). Model-driven; offline -> minimal valid stub."""
        node = node or {}
        code = _extract_code(_ask(_build_prompt(node)))
        if code and "def activate" in code and "MODULE_ID" in code:
            return code
        return _stub_source(node)

    def build_next(node=None):
        """One gated build step: synthesize the next module, persist it, verify it.

        Returns {node, file, verified}. Usable as curriculum.advance's builder by taking the
        node it passes; standalone it reads goal-graph.next_module() itself.
        """
        gg = _sib(host, "goal-graph")
        if node is None:
            node = gg["next_module"]() if gg else None
        if not node:
            return {"node": None, "file": None, "verified": None}   # nothing left to build
        code = synthesize(node)
        fname = f"{_next_number():02d}-{_slug(node.get('id') or 'module')}.py"
        if not _persist(f"modules/{fname}", code):
            host.log("build_next", node=node.get("id"), file=fname, persisted=False)
            return {"node": node, "file": None, "verified": False}
        # announce: module-registry live-registers, module-verifier.gate auto-verifies.
        host.emit("module.persisted", {"id": node.get("id"), "file": fname})
        verifier = _sib(host, "module-verifier")
        verified = verifier["verify"](fname)["ok"] if verifier else None
        host.log("build_next", node=node.get("id"), file=fname, verified=verified)
        return {"node": node, "file": fname, "verified": verified}

    def autobuild(n=1):
        """Drive curriculum.advance(builder=build_next) up to n times, honoring the gate."""
        curriculum = _sib(host, "curriculum")
        results = []
        for _ in range(max(1, int(n))):
            if not curriculum:
                r = build_next()
                results.append(r)
                if not r.get("file"):
                    break
                continue
            # curriculum's gate calls verify(built_file); hand it the filename string.
            r = curriculum["advance"](builder=lambda node: build_next(node).get("file"))
            results.append(r)
            if r.get("converged"):
                break
            if (r.get("gate") or {}).get("verified") is False:
                break                                   # stop on a failed verify (heals in #12)
        return results

    host.log("module", step="expand-11", id=MODULE_ID)
    return {"build_next": build_next, "autobuild": autobuild, "synthesize": synthesize}
