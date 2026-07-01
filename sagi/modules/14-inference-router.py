# sagi/modules/14-inference-router.py — expand-14 · "choose thy mind"
#
# Gap it closes: modules can reason (they get host.call_model), but that host is wired to exactly
# ONE backend at boot — a module cannot pick a better mind for the task, nor keep working when the
# wired backend is down. This is a backend-agnostic router: it chooses among ollama / claude-cli /
# claude-api per task kind, probes availability best-effort, and calls through a fallback chain so a
# single dead backend never blocks reasoning. It NEVER raises on a call — when nothing is reachable
# it returns a clear offline-marker string, so callers can degrade instead of crash.
#
# Grounded on (proven reuse, not reinvented):
#   • sagi_build.ask_model(model, prompt, backend)  — already dispatches all three backends
#     (ollama / claude-cli / claude-api). This module only *chooses* and *sequences* those calls.
#   • sagi_build.DEFAULT_MODEL / ANTHROPIC_MODEL / OLLAMA — the same model + endpoint the builder
#     uses, reused verbatim so a routed call matches how the individual was actually built.
#   • host.call_model  — the runtime-wired path (boot's default_call_model → ask_model); tried first
#     when present so we honour the persona prefix the host already composed.
#   • host.backend      — the individual's declared default backend (identity.json / SAGI_BACKEND);
#     used as the final routing fallback.
#   • tool-registry (DEP) — the sibling that made "acting" real; routing is the reasoning twin of it.
#
# New logic (conjecture, kept small): the task_kind → backend PREFERENCE table and the availability
# probes (a shutil.which for the CLI, an env-var check for the API key, a short TCP connect for the
# Ollama daemon). These are best-effort heuristics — the actual dispatch is the proven ask_model.
from __future__ import annotations

import os
import shutil
import socket

MODULE_ID = "inference-router"
DEPS = ["tool-registry"]
MOTTO = "choose thy mind"

# The three backends sagi_build.ask_model understands, in a neutral default order.
_ALL = ("ollama", "claude-cli", "claude-api")

# task_kind → preference order (conjecture heuristic). Heavier reasoning/coding prefers a Claude
# mind when reachable; fast/bulk/local work prefers the local Ollama daemon. Unknown kinds fall
# through to _DEFAULT_PREF (which is re-anchored on host.backend at runtime).
_PREFS = {
    "reason":    ("claude-cli", "claude-api", "ollama"),
    "plan":      ("claude-cli", "claude-api", "ollama"),
    "analyze":   ("claude-cli", "claude-api", "ollama"),
    "judge":     ("claude-cli", "claude-api", "ollama"),
    "code":      ("claude-cli", "claude-api", "ollama"),
    "build":     ("claude-cli", "claude-api", "ollama"),
    "synthesize":("claude-cli", "claude-api", "ollama"),
    "summarize": ("ollama", "claude-cli", "claude-api"),
    "fast":      ("ollama", "claude-cli", "claude-api"),
    "cheap":     ("ollama", "claude-cli", "claude-api"),
    "bulk":      ("ollama", "claude-cli", "claude-api"),
    "local":     ("ollama", "claude-cli", "claude-api"),
    "embed":     ("ollama", "claude-cli", "claude-api"),
}
_DEFAULT_PREF = ("ollama", "claude-cli", "claude-api")


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def _build():
    """Lazily import the repo-root builder. Returns the module or None (never raises here)."""
    try:
        import sagi_build            # provides ask_model + DEFAULT_MODEL / ANTHROPIC_MODEL / OLLAMA
        return sagi_build
    except Exception:
        return None


def _model_for(backend):
    """The model id sagi_build would use for a backend — reused verbatim from the builder."""
    sb = _build()
    if backend in ("claude-cli", "claude-api"):
        return getattr(sb, "ANTHROPIC_MODEL", "claude-opus-4-8") if sb else "claude-opus-4-8"
    return getattr(sb, "DEFAULT_MODEL", "gpt-oss:120b-cloud") if sb else "gpt-oss:120b-cloud"


def _probe(backend):
    """Best-effort availability probe (no exceptions escape). Runtime-only — never at import time."""
    try:
        if backend == "claude-cli":
            return shutil.which("claude") is not None
        if backend == "claude-api":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        if backend == "ollama":
            sb = _build()
            url = getattr(sb, "OLLAMA", None) or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            # crude parse of host:port from the endpoint, then a short TCP connect
            hostport = url.split("://", 1)[-1].split("/", 1)[0]
            host_, _, port_ = hostport.partition(":")
            with socket.create_connection((host_ or "127.0.0.1", int(port_ or 11434)), timeout=0.4):
                return True
    except Exception:
        return False
    return False


def activate(host):
    default_backend = getattr(host, "backend", None) or "ollama"

    def backends():
        """Best-effort list of currently reachable backends (subset of ollama/claude-cli/claude-api)."""
        avail = [b for b in _ALL if _probe(b)]
        host.log("router_backends", available=avail)
        return avail

    def route(task_kind="reason"):
        """Choose a backend for a task kind. Prefers an *available* backend by the heuristic table;
        falls back to the individual's declared host.backend when the probe finds none reachable.
        Never raises — returns a backend name string in all cases."""
        pref = _PREFS.get((task_kind or "").lower())
        if pref is None:
            # unknown kind → try the declared default first, then the neutral order
            pref = tuple([default_backend] + [b for b in _DEFAULT_PREF if b != default_backend])
        avail = set(backends())
        for b in pref:
            if b in avail:
                host.log("router_route", kind=task_kind, backend=b, basis="available")
                return b
        # nothing probed as reachable — return the declared default as a nominal route (conjecture)
        chosen = default_backend if default_backend in _ALL else pref[0]
        host.log("router_route", kind=task_kind, backend=chosen, basis="default")
        return chosen

    def _order(task_kind):
        """Ordered, de-duplicated candidate backends: routed choice first, then remaining available,
        then the declared default — so a dead primary still has fallbacks to try."""
        avail = backends()
        seq, seen = [], set()
        for b in [route(task_kind)] + avail + [default_backend]:
            if b in _ALL and b not in seen:
                seq.append(b)
                seen.add(b)
        return seq or [default_backend]

    def call(prompt, kind="reason"):
        """Route a prompt to a backend and return the text, trying fallbacks in order.

        Paths, in order: (1) the runtime-wired host.call_model if present (honours the composed
        persona prefix), then (2) sagi_build.ask_model for each candidate backend. Returns the first
        non-empty reply. NEVER raises — if no backend is reachable, returns a clear offline marker
        string so callers degrade gracefully.
        """
        prompt = "" if prompt is None else str(prompt)
        tried, last_err = [], None
        sb = _build()

        # (1) host.call_model path — the already-configured runtime backend (may be unwired/offline).
        if callable(getattr(host, "call_model", None)) and getattr(host, "_call_model", None):
            try:
                out = host.call_model(prompt)
                if out and str(out).strip():
                    host.log("router_call", kind=kind, path="host.call_model", ok=True)
                    host.emit("router.called", {"kind": kind, "path": "host.call_model", "ok": True})
                    return str(out).strip()
            except Exception as e:
                tried.append("host.call_model")
                last_err = str(e)[:160]

        # (2) explicit backend dispatch via the proven builder, in fallback order.
        for b in _order(kind):
            tried.append(b)
            if sb is None:
                last_err = "sagi_build unavailable (no ask_model)"
                break
            try:
                out = sb.ask_model(_model_for(b), prompt, b)
                if out and str(out).strip():
                    host.log("router_call", kind=kind, backend=b, ok=True)
                    host.emit("router.called", {"kind": kind, "backend": b, "ok": True})
                    return str(out).strip()
                last_err = "empty reply"
            except Exception as e:
                last_err = str(e)[:160]
                host.log("router_call", kind=kind, backend=b, ok=False, detail=last_err)

        marker = f"[inference-router: offline — no backend reachable (tried: {', '.join(tried) or 'none'})]"
        host.log("router_offline", kind=kind, tried=tried, detail=last_err)
        host.emit("router.called", {"kind": kind, "ok": False, "offline": True})
        return marker

    host.log("module", step="expand-14", id=MODULE_ID, default_backend=default_backend)
    return {"route": route, "call": call, "backends": backends}
