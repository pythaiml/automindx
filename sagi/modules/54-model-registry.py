# sagi/modules/54-model-registry.py — tier-6 · "know thy minds"
#
# GAP CLOSED: inference-router can dispatch to backends, but the individual keeps no registry of the
# MODELS themselves — their backend, capabilities, and version — which the SR-MRV spec called for.
# This is that registry: record a model (name → backend + capabilities), list them, and query one
# model's capabilities, so routing and negotiation can reason about which mind fits a task.
#
# GROUNDED REUSE: inference-router.backends (live backends), config-manager.get (default model/backend).
from __future__ import annotations

MODULE_ID = "model-registry"
DEPS = ["inference-router", "config-manager"]
MOTTO = "know thy minds"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _store():
        return host.store.read_json("models.json", default={}) or {}

    def register_model(name, backend, capabilities=None, version="1"):
        """Record a model: which backend runs it and what it's good at."""
        models = _store()
        models[name] = {"backend": backend, "capabilities": capabilities or [], "version": str(version)}
        host.store.write_json("models.json", models)
        host.log("model_register", name=name, backend=backend)
        return {"name": name, "backend": backend}

    def list_models():
        return _store()

    def capabilities(name):
        return _store().get(name, {}).get("capabilities", [])

    # seed from the live backends + the configured default, so the registry is non-empty on boot
    router = _sib(host, "inference-router")
    cfg = _sib(host, "config-manager")
    live = router["backends"]() if router else []
    default_model = cfg["get"]("model", "gpt-oss:120b-cloud") if cfg else "gpt-oss:120b-cloud"
    seed = _store()
    if not seed:
        if "ollama" in [b if isinstance(b, str) else b.get("backend") for b in live] or True:
            register_model(default_model, "ollama", ["reason", "build", "local"])
        register_model("claude-opus-4-8", "claude-cli", ["reason", "build", "code", "subscription"])

    host.log("module", step="tier6-54", id=MODULE_ID, models=len(_store()))
    return {"register_model": register_model, "list_models": list_models, "capabilities": capabilities}
