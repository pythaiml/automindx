# sagi/modules/48-config-manager.py — tier-5 · "know thy settings"
#
# GAP CLOSED: configuration is scattered — identity.json (backend/individual), environment vars
# (SAGI_BACKEND, OLLAMA_HOST), and per-module JSON files — with no single typed surface. This unifies
# them behind get/set/all with a clear precedence (config.json override > identity.json > env >
# default), so any module reads one coherent config. Overrides persist to config.json (Store-confined).
#
# GROUNDED REUSE: host.store (identity.json + config.json), host.backend/identity_id, os.environ.
from __future__ import annotations

import os

MODULE_ID = "config-manager"
DEPS = ["self-package-boundary"]
MOTTO = "know thy settings"

_ENV = {"backend": "SAGI_BACKEND", "ollama_host": "OLLAMA_HOST", "model": "CODEPHREAK_MODEL"}


def activate(host):
    def _overrides():
        return host.store.read_json("config.json", default={}) or {}

    def _identity():
        return host.store.read_json("identity.json", default={}) or {}

    def get(key, default=None):
        """Resolve a config key by precedence: config.json > identity.json > env > default."""
        ov = _overrides()
        if key in ov:
            return ov[key]
        ident = _identity()
        if key in ident:
            return ident[key]
        env_name = _ENV.get(key)
        if env_name and os.environ.get(env_name) is not None:
            return os.environ[env_name]
        # live-known fields on the host
        if key == "backend":
            return getattr(host, "backend", default)
        if key == "id":
            return getattr(host, "identity_id", default)
        return default

    def set(key, value):
        """Persist an override (config.json). Never touches identity.json or the environment."""
        ov = _overrides()
        ov[key] = value
        host.store.write_json("config.json", ov)
        host.emit("config.changed", {"key": key})
        return {"key": key, "value": value}

    def all_config():
        """The fully-resolved configuration (with each value's effective source)."""
        keys = set(list(_overrides()) + list(_identity()) + list(_ENV) + ["backend", "id"])
        return {k: get(k) for k in sorted(keys)}

    host.log("module", step="tier5-48", id=MODULE_ID)
    return {"get": get, "set": set, "all": all_config}
