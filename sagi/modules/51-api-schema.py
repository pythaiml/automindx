# sagi/modules/51-api-schema.py — tier-6 · "publish thy contract"
#
# GAP CLOSED: wire-gateway serves routes and every module exposes a handle, but there is no published
# machine-readable contract of what's callable — so an external client can't discover the surface.
# This generates that contract: a JSON schema of every module's API and a minimal OpenAPI-style
# document for the wire-gateway routes, so the individual can be integrated against, not guessed at.
#
# GROUNDED REUSE: introspection.map (module → api names), wire-gateway.routes (HTTP surface). STDLIB.
from __future__ import annotations

MODULE_ID = "api-schema"
DEPS = ["wire-gateway", "introspection"]
MOTTO = "publish thy contract"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def schema():
        """JSON schema of the whole individual: {module: {motto, deps, methods:[...]}}."""
        intro = _sib(host, "introspection")
        mods = intro["map"]()["modules"] if intro else []
        return {m["id"]: {"motto": m.get("motto"), "deps": m.get("deps", []),
                          "methods": m.get("api", [])} for m in mods}

    def openapi():
        """A minimal OpenAPI-style document for the wire-gateway routes."""
        gw = _sib(host, "wire-gateway")
        routes = gw["routes"]() if gw else []
        paths = {}
        for r in routes:
            path = r if isinstance(r, str) else (r.get("path") or str(r))
            method = "get" if isinstance(r, str) else (r.get("method", "get")).lower()
            paths.setdefault(path, {})[method] = {"summary": f"{method.upper()} {path}",
                                                  "responses": {"200": {"description": "ok"}}}
        return {"openapi": "3.0.0",
                "info": {"title": f"sAGI:{getattr(host, 'identity_id', 'sagi')}", "version": "1.0.0"},
                "paths": paths}

    def describe_route(path):
        gw = _sib(host, "wire-gateway")
        routes = gw["routes"]() if gw else []
        for r in routes:
            p = r if isinstance(r, str) else r.get("path")
            if p == path:
                return {"path": path, "route": r}
        return {"error": f"no route {path}"}

    host.log("module", step="tier6-51", id=MODULE_ID)
    return {"schema": schema, "openapi": openapi, "describe_route": describe_route}
