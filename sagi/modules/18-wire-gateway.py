# sagi/modules/18-wire-gateway.py — tier-2 · "open thyself"
#
# Gap it closes: the individual can see itself (16-introspection), score itself (07-evaluator),
# and align itself (13-policy-guard), but there is no way for ANOTHER orchestrator to drive it.
# core/interface.md's thesis is that a module only ever touches the agnostic host surface, so the
# same handles that power the console and the CLI can power a third host: a local HTTP/JSON wire.
# This exposes that surface — read-only introspection/evaluator by default, policy-guarded writes —
# as a PURE handle(method, path, body) that needs no socket to test, plus an opt-in stdlib server.
#
# Grounded reuse (nothing reinvented):
#   - introspection.map / introspection.health  -> the live self-map + verify health (dep)
#   - evaluator.report_md / evaluator.evaluate   -> the scorecard (reached guardedly via _sib)
#   - policy-guard.check(action, context)        -> the allow-by-default gate every POST passes (dep)
#   - http.server (stdlib only)                  -> the opt-in transport; NEVER auto-started
# New logic (small): a route table + a pure dispatcher that maps (method, path) -> {status, json},
# JSON body parsing, and a thin BaseHTTPRequestHandler that just forwards to the pure handle().
from __future__ import annotations

import json

MODULE_ID = "wire-gateway"
DEPS = ["introspection", "policy-guard"]
MOTTO = "open thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def _coerce_body(body):
    """Accept a dict (in-process caller) or a JSON string/bytes (HTTP). Returns (obj, error)."""
    if body is None or body == "":
        return {}, None
    if isinstance(body, (dict, list)):
        return body, None
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8")
        except Exception:
            return None, "body is not valid utf-8"
    if isinstance(body, str):
        try:
            return json.loads(body), None
        except Exception:
            return None, "body is not valid JSON"
    return None, "unsupported body type"


def _norm(path):
    """Normalise a request path: drop query string, strip a trailing slash (keep bare '/')."""
    path = (path or "/").split("?", 1)[0]
    if len(path) > 1:
        path = path.rstrip("/")
    return path or "/"


def activate(host):
    # --- the route table (declarative; read == open, gated == must pass policy-guard.check) ---
    ROUTES = [
        {"method": "GET", "path": "/", "access": "read", "desc": "route index"},
        {"method": "GET", "path": "/map", "access": "read", "desc": "live self-map (introspection)"},
        {"method": "GET", "path": "/health", "access": "read", "desc": "verify health (introspection)"},
        {"method": "GET", "path": "/report", "access": "read", "desc": "expansion scorecard (evaluator)"},
        {"method": "POST", "path": "/check", "access": "gated", "desc": "evaluate a policy-guard gate"},
        {"method": "POST", "path": "/action", "access": "gated", "desc": "policy-gated action (deny=403)"},
    ]

    def routes():
        """The declared HTTP surface: [{method, path, access, desc}] (a copy)."""
        return [dict(r) for r in ROUTES]

    # --- read-only handlers (grounded in live introspection/evaluator; degrade if absent) ---
    def _map():
        intro = _sib(host, "introspection")
        if not intro:
            return 503, {"error": "introspection unavailable"}
        try:
            return 200, intro["map"]()
        except Exception as e:
            return 500, {"error": str(e)[:200]}

    def _health():
        intro = _sib(host, "introspection")
        if not intro:
            return 503, {"error": "introspection unavailable"}
        try:
            return 200, intro["health"]()
        except Exception as e:
            return 500, {"error": str(e)[:200]}

    def _report():
        # evaluator is not a declared dep but is live in the registry; reach it guardedly.
        ev = _sib(host, "evaluator")
        if not ev:
            return 503, {"error": "evaluator unavailable"}
        try:
            metrics = ev["evaluate"]().get("metrics", {})
            return 200, {"report_md": ev["report_md"](), "metrics": metrics}
        except Exception as e:
            return 500, {"error": str(e)[:200]}

    # --- policy-gated POST handlers (every write/action passes policy-guard.check first) ---
    def _gate(body):
        """Run the action through policy-guard.check. Returns (decision_dict, error_or_None)."""
        guard = _sib(host, "policy-guard")
        if not guard:
            return None, "policy-guard unavailable"
        action = body.get("action")
        if not action:
            return None, "missing 'action'"
        context = body.get("context") or {}
        if not isinstance(context, dict):
            return None, "'context' must be an object"
        try:
            return guard["check"](action, context), None
        except Exception as e:
            return None, "gate error: " + str(e)[:160]

    def _check(body):
        """POST /check — evaluate the gate and report the decision (200 either way: it's a probe)."""
        decision, err = _gate(body)
        if err:
            return (503 if "unavailable" in err else 400), {"error": err}
        return 200, {"decision": decision}

    def _action(body):
        """POST /action — a policy-gated action: a deny is a hard 403, an allow proceeds (200)."""
        decision, err = _gate(body)
        if err:
            return (503 if "unavailable" in err else 400), {"error": err}
        if not decision.get("allow"):
            return 403, {"allowed": False, "decision": decision}
        # Allowed: this surface performs no side-effect itself — it certifies the action passed
        # the gate so an orchestrator may proceed. Real effects stay behind their own modules.
        return 200, {"allowed": True, "decision": decision, "action": body.get("action")}

    def _index():
        return 200, {"module": MODULE_ID, "motto": MOTTO, "routes": routes()}

    _GET = {"/": _index, "/map": _map, "/health": _health, "/report": _report}
    _POST = {"/check": _check, "/action": _action}

    def handle(method, path, body=None):
        """PURE request dispatcher: (method, path, body) -> {status, json}. No socket required.

        GET routes are read-only (introspection/evaluator). POST routes must carry an 'action'
        and pass policy-guard.check first. Never raises — a bad body is a 400, an unknown path a
        404, a wrong verb a 405, a missing sibling a 503.
        """
        method = (method or "GET").upper()
        path = _norm(path)
        host.log("wire_request", method=method, path=path)

        if method == "GET":
            fn = _GET.get(path)
            if not fn:
                return _resp(405 if path in _POST else 404, {"error": f"no route: GET {path}"})
            status, payload = fn()
            return _resp(status, payload)

        if method == "POST":
            fn = _POST.get(path)
            if not fn:
                return _resp(405 if path in _GET else 404, {"error": f"no route: POST {path}"})
            obj, err = _coerce_body(body)
            if err:
                return _resp(400, {"error": err})
            if not isinstance(obj, dict):
                return _resp(400, {"error": "body must be a JSON object"})
            status, payload = fn(obj)
            return _resp(status, payload)

        return _resp(405, {"error": f"method not allowed: {method}"})

    def _resp(status, payload):
        host.emit("wire.responded", {"status": status})
        return {"status": status, "json": payload}

    # --- opt-in stdlib transport (never auto-started; only a serve() call opens a socket) ---
    def serve(port=8787, host_addr="127.0.0.1"):
        """Opt-in local HTTP server bridging real requests to the pure handle(). Blocks until
        interrupted. stdlib only; no socket is opened until this is explicitly called."""
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class _Handler(BaseHTTPRequestHandler):
            def _serve(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                out = handle(method, self.path, raw)
                blob = json.dumps(out.get("json"), default=str).encode("utf-8")
                self.send_response(int(out.get("status", 200)))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)

            def do_GET(self):
                self._serve("GET")

            def do_POST(self):
                self._serve("POST")

            def log_message(self, *a):     # route access logs through the host, not stderr
                try:
                    host.log("wire_http", detail=(a[0] % a[1:]) if len(a) > 1 else str(a))
                except Exception:
                    pass

        server = HTTPServer((host_addr, int(port)), _Handler)
        host.log("wire_serve", addr=host_addr, port=int(port))
        host.emit("wire.serving", {"addr": host_addr, "port": int(port)})
        try:
            server.serve_forever()
        finally:
            server.server_close()
        return {"served": True, "addr": host_addr, "port": int(port)}

    host.log("module", step="tier2-18", id=MODULE_ID, routes=len(ROUTES))
    return {"routes": routes, "handle": handle, "serve": serve}
