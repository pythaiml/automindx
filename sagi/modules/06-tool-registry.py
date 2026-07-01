# sagi/modules/06-tool-registry.py — expand-6 · "reach beyond thyself"
#
# The audit flagged external tool/API integration as genuinely uncovered: modules can reason
# and grow, but cannot safely act. This is a registry of callable tools with schemas, where
# every invocation is boundary-guarded — writes route through self-package-boundary.guardWrite
# ("do no harm": a path that escapes the package is refused), and every call is logged and
# emitted on the host bus. Ships three grounded built-ins (write_file, read_file, recall) so the
# capability is real on boot, and lets any future module register more.
from __future__ import annotations

MODULE_ID = "tool-registry"
DEPS = ["module-loader"]
MOTTO = "reach beyond thyself"


def _sib(host, mod_id):
    reg = getattr(host, "registry", None)
    return (reg["get"](mod_id) or {}).get("handle") if reg else None


def activate(host):
    tools = {}   # name -> {"fn": callable(args)->result, "schema": {...}}

    def register_tool(name, fn, schema=None):
        """Register a tool: fn(args:dict)->result. schema documents its arguments."""
        if not callable(fn):
            raise ValueError("tool fn must be callable")
        tools[name] = {"fn": fn, "schema": schema or {}}
        host.emit("tool.registered", {"name": name})
        return name

    def list_tools():
        return [{"name": n, "schema": t["schema"]} for n, t in sorted(tools.items())]

    def invoke(name, args=None):
        """Invoke a registered tool by name with a dict of args. Boundary/error guarded."""
        args = args or {}
        t = tools.get(name)
        if not t:
            raise KeyError(f"no such tool: {name}")
        try:
            result = t["fn"](args)
            host.log("tool_invoke", name=name, ok=True)
            host.emit("tool.invoked", {"name": name, "ok": True})
            return {"ok": True, "result": result}
        except Exception as e:
            host.log("tool_invoke", name=name, ok=False, detail=str(e)[:160])
            host.emit("tool.invoked", {"name": name, "ok": False})
            return {"ok": False, "error": str(e)[:200]}

    # --- grounded built-in tools ---
    def _guarded_write(args):
        rel, content = args["path"], args.get("content", "")
        boundary = _sib(host, "self-package-boundary")
        if boundary and not boundary["guardWrite"](rel):     # do no harm
            raise PermissionError(f"write refused (escapes package): {rel}")
        host.store.write(rel, content)                        # Store._safe re-checks confinement
        return {"written": rel, "bytes": len(content)}

    def _read(args):
        return {"path": args["path"], "content": host.store.read(args["path"])}

    def _recall(args):
        mr = _sib(host, "memory-recall")
        return mr["recall"](args.get("query", ""), args.get("k", 5)) if mr else []

    register_tool("write_file", _guarded_write,
                  {"path": "str (relative, inside package)", "content": "str"})
    register_tool("read_file", _read, {"path": "str (relative)"})
    register_tool("recall", _recall, {"query": "str", "k": "int"})

    host.log("module", step="expand-6", id=MODULE_ID, tools=len(tools))
    return {"register_tool": register_tool, "list_tools": list_tools, "invoke": invoke}
