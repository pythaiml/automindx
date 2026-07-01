# Seed module 3 — module-registry (the kernel)  ·  "grow thyself"
# Turns "generates files" into "grows a live system": registers module handles,
# activates them in topological deps order (refusing cycles), and live-registers a
# freshly-persisted module via host.on("module.persisted", ...) — no restart.

MODULE_ID = "module-registry"
DEPS = ["self-package-boundary"]
MOTTO = "grow thyself"


def _toposort(specs):
    """specs: {id: {deps: [...]}} -> ordered ids; raises on cycle."""
    order, temp, done = [], set(), set()

    def visit(nid):
        if nid in done:
            return
        if nid in temp:
            raise ValueError(f"dependency cycle at {nid}")
        temp.add(nid)
        for d in specs.get(nid, {}).get("deps", []):
            if d in specs:
                visit(d)
        temp.discard(nid)
        done.add(nid)
        order.append(nid)

    for nid in specs:
        visit(nid)
    return order


def activate(host):
    reg = {}  # id -> {handle, meta}

    def register(mod_id, handle=None, meta=None):
        reg[mod_id] = {"handle": handle, "meta": meta or {}}
        host.emit("module.registered", {"id": mod_id})
        return reg[mod_id]

    def get(mod_id):
        return reg.get(mod_id)

    def listing():
        return list(reg.keys())

    def activate_all(loaders):
        """loaders: {id: {activate: callable, deps: [...]}} -> activate in deps order."""
        specs = {mid: {"deps": spec.get("deps", [])} for mid, spec in loaders.items()}
        for mid in _toposort(specs):
            spec = loaders[mid]
            handle = spec["activate"](host) if callable(spec.get("activate")) else None
            register(mid, handle, {"deps": spec.get("deps", [])})
        return listing()

    # live registration: a freshly-persisted module is registered without a restart.
    host.on("module.persisted", lambda payload: register(
        (payload or {}).get("id", "unknown"), None, {"persisted": True, **(payload or {})}))

    host.log("module", step=3, id=MODULE_ID)
    return {"register": register, "get": get, "list": listing, "activate_all": activate_all}
