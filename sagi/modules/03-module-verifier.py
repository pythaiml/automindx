# sagi/modules/03-module-verifier.py — expand-3 · "verify thyself"
#
# A module does not count as grown until it verifies. This checks the exact seed contract
# (MODULE_ID/DEPS/MOTTO/activate over the host surface from core/interface.md), then activates
# the candidate against a throwaway probe Host rooted in a temp SAGI_DIR — so a bad activate()
# can do no harm to the real package — and confirms the returned handle is a dict of callables
# and that path confinement (Store._safe) is intact. Wired to host.on("module.persisted") so
# every freshly-persisted module is auto-verified the moment the kernel registers it.
from __future__ import annotations

import os
import tempfile

MODULE_ID = "module-verifier"
DEPS = ["module-loader"]
MOTTO = "verify thyself"

_REQUIRED = ("MODULE_ID", "DEPS", "MOTTO")


def activate(host):
    from sagi.runtime.boot import _import_file
    from sagi.runtime.host import Host

    modules_dir = host.store.modules_dir

    def _probe_host():
        """A disposable host in a temp dir — an unsafe activate() cannot touch the real package."""
        d = tempfile.mkdtemp(prefix="sagi-verify-")
        return Host(d)

    def verify(name):
        """Verify one executable module by filename. Returns {ok, checks:[...], errors:[...]}."""
        checks, errors = [], []
        path = name if os.path.isabs(name) else os.path.join(modules_dir, name)

        def check(label, cond, err=None):
            checks.append({"check": label, "ok": bool(cond)})
            if not cond:
                errors.append(err or label)
            return bool(cond)

        if not check("file exists", os.path.exists(path), f"missing file: {name}"):
            return {"ok": False, "name": name, "checks": checks, "errors": errors}

        try:
            mod = _import_file(path)
        except Exception as e:
            errors.append(f"import failed: {str(e)[:160]}")
            return {"ok": False, "name": name, "checks": checks, "errors": errors}

        for attr in _REQUIRED:
            check(f"declares {attr}", hasattr(mod, attr), f"missing {attr}")
        check("MODULE_ID is str", isinstance(getattr(mod, "MODULE_ID", None), str))
        check("DEPS is list", isinstance(getattr(mod, "DEPS", None), list))
        check("activate is callable", callable(getattr(mod, "activate", None)), "activate not callable")

        if callable(getattr(mod, "activate", None)):
            probe = _probe_host()
            try:
                handle = mod.activate(probe)
                check("activate returns dict", isinstance(handle, dict), "activate must return a handle dict")
                if isinstance(handle, dict):
                    bad = [k for k, v in handle.items() if not callable(v)]
                    check("handle values are callable", not bad,
                          f"non-callable handle entries: {bad}")
                # confinement must hold on the probe host (do no harm)
                escaped = True
                try:
                    probe.store._safe("../escape.txt")
                    escaped = False
                except ValueError:
                    escaped = True
                check("path confinement intact", escaped, "Store._safe did not block escape")
            except Exception as e:
                check("activate runs without error", False, f"activate raised: {str(e)[:160]}")

        ok = not errors
        host.log("verify", name=os.path.basename(path), ok=ok, errors=len(errors))
        return {"ok": ok, "name": os.path.basename(path), "checks": checks, "errors": errors}

    def verify_all():
        """Verify every executable module the loader can see. {passed:[...], failed:[...]}."""
        loader = (host.registry["get"]("module-loader") or {}).get("handle") if getattr(host, "registry", None) else None
        names = loader["discover"]() if loader else \
            sorted(f for f in os.listdir(modules_dir) if f.endswith(".py") and f[0].isdigit())
        passed, failed = [], []
        for n in names:
            (passed if verify(n)["ok"] else failed).append(n)
        return {"passed": passed, "failed": failed}

    def gate(payload):
        """Auto-verify a freshly-persisted module (host.on('module.persisted', gate))."""
        f = (payload or {}).get("file")
        if f and f.endswith(".py"):
            r = verify(f)
            host.emit("module.verified", {"file": f, "ok": r["ok"], "errors": r["errors"]})

    host.on("module.persisted", gate)
    host.log("module", step="expand-3", id=MODULE_ID)
    return {"verify": verify, "verify_all": verify_all, "gate": gate}
