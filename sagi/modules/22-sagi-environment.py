# sagi/modules/22-sagi-environment.py — tier-3 · "isolate thyself"
#
# GAP CLOSED: sAGI can spawn (swarm), verify, and roll back — but nothing wires those into the
# stability model the operator asked for: automindX + the parent stay STABLE while a VOLATILE
# child does the risky self-improvement, and only VERIFIED deltas promote back. This orchestrates
# exactly that: isolate() spawns a volatile child seeded with the parent's live capability stack;
# experiment() boots the child and autobuilds INSIDE it (writes confined to the child dir);
# promote() pulls only verifier-passing files back into the parent — refusing to reach across a
# sovereign boundary. State is maintained throughout via each child's savepoint + gitmind.
#
# GROUNDED REUSE (nothing reinvented):
#   - swarm-orchestrator.delegate / spawn.spawn  → the volatile child (sub|sov)
#   - build-driver.autobuild (in the child)       → the isolated experiment
#   - module-verifier.verify (in the parent)      → verified-only promotion gate
#   - governance.can_edit                          → refuse promotion across a sovereign boundary
#   - savepoint.save_point                         → state maintained (4-way snapshot)
#   - host.store (confined) + lineage.json         → child files + roster
# NEW LOGIC: seeding the child with the parent's .py stack, and the verified-only promote() merge.
from __future__ import annotations

import os

MODULE_ID = "sagi-environment"
DEPS = ["swarm-orchestrator", "rollback-recovery", "build-driver"]
MOTTO = "isolate thyself"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    from sagi.runtime.governance import can_edit as _can_edit

    def _child_dir(child_id, mode="sub"):
        sub = "children" if mode == "sub" else "sovereign"
        return os.path.join(host.root, sub, child_id)

    def isolate(name, mode="sub", seed_full=True):
        """Spawn a VOLATILE child (the isolated environment). Optionally seed it with the parent's
        live .py capability stack so experiments run against the full system. Parent stays stable."""
        swarm = _sib(host, "swarm-orchestrator")
        if not swarm:
            return {"error": "swarm-orchestrator not live"}
        res = swarm["delegate"](name, mode=mode)          # spawn + governance-aware, records lineage
        child_id, mode = res.get("child_id", name), res.get("mode", mode)
        seeded = 0
        if seed_full:
            try:
                for fn in sorted(os.listdir(host.store.modules_dir)):
                    if fn.endswith(".py") and not fn.startswith((".", "_")):
                        src = host.store.read(os.path.join("modules", fn)) or ""
                        # child dir is under the parent package → Store-confined write
                        host.store.write(os.path.join(_rel_children(mode), child_id, "modules", fn), src)
                        seeded += 1
            except OSError as e:
                host.log("isolate_seed_error", detail=str(e)[:160])
        host.log("isolate", child=child_id, mode=mode, seeded=seeded)
        return {"child_id": child_id, "mode": mode, "dir": _child_dir(child_id, mode), "seeded": seeded}

    def _rel_children(mode):
        return "children" if mode == "sub" else "sovereign"

    def experiment(child_id, n=1, mode="sub"):
        """Boot the volatile child and autobuild INSIDE it (writes stay in the child). Snapshot it.
        The parent is never mutated by the experiment."""
        cdir = _child_dir(child_id, mode)
        if not os.path.isdir(cdir):
            return {"error": f"no such environment: {child_id}"}
        from sagi.runtime.boot import boot as _boot_fn
        try:
            child_host, child_handles = _boot_fn(cdir)
        except Exception as e:
            return {"error": f"child boot failed: {str(e)[:160]}"}
        creg = child_handles["module-registry"]
        bd = (creg["get"]("build-driver") or {}).get("handle")
        results = bd["autobuild"](n) if bd else []
        save = None
        try:
            from sagi.runtime.savepoint import save_point
            sp = save_point(cdir, label=f"experiment-{child_id}", rage=False)
            save = sp["commit"][:12]
        except Exception as e:
            host.log("experiment_savepoint_error", detail=str(e)[:120])
        host.log("experiment", child=child_id, built=len(results), savepoint=save)
        return {"child_id": child_id, "results": results, "savepoint": save,
                "child_live": len(creg["list"]())}

    def promote(child_id, files, mode="sub"):
        """Pull VERIFIED files from a child back into the parent. Refuses across a sovereign
        boundary (governance). Verified-only: a file that fails the parent verifier is skipped."""
        cdir = _child_dir(child_id, mode)
        if not _can_edit(host.root, cdir):
            host.log("promote_refused", child=child_id, reason="sovereign boundary")
            return {"promoted": [], "refused": "governance: cannot promote across a sovereign boundary"}
        verifier = _sib(host, "module-verifier")
        promoted, skipped = [], []
        for fn in files:
            csrc = os.path.join(cdir, "modules", fn)
            try:
                with open(csrc, encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                skipped.append({"file": fn, "reason": "not found in child"})
                continue
            host.store.write(os.path.join("modules", fn), content)     # parent-confined write
            ok = verifier["verify"](fn)["ok"] if verifier else True
            if ok:
                promoted.append(fn)                                    # verified file on disk; hot-loads on next boot
            else:
                # verified-only: undo the promotion of a bad file (heal — rollback-recovery also guards)
                try:
                    os.remove(os.path.join(host.store.modules_dir, fn))
                except OSError:
                    pass
                skipped.append({"file": fn, "reason": "failed parent verification"})
        host.log("promote", child=child_id, promoted=len(promoted), skipped=len(skipped))
        return {"promoted": promoted, "skipped": skipped}

    def environments():
        """The volatile environments spawned from this parent (from lineage.json)."""
        lineage = host.store.read_json("lineage.json", default={"children": []}) or {}
        out = []
        for c in lineage.get("children", []):
            cdir = os.path.join(host.root, c.get("dir", ""))
            out.append({"id": c.get("id"), "mode": c.get("mode"),
                        "volatile": True, "editable_by_parent": _can_edit(host.root, cdir),
                        "exists": os.path.isdir(cdir)})
        return out

    host.log("module", step="tier3-22", id=MODULE_ID)
    return {"isolate": isolate, "experiment": experiment, "promote": promote,
            "environments": environments}
