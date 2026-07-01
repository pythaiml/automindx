# sagi/runtime/spawn.py — every sAGI can spawn another sAGI.
#
# A parent individual spawns a child in one of two relationships:
#   • SUB  — a subordinate child, nested under the parent (<parent>/children/<id>);
#            governed by its lineage, part of the parent's tree.
#   • SOV  — a sovereign peer (<parent>/sovereign/<id>); autonomous — its own
#            identity, gitmind, and build loop; the parent spawned it but does not
#            govern it. (Sovereignty is a governance relationship; the folder is
#            nested only for containment and can be detached.)
#
# Both inherit the seed (be thyself · do no harm · grow thyself) and record lineage
# in both directions; the spawn is a gitmind GLOBAL milestone + a .history event.
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict

from .gitmind import GitMind


def _slug(s: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:40]) or f"sagi-{int(time.time())}"


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return default


def spawn(parent_dir: str, name: str, mode: str = "sub", individual: str = "",
          base_id: str = "sagi", boot_child: bool = True) -> Dict[str, Any]:
    if mode not in ("sub", "sov"):
        raise ValueError("mode must be 'sub' (subordinate) or 'sov' (sovereign)")
    parent = os.path.abspath(parent_dir)
    parent_ident = _read_json(os.path.join(parent, "identity.json"), {})
    parent_id = parent_ident.get("id") or os.path.basename(parent) or "sagi"
    child_id = _slug(name)
    sub_root = "children" if mode == "sub" else "sovereign"
    child_dir = os.path.join(parent, sub_root, child_id)
    os.makedirs(os.path.join(child_dir, "modules"), exist_ok=True)

    ts = int(time.time())
    identity = {
        "id": child_id, "name": name, "baseId": base_id, "individual": individual,
        "grown_from": parent_id, "mode": mode, "sovereign": mode == "sov", "created_ts": ts,
    }
    with open(os.path.join(child_dir, "identity.json"), "w", encoding="utf-8") as f:
        json.dump(identity, f, indent=2)

    # lineage in the parent (parent → children)
    lineage_path = os.path.join(parent, "lineage.json")
    lineage = _read_json(lineage_path, {"id": parent_id, "children": []})
    lineage.setdefault("children", [])
    lineage["children"] = [c for c in lineage["children"] if c.get("id") != child_id]
    lineage["children"].append({"id": child_id, "name": name, "mode": mode,
                                "dir": os.path.relpath(child_dir, parent), "ts": ts})
    with open(lineage_path, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2)

    # milestone in the parent's memory + .history
    parent_gm = GitMind(parent)
    commit = parent_gm.global_commit(moment={"event": "spawn", "child": child_id, "mode": mode},
                                     message=f"spawn {mode}: {name}", ts=ts)
    hist = os.path.join(parent, ".history")
    os.makedirs(hist, exist_ok=True)
    with open(os.path.join(hist, "build.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, "event": "spawn", "child": child_id, "mode": mode, "commit": commit}) + "\n")

    # bring the child to life (seed it) so it is immediately a real individual
    if boot_child:
        from .boot import boot
        boot(child_dir)

    return {"child_id": child_id, "dir": child_dir, "mode": mode, "sovereign": mode == "sov",
            "parent": parent_id, "commit": commit}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Spawn a child sAGI (sub | sov) from a parent")
    ap.add_argument("--parent", default=os.environ.get("SAGI_DIR", "./sagi"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["sub", "sov"], default="sub")
    ap.add_argument("--individual", default="")
    a = ap.parse_args()
    res = spawn(a.parent, a.name, a.mode, a.individual)
    print(f"⇲ spawned {res['mode']} sAGI '{res['child_id']}' ({'sovereign' if res['sovereign'] else 'subordinate'})")
    print(f"  dir    → {res['dir']}")
    print(f"  parent → {res['parent']} · gitmind {res['commit'][:12]}")
