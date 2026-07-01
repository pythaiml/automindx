# sagi/runtime/savepoint.py — capture a running sAGI's output as a Point of Save.
#
# The complement to the Point of Departure: a snapshot you can take at any moment of
# a growing individual, saved four ways at once —
#   1. gitmind GLOBAL commit  — the point-of-save in the memory tree (a milestone);
#   2. .history event         — a `point_of_save` line, timestamp-aligned;
#   3. RAGE embedding         — internal semantic memory (pgvectorscale / SQLite);
#   4. shareable export       — a single Markdown of all modules under savepoints/.
#
#   python3 -m sagi.runtime.savepoint --dir "$SAGI_DIR" --label "first sAGI"
#   python3 -m sagi.runtime.savepoint --dir "$SAGI_DIR" --json     # machine-readable
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict

from .gitmind import GitMind


def _slug(s: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:48]) or "savepoint"


def _render_share(tree: Dict[str, str], label: str, commit: str, ts: int) -> str:
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))
    out = [
        f"# sAGI — Point of Save{f': {label}' if label else ''}",
        "",
        f"_gitmind commit `{commit[:12]}` · {len(tree)} module(s) · {when} UTC_",
        "",
        "> A savepoint of a growing individual sAGI — shareable, and also committed to"
        " its gitmind memory tree, .history, and RAGE.",
        "",
    ]
    for fn in sorted(tree):
        out.append(f"## {fn}")
        out.append("")
        out.append(tree[fn].rstrip())
        out.append("")
        out.append("---")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def save_point(sagi_dir: str, label: str = "", rage: bool = True) -> Dict[str, Any]:
    root = os.path.abspath(sagi_dir)
    gm = GitMind(root)
    ts = int(time.time())

    # 1. gitmind GLOBAL commit — the point of save (a milestone in the memory tree)
    commit = gm.global_commit(moment={"event": "point_of_save", "label": label},
                              message=label or "point of save", ts=ts)
    tree = gm.tree(commit)

    # 2. .history event, timestamp-aligned to the memory snapshot
    hist_dir = os.path.join(root, ".history")
    os.makedirs(hist_dir, exist_ok=True)
    with open(os.path.join(hist_dir, "build.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, "event": "point_of_save", "commit": commit,
                            "label": label, "modules": len(tree)}) + "\n")

    # 3. shareable export under savepoints/
    sp_dir = os.path.join(root, "savepoints")
    os.makedirs(sp_dir, exist_ok=True)
    export_path = os.path.join(sp_dir, f"{ts}-{_slug(label)}.md")
    text = _render_share(tree, label, commit, ts)
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(text)

    # 4. RAGE internal memory (best-effort; falls back to SQLite)
    rage_saved = 0
    if rage:
        try:
            from .rage_sync import RageSync
            rage_saved = RageSync(gm).save_commit(commit)
        except Exception:
            rage_saved = 0

    return {"commit": commit, "modules": len(tree), "export": export_path,
            "rage_saved": rage_saved, "ts": ts, "text": text}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Take a sAGI Point of Save (gitmind + .history + RAGE + export)")
    ap.add_argument("--dir", default=os.environ.get("SAGI_DIR", "./sagi"))
    ap.add_argument("--label", default="")
    ap.add_argument("--no-rage", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = save_point(a.dir, a.label, rage=not a.no_rage)
    if a.json:
        print(json.dumps({k: v for k, v in res.items() if k != "text"}))
    else:
        print(f"⭑ point of save · commit {res['commit'][:12]} · {res['modules']} module(s)")
        print(f"  shareable export → {res['export']}")
        print(f"  RAGE embedded    → {res['rage_saved']} form(s)")
