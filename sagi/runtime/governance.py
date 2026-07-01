# sagi/runtime/governance.py — who may edit whom, and where a sAGI may build.
#
# Sandbox: every sAGI builds inside its own SAGI_DIR; the self-package-boundary seed
# ("do no harm") already refuses any write that escapes that package, so a build is
# sandboxed by construction (verified: SAGI_DIR=<box> python3 sagi_build.py ...).
#
# Edit governance (layered on top):
#   • a sAGI may edit ITSELF and its SUBORDINATE (sub) descendants;
#   • a SOVEREIGN (sov) sAGI is self-governing — only it may edit itself and its own
#     subtree; a parent may NOT reach across the sovereignty boundary.
# In the on-disk layout, sub descendants live under `children/…`, sovereign ones under
# `sovereign/<id>` — so authority ends the moment a path crosses a `sovereign/` segment.
from __future__ import annotations

import os


def _rp(p: str) -> str:
    return os.path.realpath(p)


def can_edit(actor_dir: str, target_dir: str) -> bool:
    """True iff the sAGI at actor_dir may edit the sAGI at target_dir."""
    a, t = _rp(actor_dir), _rp(target_dir)
    if a == t:
        return True                                   # always edit self
    if not t.startswith(a + os.sep):
        return False                                  # governance flows downward only
    parts = os.path.relpath(t, a).split(os.sep)
    return "sovereign" not in parts                   # a sovereign subtree is off-limits to the parent


def assert_can_edit(actor_dir: str, target_dir: str) -> None:
    if not can_edit(actor_dir, target_dir):
        raise PermissionError(
            f"{os.path.basename(_rp(actor_dir))} may not edit the sovereign/foreign sAGI at {target_dir}")


def governed_write(actor_dir: str, target_dir: str, rel_path: str, content: str) -> str:
    """Write into another sAGI's package only if governance allows it and the path
    stays inside that package (do no harm). Returns the written path."""
    assert_can_edit(actor_dir, target_dir)
    target = _rp(target_dir)
    dest = os.path.realpath(os.path.join(target, rel_path))
    if dest != target and not dest.startswith(target + os.sep):
        raise ValueError(f"path escapes the target package: {rel_path}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(content)
    return dest
