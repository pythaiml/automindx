# services/self_audit.py
# Gives Professor Codephreak *real* filesystem access to audit its own code,
# instead of describing a hypothetical architecture. codephreak reads the ACTUAL
# source (read-only, confined to the repo root) and reports on what is truly here.
#
#   python3 -m services.self_audit                 # full grounded self-audit
#   python3 -m services.self_audit --focus security
#   python3 -m services.self_audit --file services/model_service.py   # read one file
#
# Safety: read-only; every path is confined to the repo root (realpath check, no
# traversal); binaries, large files, and build/dep dirs are skipped; a total
# character budget caps how much source is sent to the model.
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from .config import settings
from .model_service import ModelService

try:
    from automind import DEFAULT_SYSTEM_PROMPT as CODEPHREAK
except Exception:  # pragma: no cover
    CODEPHREAK = "You are Professor Codephreak. Audit honestly; cite files."

IGNORE_DIRS = {".git", "node_modules", ".next", "__pycache__", "models", ".venv",
               "venv", "memory", "saindbx", "terminai", "dist", "build", ".ruff_cache"}
IGNORE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bin", ".gguf",
              ".ggml", ".pt", ".db", ".jsonl", ".lock", ".woff", ".woff2", ".map"}
MAX_FILE = 40_000      # chars read per file
MAX_FILE_BYTES = 200_000
MAX_TOTAL = 240_000    # total source budget sent to the model


class SelfAudit:
    """Read-only, path-confined view of the repo for codephreak to audit."""

    def __init__(self, root: str = "."):
        self.root = os.path.realpath(root)
        self.model = ModelService()

    def _confined(self, path: str) -> bool:
        rp = os.path.realpath(path)
        return rp == self.root or rp.startswith(self.root + os.sep)

    def files(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() in IGNORE_EXT:
                    continue
                fp = os.path.join(dirpath, fn)
                if not self._confined(fp):
                    continue
                try:
                    if os.path.getsize(fp) > MAX_FILE_BYTES:
                        continue
                    with open(fp, encoding="utf-8", errors="ignore") as f:
                        out.append((os.path.relpath(fp, self.root), f.read()))
                except OSError:
                    pass
        return out

    def tree(self) -> List[str]:
        return [rel for rel, _ in self.files()]

    def read_file(self, rel: str) -> Optional[str]:
        """Read one file on demand (path-confined)."""
        fp = os.path.join(self.root, rel)
        if not self._confined(fp) or not os.path.isfile(fp):
            return None
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                return f.read(MAX_FILE)
        except OSError:
            return None

    def context(self, budget: int = MAX_TOTAL) -> str:
        files = self.files()
        parts = ["FILE TREE:\n" + "\n".join(rel for rel, _ in files) + "\n"]
        total = len(parts[0])
        for rel, content in files:
            block = f"\n===== {rel} =====\n{content[:MAX_FILE]}\n"
            if total + len(block) > budget:
                parts.append("\n[... remaining files truncated for budget ...]\n")
                break
            parts.append(block)
            total += len(block)
        return "".join(parts)

    def audit(self, focus: str = "", think: bool = False) -> str:
        prompt = (
            "Perform a REAL self-audit of automindX by reading the ACTUAL source below — "
            "do not describe a hypothetical architecture. "
            + (f"Focus: {focus}. " if focus else "")
            + "Report: (1) what the system actually is (real stack, entry points, how a "
            "request flows); (2) top 5 concrete strengths with file references; (3) top 5 "
            "concrete risks/improvements with file references; (4) any statement in the docs "
            "that does NOT match the code. Cite specific files.\n\n"
            "===== ACTUAL SOURCE =====\n" + self.context()
        )
        return self.model.predict(
            [{"role": "system", "content": CODEPHREAK}, {"role": "user", "content": prompt}],
            think=think,
        )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="codephreak reads the filesystem to self-audit automindX")
    ap.add_argument("--root", default=".")
    ap.add_argument("--focus", default="")
    ap.add_argument("--file", help="just read one file (path-confined) and print it")
    a = ap.parse_args()
    sa = SelfAudit(a.root)
    if a.file:
        print(sa.read_file(a.file) or f"[not found or outside root: {a.file}]")
    else:
        print(f"[codephreak reading {len(sa.tree())} files under {sa.root}]\n")
        print(sa.audit(a.focus))
