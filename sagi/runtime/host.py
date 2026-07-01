# sagi/runtime/host.py — the agnostic host surface { log, store, callModel, on, emit }
# from sagi/core/interface.md. Every module receives this host and plugs into it;
# it never knows which model backend, UI, or process it runs in.
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional


class Store:
    """Path-confined read/write rooted at the individual's SAGI_DIR."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.modules_dir = os.path.join(self.root, "modules")

    def _safe(self, rel: str) -> str:
        p = os.path.abspath(os.path.join(self.root, rel))
        if p != self.root and not p.startswith(self.root + os.sep):
            raise ValueError(f"path escapes the sagi package: {rel}")
        return p

    def read(self, rel: str, default: Optional[str] = None) -> Optional[str]:
        try:
            with open(self._safe(rel), encoding="utf-8") as f:
                return f.read()
        except OSError:
            return default

    def read_json(self, rel: str, default: Any = None) -> Any:
        raw = self.read(rel)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    def write(self, rel: str, content: str) -> None:
        p = self._safe(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)

    def write_json(self, rel: str, obj: Any) -> None:
        self.write(rel, json.dumps(obj, indent=2) + "\n")

    def list_modules(self) -> List[str]:
        try:
            return sorted(f for f in os.listdir(self.modules_dir) if f.endswith(".md"))
        except OSError:
            return []


class Host:
    """The surface every sAGI module plugs into."""

    def __init__(self, sagi_dir: str, call_model: Optional[Callable[[str, str], str]] = None):
        self.root = os.path.abspath(sagi_dir)
        self.store = Store(self.root)
        self._handlers: Dict[str, List[Callable[[Any], None]]] = {}
        self._call_model = call_model
        self._prompt_prefix = ""            # set by individuality-core
        os.makedirs(self.store.modules_dir, exist_ok=True)
        ident = self.store.read_json("identity.json", default={}) or {}
        self.identity_id: str = ident.get("id") or "sagi"
        self.backend: str = ident.get("backend") or os.environ.get("SAGI_BACKEND", "ollama")

    # --- log -> .history/build.jsonl ---
    def log(self, event: str, **fields: Any) -> Dict[str, Any]:
        hist = os.path.join(self.root, ".history")
        os.makedirs(hist, exist_ok=True)
        rec = {"ts": int(time.time()), "event": event, **fields}
        try:
            with open(os.path.join(hist, "build.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        return rec

    # --- pub/sub ---
    def on(self, event: str, fn: Callable[[Any], None]) -> None:
        self._handlers.setdefault(event, []).append(fn)

    def emit(self, event: str, payload: Any = None) -> None:
        for fn in list(self._handlers.get(event, [])):
            fn(payload)

    # --- model (individuality-core sets the composed persona prefix) ---
    def set_prompt_prefix(self, prefix: str) -> None:
        self._prompt_prefix = prefix or ""

    def call_model(self, prompt: str, system: Optional[str] = None) -> str:
        sys_prompt = self._prompt_prefix or system or ""
        if self._call_model is None:
            raise RuntimeError("no model backend wired into this host")
        return self._call_model(sys_prompt, prompt)

    # camelCase alias to match interface.md's { callModel }
    callModel = call_model
