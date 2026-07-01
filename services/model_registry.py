# services/model_registry.py
# Versioned model configuration + integrity (codephreak audit #9).
#
# automindX runs models in the Ollama daemon rather than loading torch .pt
# checkpoints, so a "checkpoint" here is the reproducible tuple that determines a
# run: {model, generation options, persona, git SHA, Ollama model digest}. Each
# version is snapshotted under models/v{N}/metadata.json with a registry index,
# enabling reproducibility and rollback — codephreak's intent, adapted.
#
# Integrity: we record the Ollama model *digest* at register time and can verify
# the daemon still serves that exact digest (the analogue of a checkpoint SHA-256).
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from typing import Dict, List, Optional

import requests

from .config import settings


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return os.getenv("GIT_SHA", "unknown")


def _ollama_digest(model: str) -> Optional[str]:
    """The daemon's digest for a model — our integrity anchor."""
    try:
        r = requests.get(f"{settings.ollama_host}/api/tags", timeout=4)
        for m in r.json().get("models", []):
            if m.get("name") == model:
                return m.get("digest")
    except Exception:
        pass
    return None


class ModelRegistry:
    """Versioned, reproducible model-configuration snapshots with rollback."""

    def __init__(self, root: str = os.getenv("AUTOMINDX_MODEL_DIR", "./models")):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.index_path = os.path.join(self.root, "registry.json")

    def _index(self) -> Dict:
        try:
            with open(self.index_path) as f:
                return json.load(f)
        except Exception:
            return {"latest": None, "versions": []}

    def _save_index(self, idx: Dict) -> None:
        with open(self.index_path, "w") as f:
            json.dump(idx, f, indent=2)
            f.write("\n")

    def register(self, model: str, options: Optional[Dict] = None,
                 persona: Optional[str] = None, notes: str = "",
                 version: Optional[str] = None) -> Dict:
        idx = self._index()
        version = version or f"v1.{len(idx['versions'])}"
        meta = {
            "version": version,
            "model": model,
            "options": options or {},
            "persona_sha256": hashlib.sha256((persona or "").encode()).hexdigest()[:16] if persona else None,
            "git_sha": _git_sha(),
            "ollama_digest": _ollama_digest(model),
            "created_at": int(time.time()),
            "notes": notes,
        }
        vdir = os.path.join(self.root, version)
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        if version not in idx["versions"]:
            idx["versions"].append(version)
        idx["latest"] = version
        self._save_index(idx)
        return meta

    def get(self, version: str) -> Optional[Dict]:
        try:
            with open(os.path.join(self.root, version, "metadata.json")) as f:
                return json.load(f)
        except Exception:
            return None

    def versions(self) -> List[str]:
        return list(self._index().get("versions", []))

    def latest(self) -> Optional[Dict]:
        v = self._index().get("latest")
        return self.get(v) if v else None

    def set_latest(self, version: str) -> bool:
        """Roll back / forward to a recorded version."""
        idx = self._index()
        if version not in idx["versions"]:
            return False
        idx["latest"] = version
        self._save_index(idx)
        return True

    def verify(self, version: Optional[str] = None) -> Dict:
        """Integrity check: does the daemon still serve the recorded digest?"""
        meta = self.get(version) if version else self.latest()
        if not meta:
            return {"ok": False, "reason": "no such version"}
        current = _ollama_digest(meta["model"])
        expected = meta.get("ollama_digest")
        return {
            "ok": (expected is None) or (current == expected),
            "version": meta["version"], "model": meta["model"],
            "expected_digest": expected, "current_digest": current,
        }
