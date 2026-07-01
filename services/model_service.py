# services/model_service.py
# Lazy, cached, thread-safe model client (codephreak audit #3).
# Wraps the working Ollama backend (not a torch checkpoint that must reload each
# call), so inference is warm and cheap. Singleton per process.
from __future__ import annotations

import os
import threading
from typing import List, Dict, Optional

import requests

from .config import settings


class ModelService:
    """Singleton Ollama client. The daemon keeps the model warm; we keep one
    session and reuse it."""

    _instance: Optional["ModelService"] = None
    _lock = threading.Lock()

    def __new__(cls, *a, **k):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model: Optional[str] = None):
        if getattr(self, "_ready", False):
            return
        self.model = model or settings.model
        self.host = settings.ollama_host
        self._session = requests.Session()
        # Secret resolved via keyring→env (audit #10) — never hard-coded/logged.
        from .secrets import get_secret
        key = get_secret(settings.ollama_api_key_env)
        if key:
            self._session.headers["Authorization"] = f"Bearer {key}"
        self._ready = True

    def ping(self) -> bool:
        """Is the model backend reachable? (for health checks)."""
        try:
            return self._session.get(f"{self.host}/api/tags", timeout=3).status_code == 200
        except Exception:
            return False

    def predict(self, messages: List[Dict], think: bool = False) -> str:
        """One chat completion. `messages` is [{role, content}, ...]."""
        try:
            r = self._session.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False, "think": think},
                timeout=settings.request_timeout,
            )
            if r.status_code != 200:
                return f"[model unavailable: HTTP {r.status_code}] {r.text[:200]}"
            return (r.json().get("message") or {}).get("content", "").strip() or "[empty response]"
        except requests.exceptions.ConnectionError:
            return "[Ollama unreachable — start `ollama serve`]"
        except Exception as e:  # pragma: no cover
            return f"[model error: {e}]"
