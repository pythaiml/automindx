# services/inference_orchestrator.py
# Ties memory + model together with input sanitization and structured logging
# (codephreak audit #1, #4, #5). This is the one public entry point; the modules
# below it never touch each other directly.
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Optional

# Resource guard (audit #8): cap in-flight requests so one client can't exhaust
# the box. Excess requests are rejected fast rather than queuing unboundedly.
_MAX_CONCURRENCY = int(os.getenv("AUTOMINDX_MAX_CONCURRENCY", str(os.cpu_count() or 4)))
_INFLIGHT = threading.BoundedSemaphore(_MAX_CONCURRENCY)

from .config import settings
from .memory import get_memory
from .model_service import ModelService

# The Professor Codephreak persona (audit uses the real prompt).
try:
    from automind import DEFAULT_SYSTEM_PROMPT as CODEPHREAK
except Exception:  # pragma: no cover
    CODEPHREAK = "You are Professor Codephreak. Answer step by step; be concise."

# Reject control chars / non-printables; keep printable ASCII + Latin-1 range.
_UNSAFE = re.compile(r"[^\x20-\x7E -ÿ\n\t]+")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {"level": record.levelname, "msg": record.getMessage()}
        for k in ("request_id", "session_id", "duration_ms", "status"):
            if hasattr(record, k):
                base[k] = getattr(record, k)
        return json.dumps(base, ensure_ascii=False)


def _logger() -> logging.Logger:
    lg = logging.getLogger("automindx.inference")
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(_JsonFormatter())
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


class InferenceOrchestrator:
    """Perceive → build → infer → persist → log, per request."""

    def __init__(self, system_prompt: str = CODEPHREAK):
        self.system_prompt = system_prompt
        self.mem = get_memory()          # sqlite (default) or pgvector/RAGE
        self.model = ModelService()
        self.log = _logger()

    @staticmethod
    def _sanitize(text: str) -> str:
        cleaned = _UNSAFE.sub("", text or "")
        # Neutralize chat-template role markers so user input can't spoof a
        # system/assistant turn (prompt-injection defense — codephreak checklist).
        cleaned = re.sub(
            r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>|<</?SYS>>|\[/?INST\]",
            "", cleaned, flags=re.I,
        )
        return cleaned.strip()[:8000]

    def _build_messages(self, sanitized: str, session_id: str) -> list:
        msgs = [{"role": "system", "content": self.system_prompt}]
        # RAG: pull semantically-relevant prior turns (pgvector) or keyword
        # matches (sqlite) and surface them as recalled context.
        try:
            relevant = self.mem.search(sanitized, session_id, limit=3)
        except Exception:
            relevant = []
        recalled = "\n".join(f"- {r.get('text', '')}" for r in relevant if r.get("text"))
        if recalled:
            msgs.append({"role": "system", "content": "Relevant recalled memory:\n" + recalled})
        for turn in self.mem.retrieve(session_id, limit=settings.max_context):
            msgs.append({"role": turn.get("role", "user"), "content": turn.get("text", "")})
        msgs.append({"role": "user", "content": sanitized})
        return msgs

    def health(self) -> dict:
        """Liveness/readiness snapshot (audit #12)."""
        ok = self.model.ping()
        return {"status": "ok" if ok else "degraded", "model": self.model.model,
                "memory": type(self.mem).__name__, "ollama": ok,
                "capacity": _MAX_CONCURRENCY}

    def run(self, user_input: str, session_id: Optional[str] = None) -> dict:
        session_id = session_id or str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        start = time.time()

        sanitized = self._sanitize(user_input)
        if not sanitized:
            return {"session_id": session_id, "response": "", "status": "empty-input"}

        # Overload protection: reject fast when at capacity (audit #8).
        if not _INFLIGHT.acquire(blocking=False):
            self.log.warning("rejected", extra={"request_id": request_id, "session_id": session_id, "status": "busy"})
            return {"session_id": session_id, "request_id": request_id,
                    "response": "Server at capacity — please retry shortly.", "status": "busy"}
        try:
            messages = self._build_messages(sanitized, session_id)
            response = self.model.predict(messages)
            self.mem.append(session_id, "user", {"text": sanitized})
            self.mem.append(session_id, "assistant", {"text": response})
        finally:
            _INFLIGHT.release()

        duration_ms = round((time.time() - start) * 1000, 1)
        self.log.info("inference", extra={
            "request_id": request_id, "session_id": session_id,
            "duration_ms": duration_ms, "status": "ok",
        })
        return {"session_id": session_id, "request_id": request_id,
                "response": response, "duration_ms": duration_ms, "status": "ok"}


if __name__ == "__main__":
    # Tiny REPL demo: python3 -m services.inference_orchestrator
    orch = InferenceOrchestrator()
    sid = None
    print("automindX service layer — type 'exit' to quit.")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit"):
            break
        out = orch.run(q, sid)
        sid = out["session_id"]
        print("codephreak>", out["response"])
