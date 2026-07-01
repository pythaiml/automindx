# services/ — the decoupled service layer for automindX.
#
# Implements Professor Codephreak's audit blueprint (mindx.pythai.net/automindX):
#   1. Decouple via a service layer      → MemoryService · ModelService · InferenceOrchestrator
#   2. Persist long-term memory          → SQLite (MemoryService)
#   3. Lazy-load & cache the model       → ModelService (thread-safe singleton)
#   4. Input sanitization                → InferenceOrchestrator._sanitize
#   5. Structured logging & metrics      → JSON logs with request_id / duration_ms / status
#   6. Modular config                    → services.config.settings (env-driven, stdlib)
#
# Adapted from codephreak's sketch to the working Ollama backend (no torch
# checkpoint), so it actually runs and loads.
from .config import settings
from .memory_service import MemoryService
from .memory import get_memory
from .model_service import ModelService
from .inference_orchestrator import InferenceOrchestrator
from .model_registry import ModelRegistry          # audit #9 — versioned config + integrity
from .secrets import get_secret, set_secret, redact  # audit #10 — secrets management

# RAGE semantic memory (pgvector) is optional; imported lazily by get_memory().
__all__ = ["settings", "MemoryService", "get_memory", "ModelService",
           "InferenceOrchestrator", "ModelRegistry", "get_secret", "set_secret", "redact"]
