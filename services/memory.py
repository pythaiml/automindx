# services/memory.py
# Memory backend factory. Both backends share the same interface
# (append / retrieve / search / sessions / clear):
#   • "sqlite"   → MemoryService  — zero-dependency, always works (compatible default)
#   • "pgvector" → RageMemory     — RAGE semantic memory (mindX pattern)
#
# If the requested backend can't initialize (deps/DB missing), we fall back to
# SQLite so automindX always has a working memory service.
from __future__ import annotations

import logging

from .config import settings
from .memory_service import MemoryService

logger = logging.getLogger("automindx.memory")


def get_memory():
    backend = (settings.memory_backend or "sqlite").lower()
    if backend in ("pgvector", "rage", "postgres", "postgresql"):
        try:
            from .rage_memory import RageMemory
            mem = RageMemory()
            logger.info("memory backend: pgvector (RAGE)")
            return mem
        except Exception as e:
            logger.warning("pgvector backend unavailable (%s) — falling back to SQLite", e)
    logger.info("memory backend: sqlite")
    return MemoryService()
