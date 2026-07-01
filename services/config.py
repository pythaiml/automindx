# services/config.py
# Centralized, env-driven configuration (codephreak audit #6). Stdlib only, so it
# always loads — pydantic/hydra optional, not required.
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model: str = os.getenv("AUTOMINDX_MODEL", "qwen3:0.6b")
    max_context: int = int(os.getenv("AUTOMINDX_MAX_CONTEXT", "10"))
    request_timeout: int = int(os.getenv("AUTOMINDX_TIMEOUT", "300"))
    # Never hard-code secrets (audit #10). API keys/tokens come from the env or an
    # OS keyring; this layer only reads names, never embeds values.
    ollama_api_key_env: str = os.getenv("AUTOMINDX_OLLAMA_KEY_ENV", "OLLAMA_API_KEY")

    # ── Memory backend ────────────────────────────────────────────────────
    # "sqlite" (default, zero-dep, always works) or "pgvector" (RAGE semantic
    # memory, mindX pattern). SQLite is the compatible fallback service.
    memory_backend: str = os.getenv("AUTOMINDX_MEMORY_BACKEND", "sqlite")
    db_path: str = os.getenv("AUTOMINDX_DB", "./memory/automindx.db")

    # pgvector / pgvectorscale (RAGE) — mirrors mindX's memory config.
    pg_host: str = os.getenv("AUTOMINDX_PG_HOST", "localhost")
    pg_port: int = int(os.getenv("AUTOMINDX_PG_PORT", "5432"))
    pg_db: str = os.getenv("AUTOMINDX_PG_DB", "automindx_memory")
    pg_user: str = os.getenv("AUTOMINDX_PG_USER", "automindx")
    pg_password_env: str = os.getenv("AUTOMINDX_PG_PASSWORD_ENV", "AUTOMINDX_DB_PASSWORD")
    embed_model: str = os.getenv("AUTOMINDX_EMBED_MODEL", "all-MiniLM-L6-v2")
    embed_dim: int = int(os.getenv("AUTOMINDX_EMBED_DIM", "384"))
    min_similarity: float = float(os.getenv("AUTOMINDX_MIN_SIMILARITY", "0.5"))


settings = Settings()
