# services/rage_memory.py
# RAGE semantic memory — pgvector / pgvectorscale backend, brought over from the
# mindX pattern (scripts/setup_memory_db.py, migrate_to_pgvector.py):
#   • embeddings via sentence-transformers all-MiniLM-L6-v2 (384 dims)
#   • memory + memory_embeddings tables, vector column, cosine search
#   • ivfflat index; ON CONFLICT DO NOTHING
#
# Optional: requires psycopg2 + pgvector + sentence-transformers + a running
# Postgres with the `vector` extension. If any piece is missing the constructor
# raises RuntimeError and services.memory.get_memory() falls back to SQLite —
# so automindX always has a working memory service.
from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

from .config import settings


class RageMemory:
    """pgvector-backed semantic memory (drop-in for MemoryService + .search())."""

    _lock = threading.RLock()

    def __init__(self):
        try:
            import psycopg2  # noqa
            from pgvector.psycopg2 import register_vector  # noqa
            from sentence_transformers import SentenceTransformer  # noqa
        except Exception as e:  # deps not installed → caller falls back to SQLite
            raise RuntimeError(f"RAGE (pgvector) deps unavailable: {e}")

        import psycopg2
        from pgvector.psycopg2 import register_vector
        from sentence_transformers import SentenceTransformer

        password = os.getenv(settings.pg_password_env, "")
        self._conn = psycopg2.connect(
            host=settings.pg_host, port=settings.pg_port, dbname=settings.pg_db,
            user=settings.pg_user, password=password,
        )
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(self._conn)
            cur.execute(
                """CREATE TABLE IF NOT EXISTS memory (
                       id BIGSERIAL PRIMARY KEY,
                       session_id TEXT NOT NULL,
                       role TEXT NOT NULL,
                       payload JSONB NOT NULL,
                       ts TIMESTAMPTZ DEFAULT now()
                   );"""
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_session ON memory(session_id, id);")
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS memory_embeddings (
                        memory_id BIGINT PRIMARY KEY REFERENCES memory(id) ON DELETE CASCADE,
                        embedding vector({settings.embed_dim}) NOT NULL,
                        text_content TEXT NOT NULL
                    );"""
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_embed ON memory_embeddings "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
            )
        self._model = SentenceTransformer(settings.embed_model)

    def _embed(self, text: str) -> List[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def append(self, session_id: str, role: str, payload: Dict) -> None:
        text = payload.get("text", "")
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memory (session_id, role, payload) VALUES (%s, %s, %s) RETURNING id",
                (session_id, role, json.dumps(payload, ensure_ascii=False)),
            )
            mem_id = cur.fetchone()[0]
            if text.strip():
                cur.execute(
                    "INSERT INTO memory_embeddings (memory_id, embedding, text_content) "
                    "VALUES (%s, %s, %s) ON CONFLICT (memory_id) DO NOTHING",
                    (mem_id, self._embed(text), text),
                )

    def retrieve(self, session_id: str, limit: int = 10) -> List[Dict]:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT role, payload FROM memory WHERE session_id=%s ORDER BY id DESC LIMIT %s",
                (session_id, limit),
            )
            rows = cur.fetchall()
        rows.reverse()
        return [{"role": r[0], **(r[1] if isinstance(r[1], dict) else json.loads(r[1]))} for r in rows]

    def search(self, query: str, session_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """True semantic search: nearest embeddings by cosine distance (<=>)."""
        emb = self._embed(query)
        sql = ("SELECT m.role, m.payload, 1 - (e.embedding <=> %s::vector) AS sim "
               "FROM memory_embeddings e JOIN memory m ON m.id = e.memory_id ")
        args: list = [emb]
        if session_id:
            sql += "WHERE m.session_id = %s "; args.append(session_id)
        sql += "ORDER BY e.embedding <=> %s::vector LIMIT %s"; args += [emb, limit]
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()
        out = []
        for role, payload, sim in rows:
            if sim is not None and sim < settings.min_similarity:
                continue
            d = payload if isinstance(payload, dict) else json.loads(payload)
            out.append({"role": role, "similarity": round(float(sim), 3), **d})
        return out

    def sessions(self) -> List[str]:
        with self._lock, self._conn.cursor() as cur:
            cur.execute("SELECT DISTINCT session_id FROM memory")
            return [r[0] for r in cur.fetchall()]

    def clear(self, session_id: str) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute("DELETE FROM memory WHERE session_id=%s", (session_id,))
