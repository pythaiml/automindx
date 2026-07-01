# services/memory_service.py
# Thread-safe, persistent short-/long-term memory (codephreak audit #1, #2).
# SQLite so context survives across sessions and processes — no external service.
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import closing
from typing import Dict, List, Optional

from .config import settings


class MemoryService:
    """Persistent per-session context store. Safe for concurrent use."""

    _lock = threading.RLock()

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.db_path
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False + our RLock = safe across threads.
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _ensure_schema(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS context (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       session_id TEXT NOT NULL,
                       role TEXT NOT NULL,
                       payload TEXT NOT NULL,
                       ts DATETIME DEFAULT CURRENT_TIMESTAMP
                   );"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON context(session_id, id);")
            conn.commit()

    def append(self, session_id: str, role: str, payload: Dict) -> None:
        """Store one JSON-serializable turn."""
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO context (session_id, role, payload) VALUES (?, ?, ?)",
                (session_id, role, json.dumps(payload, ensure_ascii=False)),
            )
            conn.commit()

    def retrieve(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Most recent `limit` turns for a session, oldest-first for prompting."""
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, payload FROM context WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        rows.reverse()
        return [{"role": r[0], **json.loads(r[1])} for r in rows]

    def search(self, query: str, session_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Keyword search — SQLite's compatible stand-in for vector similarity.
        (RageMemory overrides this with true semantic search.)"""
        like = f"%{query.strip()}%"
        sql = "SELECT role, payload FROM context WHERE payload LIKE ?"
        args: list = [like]
        if session_id:
            sql += " AND session_id = ?"; args.append(session_id)
        sql += " ORDER BY id DESC LIMIT ?"; args.append(limit)
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(sql, args).fetchall()
        return [{"role": r[0], **json.loads(r[1])} for r in rows]

    def sessions(self) -> List[str]:
        with self._lock, closing(self._connect()) as conn:
            return [r[0] for r in conn.execute("SELECT DISTINCT session_id FROM context").fetchall()]

    def clear(self, session_id: str) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute("DELETE FROM context WHERE session_id = ?", (session_id,))
            conn.commit()
