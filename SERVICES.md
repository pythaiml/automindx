# automindX service layer

Implements Professor Codephreak's audit blueprint (mindx.pythai.net/automindX) as
a decoupled `services/` package, adapted to the working Ollama backend so it runs.

```
services/
  config.py                 env-driven settings (audit #6) — stdlib, always loads
  memory_service.py         SQLite persistent memory (audit #1, #2) — the compatible default
  rage_memory.py            RAGE semantic memory — pgvector/pgvectorscale (mindX pattern)
  memory.py                 get_memory() factory: picks backend, falls back to SQLite
  model_service.py          lazy, cached, thread-safe Ollama client (audit #3)
  inference_orchestrator.py sanitize + RAG recall + infer + persist + structured logs (audit #1,#4,#5)
tests/test_services.py      pytest coverage (audit #7)
```

## Use

```python
from services import InferenceOrchestrator
orch = InferenceOrchestrator()
out = orch.run("hello", session_id=None)   # returns {session_id, response, duration_ms, status}
```

Or via the thin façade (codephreak integration step C) — `automind.py` now
delegates to the service layer:

```python
from automind import chat
chat("hello")                 # → assistant text
chat("hello", full=True)      # → {session_id, request_id, response, duration_ms, status}
```

Or the REPL: `python3 -m services.inference_orchestrator`.

## Memory backends (SQLite ↔ pgvector)

Selected by `AUTOMINDX_MEMORY_BACKEND`:

| Backend | Value | Needs | Retrieval |
|---|---|---|---|
| **SQLite** (default) | `sqlite` | nothing (stdlib) | recency + keyword `search()` |
| **RAGE / pgvector** | `pgvector` | psycopg2 + pgvector + sentence-transformers + Postgres | cosine semantic `search()` |

Both expose the same interface (`append · retrieve · search · sessions · clear`),
so the orchestrator is backend-agnostic. **SQLite is the compatible fallback
service** — if the pgvector backend can't initialize (missing deps or DB), the
factory logs a warning and returns SQLite, so automindX always has working memory.

The pgvector backend mirrors mindX: `all-MiniLM-L6-v2` (384-dim) embeddings, a
`memory_embeddings` vector column, an `ivfflat` cosine index, and `min_similarity`
filtering. RAG recall surfaces semantically-relevant prior turns into the prompt.

## Config (env)

```
AUTOMINDX_MEMORY_BACKEND=sqlite|pgvector   AUTOMINDX_DB=./memory/automindx.db
AUTOMINDX_MODEL=qwen3:0.6b                 OLLAMA_HOST=http://localhost:11434
AUTOMINDX_PG_HOST / _PORT / _DB / _USER    AUTOMINDX_PG_PASSWORD_ENV (secret via env, never hard-coded)
AUTOMINDX_EMBED_MODEL=all-MiniLM-L6-v2     AUTOMINDX_EMBED_DIM=384   AUTOMINDX_MIN_SIMILARITY=0.5
```

Optional deps for the RAGE backend: `pip install psycopg2-binary pgvector sentence-transformers`.

## Resource limits & health (audit #8, #12)

- **Overload guard** — `InferenceOrchestrator.run()` acquires a bounded semaphore
  (`AUTOMINDX_MAX_CONCURRENCY`, default = CPU count); at capacity it rejects fast
  with `status: "busy"` rather than queuing unboundedly (DoS protection).
- **Health** — `orchestrator.health()` reports model reachability + memory backend;
  the running `codephreak.py` engine exposes **`GET /healthz`** for Docker/K8s probes.
- Per-request timeout is enforced on the model call (`AUTOMINDX_TIMEOUT`).

## Codephreak audit coverage

Done: **#1** service layer · **#2** persistent memory (SQLite + pgvector) ·
**#3** lazy cached model · **#4** input sanitization · **#5** structured JSON logs ·
**#6** modular env config · **#7** pytest · **#8** resource/overload limits ·
**#12** health check. Deferred (rationale in [CODEPHREAK_AUDIT.md](CODEPHREAK_AUDIT.md)):
**#9** versioned checkpoints and **#10** OS-keyring secrets (N/A / minimal for the
Ollama-daemon model + local-first posture; secrets already env-only, never embedded).
