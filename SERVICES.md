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

## Versioned model config + integrity (audit #9)

`ModelRegistry` snapshots the reproducible tuple that determines a run —
`{model, options, persona hash, git SHA, Ollama model digest}` — under
`models/v{N}/metadata.json` with a `registry.json` index. Enables rollback
(`set_latest`) and reproducibility. Since models live in the Ollama daemon (not
torch `.pt` files), the **Ollama model digest** is the integrity anchor:
`verify()` confirms the daemon still serves the exact digest recorded at register
time — the analogue of a checkpoint SHA-256.

```python
from services import ModelRegistry
reg = ModelRegistry()
reg.register("gpt-oss:120b-cloud", options={"temperature": 0.7}, persona=prompt)
reg.verify()          # {ok, expected_digest, current_digest}
reg.set_latest("v1.0")  # rollback
```

## Secrets (audit #10)

`services.secrets.get_secret(name)` resolves from the **OS keyring** first
(`keyring`/Vault), then env — never hard-coded, never logged (`redact()`).
`ModelService` pulls the Ollama cloud key through it. Store one with
`set_secret("OLLAMA_API_KEY", value)` (goes to the keyring, not disk).

## Real self-audit — codephreak reads the filesystem

`services/self_audit.py` gives Professor Codephreak **read-only** access to the
actual source so it audits the real code instead of a hypothetical architecture:

```bash
python3 -m services.self_audit                          # grounded self-audit
python3 -m services.self_audit --focus security
python3 -m services.self_audit --file services/model_service.py   # read one file
```

```python
from services import SelfAudit
sa = SelfAudit(".")
sa.tree()          # files codephreak may read (dep/build dirs + binaries excluded)
sa.read_file(rel)  # one file, path-confined
sa.audit(focus)    # feeds the real source to codephreak → grounded report
```

Safety: every path is confined to the repo root (realpath check — no traversal),
binaries/large files and `node_modules`/`.git`/`models`/`.next` are skipped, and a
character budget caps how much source is sent to the model. Read-only — codephreak
inspects code, it does not execute or modify it.

## Container hardening (Podman)

Run the engine in a rootless, resource-capped **Podman** container:

```bash
podman run --rm --userns=keep-id --user 1000 \
  --memory 2g --cpus 2 --cap-drop ALL --read-only \
  -p 5001:5001 automindx:latest python3 codephreak.py
```

(Podman is rootless and daemonless by default — preferred over Docker here.)

## Codephreak audit coverage — complete

**#1** service layer · **#2** persistent memory (SQLite + pgvector) ·
**#3** lazy cached model · **#4** input sanitization (+ role-injection defense) ·
**#5** structured JSON logs · **#6** modular env config · **#7** pytest (15) ·
**#8** resource/overload limits · **#9** versioned config + digest integrity ·
**#10** keyring secrets · **#12** health check. Plus file perms (DB 0600) and
Podman container hardening. Model inference runs in the Ollama daemon, so the
torch-checkpoint sandbox subprocess (#5 variant) is N/A — isolation comes from
the daemon boundary + the overload guard + per-request timeout.
