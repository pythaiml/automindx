# Save the Trees — sAGI memory as a git-like tree, embedded into RAGE

How an individual sAGI **saves its memory**: it grows a content-addressed memory
tree (**gitmind**) and embeds those trees into **RAGE** (pgvectorscale) so memory is
both *time-travellable* (revisit any `.history` moment) and *semantically
searchable*. Nothing leaves the machine unless you point RAGE at a remote database.

## The stack

```
modules/ (what sAGI grows)
   │  every persisted moment
   ▼
gitmind  ── sagi/runtime/gitmind.py ───────────────────────────────────────────
   • content-addressed object store under <SAGI_DIR>/.gitmind/ (blob→tree→commit)
   • LOCAL commits  = per-timestamp timeline   (vertical scaling)
   • GLOBAL commits = expansion / massive upgrade milestones (horizontal scaling)
   • ports github.com/Professor-Codephreak/gitmind; Forgejo / mindX /mindx/gitmind
   │  RageSync (on global commits, or on demand)
   ▼
RAGE  ── sagi/runtime/rage_sync.py → services/rage_memory.py ────────────────────
   • pgvector / pgvectorscale, all-MiniLM-L6-v2 embeddings (384-dim), ivfflat cosine
   • the mindX RAGE substrate — https://rage.pythai.net · github.com/GATERAGE/RAGE
   • falls back to SQLite keyword search when pgvector isn't configured
```

## What gets saved, and how it maps to `.history`

Each `.history/build.jsonl` moment (a `run_start` / `module` / `run_end` line, stamped
with `ts`) corresponds to a **gitmind commit** taken at that moment. So:

- `gitmind.at_moment(ts)` → the commit as of a `.history` timestamp,
- `gitmind.tree_at_moment(ts)` → the exact `{file: content}` memory then,
- `gitmind.log()` / `global_log()` → the full timeline / the milestone chain,
- `RageSync.search(query)` → semantic recall across every saved tree, each hit
  tagged with the `commit`, `ts`, `scope`, and `moment` it came from.

## Workflow

1. **Grow.** From the console green-light chooser (or `sagi_build.py`), sAGI proposes
   → specifies → **persists** a module. See [POINT_OF_DEPARTURE.md](../sagi/POINT_OF_DEPARTURE.md).
2. **Commit (local).** `boot()` attaches `host.memory` (gitmind) and commits a **local**
   snapshot on every `module.persisted` — the per-timestamp timeline.
3. **Milestone (global).** Genesis and any expansion / massive-upgrade move is a
   **global** commit (`host.memory.global_commit(...)`), walkable via `global_log()`.
4. **Save the trees to RAGE.** Embed trees into pgvectorscale — automatically on
   global commits (`boot(rage=True)`), or on demand:

   ```bash
   python3 -m sagi.runtime.rage_sync --dir "$SAGI_DIR"              # save all trees
   python3 -m sagi.runtime.rage_sync --dir "$SAGI_DIR" --search "zk verifier"
   ```

   ```python
   from sagi.runtime import boot, RageSync
   host, _ = boot(SAGI_DIR, rage=True)     # global commits auto-embed into RAGE
   RageSync(host.memory).save_all()        # or embed the whole history now
   RageSync(host.memory).search("how did I model consensus?")
   ```
5. **Recall from a moment.** Combine both: find a `.history` moment, get its tree, and
   semantically search the saved trees.

   ```python
   snap = host.memory.at_moment(ts)                 # memory as of a .history moment
   past = host.memory.tree_at_moment(ts)            # {file: content} then
   hits = host.rage.search("what did I know about X?")
   ```

## Configuration

| Env | Meaning |
|---|---|
| `AUTOMINDX_MEMORY_BACKEND` | `pgvector` (RAGE) or `sqlite` (fallback). See [SERVICES.md](SERVICES.md). |
| `SAGI_DIR` | the individual's package (its own `modules/`, `.history/`, `.gitmind/`). |
| `SAGI_BACKEND` | model backend for building (`ollama` / `claude-cli` / `claude-api`). |

The `.gitmind/` object store and the SQLite/pgvector data are **runtime state** (git-ignored);
these seed specs and code are what's version-controlled.

## References

- **[Professor-Codephreak/gitmind](https://github.com/Professor-Codephreak/gitmind)** — upstream gitmind
- **[rage.pythai.net](https://rage.pythai.net)** · **[GATERAGE/RAGE](https://github.com/GATERAGE/RAGE)** — the RAGE substrate
- **mindX `/mindx/gitmind`** · **Forgejo** — the git-based-memory / self-hosted-forge lineage
- [SERVICES.md](SERVICES.md) — the RAGE / pgvector service layer
- [../sagi/README.md](../sagi/README.md) · [../sagi/POINT_OF_DEPARTURE.md](../sagi/POINT_OF_DEPARTURE.md)
- <a href="https://rage.pythai.net/save-the-trees-professor-codephreak-automindx/">Save the trees</a>
