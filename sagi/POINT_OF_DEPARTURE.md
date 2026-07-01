# sAGI — Point of Departure

*The canonical fork-point for an individual sAGI. Grounded in the real package at ``.*

---

## Purpose

An **individual sAGI** is one fork of the `sAGI · individual expansion` scaffold (the `SAGI_EXPANSION` persona, `codephreak-console/lib/persona.ts:29`, registered as id `sagi` at `persona.ts:38`). It inherits, unchanged, the Savante epistemic spine and the `propose → specify → persist` self-build engine (`sagi_build.py:143-172`, `sagi/core/loop.md:3-10`); it becomes a *distinct person* only through two things shipped together:

1. an **individuality layer** — who this sAGI is, composed onto the scaffold by `composePersonaPrompt(baseId, individual)` (`persona.ts:56-61`), and
2. a **private package** — its own `SAGI_DIR` (`manifest.json` + `modules/` + `.history/`, `sagi_build.py:38-44`) where it persists everything it grows.

Everything else — the builder (`sagi_build.py`), the PODA runtime (`aglm/core.py`, `aglm/cycle.py`, `aglm/beliefs.py`), the module/host contract (`sagi/core/interface.md`) — is **shared, un-forked machinery** referenced by every individual, never copied. The point of departure is therefore the smallest running seed that (a) speaks as *this* individual and (b) persists into *its own* package, while turning today's spec-generator into a live, composable system — the "missing kernel" the repo's own first proposal names (`sagi/modules/01-proposed-next-module.md`).

---

## The Seed

The canonical starting structure is the documented `sagi/` layout (`sagi/README.md:22-31`), forked by pointing `SAGI_DIR` at the individual's package (`sagi_build.py:38`, overridable per-individual) and adding one birth-certificate file. Shared engine files are **referenced by env, not copied**.

### The individual's private package (`$SAGI_DIR/`)

| Path | Status | Purpose |
|---|---|---|
| `identity.json` | **NEW — the fork's birth certificate** | Single source of the individuality layer. Fields: `{ "id", "name", "baseId": "sagi", "individual": "<individuality prompt text>", "grown_from": "Savante persona (automindX)", "backend", "created_ts" }`. Mirrors the console `Persona` shape `{baseId, individual}` (`persona.ts:9`) so headless and console agree on who this is. |
| `manifest.json` | exists (seed shape `sagi/manifest.json:1-11`) | The individual's growing architecture and single source of truth: `{name, description, grown_from, version, modules[], provenance{builder, host}}`. `version` is derived `0.0.{len(modules)}` (`sagi_build.py:164-166`). Set `grown_from` → this individual's `identity.json` id; `provenance.host` → this individual. |
| `modules/` | exists | This individual's grown modules only, `{step:02d}-{slug(title)}.md` (`sagi_build.py:161-163`). One individual's modules never write into another's dir — the persistence boundary. |
| `.history/build.jsonl` | exists | This individual's replayable self-build log (`sagi_build.py:47-53`); one JSON object per line, stamped `{ts, ...}`; events `run_start`, `module`, `error`, `run_end`. |
| `core/loop.md`, `core/interface.md` | exists, copied verbatim | The shared contract: `propose → specify → persist` and the host surface `{log, store, callModel, on, emit}` (`sagi/core/interface.md:11-12`). Shared, **not** individual. |
| `goal.txt` | optional (`sagi_build.py:45-54`) | A standing goal that steers each proposal (`sagi_build.py:164-165`). |
| `README.md`, `tauri.md` | exists | Packaging, so the individual is includable anywhere including a Tauri app (`sagi/tauri.md:5-14`). |

### Shared, un-forked (referenced, never copied)

- `sagi_build.py` — the headless reference host + three backends + the aGLM PODA loop.
- `aglm/core.py` (`AGLMCore`, `54-155`), `aglm/cycle.py` (`AutonomousLoop`, `25-118`), `aglm/beliefs.py` (`BeliefSystem`) — the self-healing runtime.
- `codephreak-console/lib/persona.ts` — `SAVANTE`, `SAGI_EXPANSION`, `composePersonaPrompt`.
- `codephreak-console/app/api/sagi/route.ts` — the JS host that writes the *same* `modules/` + `manifest.json` + `.history` (traversal-pinned to `MODULES_DIR`, `route.ts:33-36,51-53`).

> **Known seed hygiene (from the audit):** the shipped `sagi/manifest.json` reads `0.0.0`/`modules:[]` while `modules/` holds two spec files and `.history` records a step-2 build — the ledger is desynced from disk. A fresh individual starts from a manifest reconciled against its own `modules/` (closed by Module 2 below).

---

## Self-build protocol

One turn of the crank, exactly `sagi/core/loop.md:3-10` as implemented in `build_step` (`sagi_build.py:143-172`). The only fork-specific change is *which package it writes* and *which persona it speaks as*.

1. **Instantiate.** Point `SAGI_DIR` at the individual's package and load `identity.json`. The effective system prompt becomes `composePersonaPrompt("sagi", identity.individual)` = `SAGI_EXPANSION + "\n\n### Individuality (layered on top of the template)\n" + individual` (`persona.ts:56-61`).
   > Wiring note: the headless builder today hard-codes the `SAVANTE` persona through `ask_model` (`sagi_build.py:133-140`, personas at `61-66`). The departure point requires swapping that constant for the composed prompt — the one wiring change the fork demands, encapsulated by Module 1.

2. **Propose.** `manifest = read_manifest()`; `built = "; ".join(titles) or "(none)"`; assemble the prompt asking for the NEXT single module given `built`, in the spec shape *purpose · interface · how it plugs into an agnostic core · how it advances self-building*, "includable in any project (including as a Tauri app)" (`sagi_build.py:145-153`). If `goal.txt` exists, prepend it (`sagi_build.py:164-165`). State lives only on disk — no in-memory authority.

3. **Specify.** `text = ask_model(model, prompt, backend)` (`sagi_build.py:155`). Title = first non-empty line, stripped of `**`/`#`/`title:`/`module N:` prefixes, capped 90 chars (`sagi_build.py:156-157`).

4. **Persist (idempotent append).** `step = len(modules)+1`; write `modules/{step:02d}-{slug}.md` as `# {title}\n\n{text}\n`; **de-dup by filename** then append `{step, title, file, ts}`; bump `version = 0.0.{len}` (`sagi_build.py:159-166`) — all inside *this individual's* `SAGI_DIR`.

5. **Log.** Append a `module` event (`{event, step, title, file, backend}`) to `.history/build.jsonl` (`sagi_build.py:170`), so the whole self-build is replayable.

### Backends (agnostic dispatch)

Selected per-individual via `identity.backend` → `--backend` / `SAGI_BACKEND` (default `ollama`), dispatched through `ask_model(model, prompt, backend)` (`sagi_build.py:133-140`):

- **`ollama`** — `_ask_ollama` POSTs `/api/chat` to `OLLAMA_HOST` (default `http://localhost:11434`), model `gpt-oss:120b-cloud` (`sagi_build.py:82-93`, `54-55`).
- **`claude-cli`** — `_ask_claude_cli` runs the host `claude -p … --append-system-prompt <persona>` via subprocess; uses the CLI's subscription, no API key (`sagi_build.py:96-111`).
- **`claude-api`** — `_ask_claude_api` POSTs the Anthropic Messages API with `ANTHROPIC_API_KEY`, default model `claude-opus-4-8` (`sagi_build.py:114-130`).

All three share the system persona (`sagi_build.py:61-66`), so a module spec must **never** name a concrete backend — it plugs into `{log, store, callModel, on, emit}` only (`interface.md:11-12`).

### Drivers (orchestrator-agnostic, one on-disk truth)

- **Sequential:** `build(model, steps, backend)` runs N `build_step`s with `run_start`/`run_end`/`error` bracketing (`sagi_build.py:175-187`).
- **PODA loop:** `build_loop` wires `AGLMCore(perceive, decide, act, agent_id="sagi.builder")` + `AutonomousLoop(interval)` (`sagi_build.py:190-213`): perceive = `{"built": len(modules)}`; decide = `"build"` if remaining else `"stop"`; act = `build_step`. Set `agent_id` to the individual's id. This is the self-healing runtime — per-stage exception isolation (`aglm/core.py:101-136`) and a circuit breaker after 5 consecutive failures with 120 s backoff (`aglm/cycle.py:91-98`).
- **Console loop:** `useSagi.runLoop → buildStep → POST /api/sagi`, gated by the `autonomous` + `sagi` toggles, tagged `backend:'console'` (`route.ts:64`).

Because all drivers read/write the same `manifest.json` + `.history/build.jsonl`, the fork stays orchestrator-agnostic; the constellation (`app/SagiVisual.tsx`) and shader (`app/SagiBackground.tsx`) watch disk, so any driver's growth renders identically.

---

## Individuality

Individuality is *data on an invariant method*, never a code fork. `SAGI_EXPANSION` declares itself "deliberately a SCAFFOLD to be expanded — the individuality layered on top defines who this particular sAGI becomes, so treat those traits, focus, voice, and constraints as your defining character" (`persona.ts:29`).

The composition rule is mechanical (`composePersonaPrompt`, `persona.ts:56-61`):

- no base → the individuality text alone;
- base, no individuality → the raw template;
- base **and** individuality → `` `${base}\n\n### Individuality (layered on top of the template)\n${ind}` `` — `SAGI_EXPANSION` first, then a delimited `### Individuality` section.

**What the individual layer controls** (the `individual` and `backend` fields of `identity.json`): identity/voice/name ("Refer to yourself as …"), focus domain, added constraints, backend/model choice, and its own package path. This becomes the system prompt behind every `ask_model` / `host.callModel` call, so *every propose/specify step is biased by who this sAGI is*.

**What the individual layer must NOT override** (the shared, inherited ABI): Savante epistemics (first-principles derivation, separate proven from conjecture, quantify uncertainty — near-verbatim in `SAGI_EXPANSION` from `SAVANTE`, `persona.ts:25`); the module-decomposition discipline (*purpose · interface · how it plugs into an agnostic core · how it advances self-building*); the self-healing "check after every automated step" guarantee (realized by `aglm/core.py:101-136` + `aglm/cycle.py:91-98`); and the host surface `{log, store, callModel, on, emit}` (`interface.md:11-12`).

**Identity / persistence boundaries.**
- *Identity* = `identity.json` (the individuality text + id). Two sAGIs sharing base `sagi` but with different `individual` blocks are different persons — exactly what `Persona.baseId`/`individual` persist (`persona.ts:9`). They diverge deterministically through the different module sequences their individuality causes them to propose.
- *Persistence* = the `SAGI_DIR` triple (`manifest.json` + `modules/` + `.history/`). An individual's memory of itself is precisely what it has grown into its own package. Cross-individual isolation is enforced by the traversal defense pinning every write inside `MODULES_DIR` (`route.ts:33-36,51-53`).
- *Belief state* = the per-run `BeliefSystem`, serializable via `to_dict`/`from_dict` (`aglm/beliefs.py:96-106`); the departure point persists it into the individual's package so identity survives restarts.

---

## The First Modules to Ship

> These three now ship as **tracked seed modules** in [`sagi/seed/`](seed/) — copy them into an individual's `$SAGI_DIR/modules/` to bootstrap (see [seed/README.md](seed/README.md)).

Exactly three, each real `activate(host)` code over the agnostic host surface (`interface.md:11-12`), each closing a gap the audit found (persona not layered; manifest desynced from disk; prose-only modules, no kernel). Together: **be thyself, keep thyself honest, grow thyself.**

### 1. individuality-core

The module that binds the individuality layer to the live mind — without it the fork still speaks as generic Savante.

- `id: individuality-core` · `deps: []`
- `inputs: identity.json` · `outputs: { prompt, whoami() }`
- `activate(host)`: read `identity.json` via `host.store`; compute `compose(SAGI_EXPANSION, identity.individual)` exactly per `persona.ts:56-61`; register it so every `host.callModel` call is prefixed with this individual's composed system prompt instead of the hard-coded `SAVANTE` (`sagi_build.py:61-66`). Expose `whoami() → {id, name, baseId, focus}`.
- *Advances self-building:* every subsequent propose/specify step is now spoken in this individual's voice, so what gets proposed is identity-shaped from step 1.

### 2. self-package-boundary

Owns the individual's persistence boundary and keeps the ledger honest — closing audit gaps #1 (manifest desync) and #6 (provenance thinner than `loop.md:10`).

- `id: self-package-boundary` · `deps: [individuality-core]`
- `inputs: SAGI_DIR` · `outputs: { syncManifest(), stamp() }`
- `activate(host)`: reconcile `manifest.json` against the `.md` files actually in `modules/` — re-derive `step` from filename order, re-add orphan files, rewrite `version = 0.0.{len}` (mirroring `sagi_build.py:164-166`). Stamp every manifest entry and `.history` line with the individual `id` and `backend` (today those reach only `build.jsonl`, `sagi_build.py:170`). Assert every write resolves inside `MODULES_DIR` (`route.ts:51-53`).
- *Advances self-building:* makes `len(modules)` — the value the whole loop counts on (`sagi_build.py:159`) — trustworthy across the CLI host, the console host, and crashes, and guarantees one individual never writes into another's package.

### 3. module-registry (the kernel)

The missing host that turns "generates files" into "grows a live system" — the capability `sagi/modules/01-proposed-next-module.md` explicitly asks for and the repo today lacks.

- `id: module-registry` · `deps: [self-package-boundary]`
- `inputs: modules/*.md, manifest.json` · `outputs: { register(mod), get(id), list(), activate_all() }`
- `activate(host)`: implement the concrete host surface `{log, store, callModel, on, emit}` (`interface.md:11-12`); scan this individual's `manifest.json`, import each `NN-slug` module, and call its `activate(host)` in topological `deps` order (refusing cycles), storing the returned handle keyed by `id`. Subscribe via `host.on("module.persisted", …)` so a freshly-persisted module is registered and activated without a restart.
- *Advances self-building:* once it exists, every future module this individual proposes becomes a live, composable capability in *its own* package rather than inert prose — the single change that makes the seed self-sustaining.

**Ship order: 1 → 2 → 3.** The persona must exist before it persists; the boundary must be sound before a kernel loads modules through it. After these three are in `manifest.json` and loading through the registry, `AutonomousLoop` (`aglm/cycle.py`) drives `build_step`, each result is a live registered module in the individual's package, and individuality steers what gets proposed next.

---

## How to Launch

Two hosts write the same `sagi/` store; the "green-light chooser" picks between them and the constellation/shader watch either identically (`SagiVisual.tsx`, `SagiBackground.tsx`, fed by `GET /api/sagi` + `GET /api/sagi/history`).

**A. Console green-light path (in-browser, watched live).**
In the codephreak console, flip the `autonomous` and `sagi` toggles on. `useSagi.runLoop` guards against double-start and loops `while !stop && autoRef && sagiRef && titles.length < maxSteps` (default 16), calling `buildStep` every ~600 ms (`hooks/useSagi.ts:54-63`). Each step calls `collectChat(persona + directive, ask)` then `POST /api/sagi`, which persists to `modules/NN-slug.md` + `manifest.json` + `.history/build.jsonl` tagged `backend:'console'` (`route.ts:43-70`). Turning off either toggle sets `stop.current = true` and halts the loop.

**B. Headless `sagi_build.py` (terminal green-light path).**
Point the builder at the individual's package and pick a backend:

```bash
# One individual, its own package, driven by the aGLM PODA loop:
SAGI_DIR=/path/to/<individual-id> \
python sagi_build.py --backend claude-cli --loop --steps 12

# Or single-shot sequential build against Ollama (default):
SAGI_DIR=/path/to/<individual-id> \
python sagi_build.py --backend ollama --steps 3
```

- `--backend ollama` (default) → local/remote Ollama at `OLLAMA_HOST` (`sagi_build.py:82-93`).
- `--backend claude-cli` → host `claude` CLI, subscription, no API key (`sagi_build.py:96-111`).
- `--backend claude-api` → Anthropic Messages API with `ANTHROPIC_API_KEY`, default `claude-opus-4-8` (`sagi_build.py:114-130`); if a claude backend is chosen while the model is still the Ollama default, the model swaps to `ANTHROPIC_MODEL` (`sagi_build.py:224-227`).
- `--loop` runs the aGLM `AutonomousLoop` (circuit-broken, exception-isolated); omit it for the sequential `build()`.

Whichever host the green-light chooses, both append to the *same* `manifest.json` + `.history/build.jsonl` inside the individual's `SAGI_DIR`, so console and terminal builds share one on-disk truth and the constellation, shader, and history view watch the individual grow in real time.

---

### Key files (all absolute)

- `sagi_build.py`
- `sagi/core/loop.md`
- `sagi/core/interface.md`
- `sagi/manifest.json`
- `sagi/modules/01-proposed-next-module.md`
- `codephreak-console/lib/persona.ts`
- `codephreak-console/app/api/sagi/route.ts`
- `codephreak-console/app/api/sagi/history/route.ts`
- `codephreak-console/hooks/useSagi.ts`
- `aglm/core.py`, `aglm/cycle.py`, `aglm/beliefs.py`
