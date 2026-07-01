# sagi — self-building, agnostic, modular scientific general intelligence

> **Start here:** [POINT_OF_DEPARTURE.md](POINT_OF_DEPARTURE.md) — the canonical
> fork-point for an individual sAGI (seed structure, self-build protocol, and how
> individuality layers on the Savante / sAGI templates).

`sagi` is an **agnostic, modular** scaffold that any project can include. It is
grown from the **Savante** persona inside automindX and built one module at a
time by an **autonomous loop** driving a chosen model. Each build step proposes
and specifies the next module; modules are self-contained and pluggable into an
agnostic core, so `sagi` can be embedded in a server, a CLI, or a desktop app
(including a **Tauri** app).

## Principles

- **Agnostic** — no hard dependency on a specific model, runtime, or UI. A module
  declares a small interface; the host wires it in.
- **Modular** — one capability per module; modules compose. Nothing global.
- **Self-building** — the autonomous loop (see `core/loop`) asks the chosen model
  to specify the next module; approved specs become modules under `modules/`.
- **Includable anywhere** — drop `sagi/` into any project. For desktop, wrap it
  in Tauri (`src-tauri/`) with the core exposed over the Tauri command bridge.

## Layout

```
sagi/
  README.md            ← this file
  manifest.json        ← the growing architecture (modules + provenance)
  core/
    loop.md            ← the self-building autonomous loop contract
    interface.md       ← the agnostic module interface every module implements
  modules/             ← grown modules land here — prose specs (*.md) AND executable
                         activate(host) modules (NN-slug.py), loaded live by the kernel
  runtime/             ← the executable host: boot, host surface, seed modules, gitmind,
                         governance, spawn, savepoint, rage_sync
  tauri.md             ← how to ship sagi as a Tauri desktop app
```

## How it is built

In the automindX console, turn **sAGI** on (Preferences). With **Autonomous on**
the loop builds continuously with the chosen model; with **Autonomous off** it
builds one module per input → response. Each step is appended to `manifest.json`
and can be exported. Turning **Autonomous off** stops the continuous loop.

> sagi is agnostic by construction: the console is only one host. The same
> contract is driven **headless** by `sagi_build.py` (which can run the build via
> the migrated `aglm.AutonomousLoop` PODA cycle):
>
> ```bash
> python3 sagi_build.py --steps 3                 # build 3 modules
> python3 sagi_build.py --loop --steps 5          # via the aGLM autonomous loop
> ```
>
> Both hosts write the same `modules/` + `manifest.json`, proving sagi is
> orchestrator-agnostic.

## Memory grows as a tree — gitmind

Each individual sAGI grows an internal, content-addressed memory tree
(`sagi/runtime/gitmind.py`): every persisted moment snapshots `modules/` into
blobs → a tree → a commit chained to its parent, under `<SAGI_DIR>/.gitmind/`. It
is the in-package, server-less analogue of a self-hosted forge (**Forgejo**) and of
mindX's **`/mindx/gitmind`** — so memory can be revisited from any moment in
`.history`. It ports **[Professor-Codephreak/gitmind](https://github.com/Professor-Codephreak/gitmind)**
and adds two commit tiers:

- **local** — the per-timestamp timeline (**vertical scaling**: this individual
  deepening over time); the default, deduped on no-op.
- **global** — reserved for expansion & massive-upgrade moves (**horizontal
  scaling**), walkable on its own `GLOBAL` chain via `global_log()`.

```python
from sagi.runtime import boot
host, _ = boot(SAGI_DIR)                 # host.memory is the gitmind tree
host.memory.log()                        # commits, newest first
host.memory.tree_at_moment(ts)           # the exact memory state at a .history timestamp
```

`boot()` commits a snapshot on every `module.persisted` event, so the memory tree
and `.history/build.jsonl` stay aligned by timestamp. The object store is runtime
state (git-ignored).

## Sandbox & governance

Every sAGI builds inside its **own `SAGI_DIR`** — a sandbox. The `self-package-boundary`
seed module (*do no harm*) refuses any write that escapes the package, so a build is
sandboxed by construction:

```bash
SAGI_DIR=/path/to/box python3 sagi_build.py --backend ollama --steps 3   # writes only inside the box
```

**Edit governance** (`sagi/runtime/governance.py`) layers on top of spawning:

- a sAGI may edit **itself** and its **subordinate (sub)** descendants;
- a **sovereign (sov)** sAGI is self-governing — only it may edit itself and its own
  subtree; a parent may **not** reach across the sovereignty boundary.

`can_edit(actor, target)` / `assert_can_edit(...)` / `governed_write(...)` enforce it.

## The living runtime — 70 grown modules

Beyond the three seed modules, sAGI has grown into a **live, self-composing runtime of 70
executable modules** (`sagi/modules/NN-slug.py`). Each declares the seed contract —
`MODULE_ID`, `DEPS: list[str]`, `MOTTO`, and `activate(host) -> {callables}` — and is imported and
activated in dependency (toposort) order by the **`module-loader`** kernel at boot. Booting the
package brings the whole stack live:

```python
from sagi.runtime import boot
host, handles = boot("./sagi")             # 73 modules live (3 seed + 70 grown), ~0.4s
host.registry["get"]("evaluator")["handle"]["evaluate"]()   # live scorecard
host.registry["get"]("introspection")["handle"]["map"]()    # the live self-map
```

Every module is **offline-safe** (degrades gracefully with no model backend, RAGE, or network),
**Store-confined** (writes never escape `SAGI_DIR`), and continuously **verified** by
`module-verifier` (0 failures). Growth is organised in tiers, each building on the last:

- **Tier 1 — make growth real & directed** (`01`–`10`): module-loader, epistemic-calibration
  (separate proof from conjecture), module-verifier, memory-recall, goal-graph, tool-registry,
  evaluator, reflection-journal, curriculum, swarm-orchestrator.
- **Tier 2 — autonomous, robust, aligned, connected** (`11`–`20`): build-driver (the self-driving
  build loop), rollback-recovery, policy-guard, inference-router, episodic-consolidation,
  introspection, telemetry, wire-gateway, benchmark-suite, lineage-federation.
- **Tier 3 — self-improvement & isolation** (`21`–`30`): **self-prompt-auditor** (a dynamic,
  self-adjusting Claude system prompt built from live state + a read-only map of the automindX repo),
  **sagi-environment** (stable parent spawns a volatile child to experiment; verified deltas promote
  back), skill-library, outcome-learning, knowledge-grounding, hypothesis-engine, security-hardening,
  persistence-deploy, human-interface, **meta-kernel-evolution** (gated safe self-modification).
- **Tier 4 — society, economy, continuity** (`31`–`40`): scheduler, resource-economy,
  negotiation-protocol, consensus-ledger, anomaly-watch, explainability, multimodal-io,
  self-documentation, goal-lifecycle, continuity.
- **Tier 5 — depth & rigor** (`41`–`50`): planning-search, constraint-solver, test-synthesis,
  dependency-audit, refactor-engine, provenance-tracker, versioning, config-manager, event-sourcing,
  conflict-resolver.
- **Tier 6 — interface & ecosystem** (`51`–`60`): api-schema, plugin-loader, dataset-manager,
  model-registry, prompt-library, conversation-memory, notification, metrics-dashboard,
  access-control, federation-sync.
- **Tier 7 — autonomy & governance** (`61`–`70`): governance-council, ethics-monitor,
  self-test-harness, rollback-orchestrator, capability-discovery, goal-negotiation,
  resource-scheduler, audit-log, disaster-recovery, sovereignty-manager.

**Guiding principle — automindX stays stable; sAGI improves dynamically.** sAGI's writes are
confined to its own package, so the surrounding repo is never mutated (filesystem *awareness* above
the boundary is read-only). A stable parent spawns volatile children to experiment; verified state
promotes back under governance and is preserved via gitmind + savepoints; kernel self-edits pass
scan → verify → benchmark → rollback gates. Boot the runtime and call
`host.registry["get"]("self-documentation")["handle"]["generate"]()` for an always-current README of
whatever the individual has become.

> The grown `.py` modules and per-individual runtime state (`.history/`, `.gitmind/`, `savepoints/`,
> `goal.txt`, and the various `*.json` a module persists) are git-ignored runtime; the seed modules,
> the shared runtime, and this documentation are version-controlled.

## Console — the sAGI terminal UI

The [`codephreak-console`](../codephreak-console) (Next.js) drives sAGI in-browser. The top bar has
an **sAGI model selector** (Claude · Local · API) that chooses the build backend
(`claude-cli` / `ollama` / `claude-api`), and a **maximize/shrink** control that toggles the main
screen between a centered medium view and a wide layout. Both persist across sessions.
