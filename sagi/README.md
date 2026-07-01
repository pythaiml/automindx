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
  modules/             ← generated modules land here (one file per module)
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
