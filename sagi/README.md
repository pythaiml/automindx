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
