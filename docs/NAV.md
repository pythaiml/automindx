# automindX — documentation index

Navigation for all project docs. (Start at the [README](../README.md).)

## Getting started
- [../README.md](../README.md) — overview, entrypoints, modules
- [INSTALL.md](INSTALL.md) — environment setup
- [CODEPHREAK_CONSOLE.md](CODEPHREAK_CONSOLE.md) — the flagship AI SDK console (run, features)

## Architecture
- [TECHNICAL.md](TECHNICAL.md) — module map, entrypoints, request flow
- [SERVICES.md](SERVICES.md) — the `services/` layer (memory · model · orchestrator · registry · secrets · self-audit)
- [AGLM.md](AGLM.md) — the migrated Autonomous General Learning Model package

## sAGI
- [../sagi/POINT_OF_DEPARTURE.md](../sagi/POINT_OF_DEPARTURE.md) — the canonical sAGI point of departure (seed + self-build protocol + individuality)
- [../sagi/seed/](../sagi/seed/) — the three point-of-departure seed modules (be thyself · keep thyself honest · grow thyself)
- [../sagi/README.md](../sagi/README.md) — the agnostic sAGI package

## Audit & fixes
- [CODEPHREAK_AUDIT.md](CODEPHREAK_AUDIT.md) — audit to Professor Codephreak standards
- [AUDIT.md](AUDIT.md) — the original code-audit pass (bug fixes)
- [../4096/CONTEXT.md](../4096/CONTEXT.md) — the 4096-token limitation, fix, and modern workflow

## Module notes (historical)
- [automind.md](automind.md) · [memory.md](memory.md) · [uiux.md](uiux.md) · [algm.md](algm.md)
- [DOCUMENTATION.md](DOCUMENTATION.md) — the original module walkthrough

## Ecosystem
- [github.com/GATERAGE](https://github.com/GATERAGE) · [aglm](https://github.com/GATERAGE/aglm) · [RAGE](https://github.com/GATERAGE/RAGE) · [mastermind](https://github.com/GATERAGE/mastermind)
- [github.com/aiterm](https://github.com/aiterm) — *augmented intelligence terminal* (terminal-AI arm of the ecosystem, [pythai.net](https://pythai.net)); realized here by the console's ⌘ Terminal
- [Professor Codephreak](https://github.com/Professor-Codephreak) · [mindx.pythai.net/automindx](https://mindx.pythai.net/automindx) · [rage.pythai.net](https://rage.pythai.net)

## Layout
```
automindX/
  README.md  LICENSE  Dockerfile  requirements*.txt
  services/      the decoupled service layer (+ tests/)
  aglm/          the Autonomous General Learning Model package
  sagi/          self-building sAGI package
  4096/          the 4096-token issue: code (context4096.py, chunk4096.py) + docs
  scripts/       automindx.install, run_codephreak_console.sh
  docs/          all documentation (this folder)
  codephreak-console/   the Next.js + AI SDK console
  automind.py codephreak.py memory.py llama_model.py ollama_codephreak.py …
```
