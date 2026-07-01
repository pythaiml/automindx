# aGLM Code Release (2026-05-14)

This file lives next to the modern `aglm/` Python package and explains
the relationship between the new code and the existing historical
research files at the repo root.

## Two layers in this repo

1. **Historical research** (repo root) — `MASTERMIND.py`, `aglm.py`,
   `automind.py`, `bdi.py`, `reasoning.py`, `socratic.py`, etc. These
   are the Professor-Codephreak / easyAGI foundation that aGLM grew
   out of. The repo's README says these files are "BROKEN" and useful
   as a reference. They are preserved unchanged.

2. **Modern agnostic distribution** (`aglm/` Python package) — added
   2026-05-14. A clean, installable, Apache-2.0 Python package that
   distills the autonomous-learning-loop pattern from
   [mindX](https://github.com/agenticplace) into reusable primitives:
   - `AGLMCore` — Perceive-Orient-Decide-Act cycle
   - `BeliefSystem` — claim + confidence + source attribution
   - `AutonomousLoop` — periodic runner with circuit breaker

## Why both

The historical files document the *philosophy* — the multi-module
reasoning architecture (Mastermind, socratic, non-monotonic, etc.).
The modern `aglm/` package captures the *operational distillation*
that emerged from running those ideas in mindX production for a year.

If you want to study the architecture: read the root files + the
[ragepaper.md companion in the RAGE repo](https://github.com/GATERAGE/RAGE/blob/main/ragepaper.md).
If you want to *install and use* the loop: `pip install .`.

## Spec

`docs/aglm_as_a_service.md` — the canonical service contract for
what the modern aGLM offers as a primitive in a multi-agent system.
