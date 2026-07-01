# aGLM in automindX

The **advanced aGLM** — the modern, agnostic *Autonomous General Learning Model*
package — has been migrated into automindX from
[github.com/GATERAGE/aglm](https://github.com/GATERAGE/aglm). It gives automindX a
clean, dependency-free autonomous-learning core, and it is wired to Professor
Codephreak's realtime feedback so the environment can **refine its own persona**.

> Professor Codephreak has been busy evolving. What began as a single Gradio
> chat around a 4096-token Llama-2 model is now: an instant-loading Ollama UI, a
> cutting-edge AI SDK console, a self-improving feedback engine, and — with this
> migration — an autonomous PODA learning loop. History:
> [github.com/Professor-Codephreak](https://github.com/Professor-Codephreak).

## What was migrated

The `aglm/` package (Apache-2.0), distilled from the autonomous-learning-loop
pattern in [mindX](https://github.com/agenticplace):

| Module | Class | Responsibility |
|---|---|---|
| `aglm/core.py` | `AGLMCore` | Perceive → Orient → Decide → Act (PODA) cycle |
| `aglm/beliefs.py` | `BeliefSystem` | claim + confidence + source attribution, with revision |
| `aglm/cycle.py` | `AutonomousLoop` | periodic runner with circuit breaker + backoff |

### Audit / modernization performed during migration

- **Name-collision resolved.** The repo's legacy `aglm.py` (an in-process
  transformers Llama loader) was renamed to **`llama_model.py`** so the name
  `aglm` now refers to the Autonomous General Learning Model package. Imports in
  `uiux.py` and `hfUIUX.py` were updated accordingly.
- The package is modern already (async, typed, SPDX-headed, exception-isolated);
  it is migrated intact and verified to import and run in automindX.

## How automindX uses it — `automind_aglm.py`

`automind_aglm.py` bridges aGLM to the self-improving engine (`codephreak.py`):

```
PERCEIVE  read 👍/👎 feedback stats from codephreak.py
ORIENT    AGLMCore turns them into beliefs
DECIDE    if the dislike ratio is high, decide to refine the persona
ACT       fold codephreak.py's learned directives into the persona
```

```bash
python3 automind_aglm.py          # one PODA cycle (demo)
python3 automind_aglm.py --loop   # autonomous loop (self-refines periodically)
```

Verified: with a majority-dislike feedback history the agent decides
`refine_persona` and folds in learned directives (e.g. *"prefer shorter answers"*,
*"always include a code block when code is requested"*), recording the outcome as
beliefs.

## The GATERAGE ecosystem

> **RAGE remembers · aGLM decides · MASTERMIND orchestrates.**

- **[github.com/GATERAGE](https://github.com/GATERAGE)** — the organization
- **[GATERAGE/aglm](https://github.com/GATERAGE/aglm)** — this package's canonical home + the AGLM consoles
- **[GATERAGE/RAGE](https://github.com/GATERAGE/RAGE)** — retrieval substrate (memory + grounding)
- **[GATERAGE/mastermind](https://github.com/GATERAGE/mastermind)** — strategic orchestrator (directive layer)
- **[Professor Codephreak](https://github.com/Professor-Codephreak)** — the origin and the evolving author
