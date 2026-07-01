# automindX — audited to Professor Codephreak standards

This audit was produced by running the **actual Professor Codephreak prompt**
(`automind.py` `DEFAULT_SYSTEM_PROMPT`) over the automindX architecture, then
applying the fixes that fit a local-first tool. Codephreak's standard: step-by-step
logic, production-ready and secure code, resource-optimized, modular.

## Codephreak's findings (verbatim, most critical first)

1. **`/api/install-ollama` executes shell code** → sanitize, rate-limit, sandbox.
2. **`/api/chat` and `/api/models` proxy Ollama without auth** → add token validation, CORS/CSRF.
3. **Feedback and persona data stored unencrypted** → encrypt at rest, add ACLs.
4. **Node spawns external processes without timeout/resource caps** → strict timeout, capped buffers, kill.
5. **Global state in `codephreak.py` / aGLM can race / leak** → immutable flows, async queues, cleanup hooks.

## What was fixed

- **(1 & 4) The shell-executing install route is now hardened.**
  - Runs the **official** script only (`https://ollama.com/install.sh`), and only
    after an explicit confirm token.
  - **Killed after a 5-minute timeout** (`SIGKILL`); **output capped** at 64 KB.
  - **Opt-out** via `CODEPHREAK_DISABLE_INSTALL=1` for shared/hardened hosts.
- **Correctness (self-building loop):** fixed a stale-closure bug where the
  autonomous sAGI loop didn't pass previously-built modules to the next step —
  now context accumulates so it never repeats a module.
- **Resource cap:** the sAGI autonomous loop is bounded (`SAGI_MAX_STEPS`) and
  stops immediately when Autonomous is turned off.

## Deferred — with rationale (scope: local-first tool)

- **(2) Endpoint auth / CORS:** automindX binds to `localhost` for a single
  operator; the routes proxy a local Ollama. Adding JWT/OAuth is appropriate when
  exposing it on a network — documented as the hardening step for that case, not
  forced on the local default.
- **(3) Encryption at rest:** feedback/persona are the operator's own data in
  their own `./memory` on their own machine. Encryption belongs with a
  multi-tenant deployment, not the local default.
- **(5) Global state:** `codephreak.py` uses one module-level engine per process
  and appends feedback line-by-line (atomic per line). Fine for the single-process
  server; the async-queue refactor is the right move only under real concurrency.

## Codephreak standards checklist

- **Step-by-step logic** — the PODA loop and the sAGI build loop are explicit,
  one observable step at a time.
- **Production-ready & secure** — official-source-only installs, confirm-gated,
  timed-out, capped, opt-out; graceful (never-crashing) model-access handling.
- **Resource-optimized** — bounded loops, no heavy deps in the UIs, model runs in
  the Ollama daemon (not in-process).
- **Modular** — agnostic `sagi/` interface, `aglm/` package, pluggable personas,
  isolated API routes.
