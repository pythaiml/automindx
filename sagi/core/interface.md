# The agnostic module interface

Every sagi module is self-contained and declares a tiny contract so the host can
wire it in without knowing the module's internals:

- `id`        — stable kebab-case name
- `purpose`   — one line
- `inputs`    — what it consumes (typed, host-agnostic)
- `outputs`   — what it produces
- `deps`      — other module ids it composes with (may be empty)
- `activate(host)` — receives an agnostic `host` (logger, store, model-call,
  event-bus) and returns a handle. No global state; no runtime assumptions.

A host implements `host = { log, store, callModel, on, emit }`. That is the only
surface a module may touch — which is what keeps sagi agnostic and portable
(server, CLI, or Tauri).
