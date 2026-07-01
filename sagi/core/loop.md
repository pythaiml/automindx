# The self-building autonomous loop

One step = propose → specify → (approve) → append.

1. **Propose** — ask the chosen model for the NEXT single module given the
   modules built so far.
2. **Specify** — the model returns a Title + spec (purpose · interface · how it
   plugs into the agnostic core · how it advances self-building).
3. **Append** — the module is written under `modules/` and recorded in
   `manifest.json` with provenance (model, step, timestamp).

Modes (set in the automindX console):
- **Autonomous on** → the loop runs continuously (capped) with the chosen model.
- **Autonomous off** → one step per operator input → response.
- Turning **Autonomous off** halts a running loop.

The loop is orchestrator-agnostic: the console drives it in-browser; the same
contract is driven by `automind_aglm.py` (the aGLM PODA cycle) headless.
