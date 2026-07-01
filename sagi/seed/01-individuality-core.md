# individuality-core

*be thyself* — binds the individuality layer to the live mind. Without it the fork
still speaks as generic Savante.

- **id:** `individuality-core`
- **deps:** `[]`
- **inputs:** `identity.json`
- **outputs:** `{ prompt, whoami() }`

## activate(host)

Read `identity.json` via `host.store`; compute
`compose(SAGI_EXPANSION, identity.individual)` exactly per `composePersonaPrompt`
(`codephreak-console/lib/persona.ts`). Register it so every `host.callModel` call is
prefixed with **this** individual's composed system prompt instead of the
hard-coded `SAVANTE` constant in `sagi_build.py`. Expose
`whoami() → { id, name, baseId, focus }`.

## Advances self-building

Every subsequent propose → specify step is now spoken in this individual's voice,
so what gets proposed is identity-shaped from step 1 — the fork becomes a *person*,
not generic Savante.
