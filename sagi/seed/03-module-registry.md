# module-registry (the kernel)

*grow thyself* — the missing host that turns "generates files" into "grows a live
system". This is the kernel `modules/01-proposed-next-module.md` asks for and the
repo today lacks.

- **id:** `module-registry`
- **deps:** `[self-package-boundary]`
- **inputs:** `modules/*.md`, `manifest.json`
- **outputs:** `{ register(mod), get(id), list(), activate_all() }`

## activate(host)

Implement the concrete host surface `{ log, store, callModel, on, emit }`
(`core/interface.md`); scan this individual's `manifest.json`, import each
`NN-slug` module, and call its `activate(host)` in topological `deps` order
(refusing cycles), storing the returned handle keyed by `id`. Subscribe via
`host.on("module.persisted", …)` so a freshly-persisted module is registered and
activated **without a restart**.

## Advances self-building

Once it exists, every future module this individual proposes becomes a live,
composable capability in its own package rather than inert prose — the single
change that makes the seed self-sustaining. After 1 → 2 → 3 are registered,
`AutonomousLoop` (`aglm/cycle.py`) drives `build_step`, each result is a live
registered module, and individuality steers what gets proposed next.
