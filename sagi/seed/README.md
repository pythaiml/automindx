# sagi seed — the point-of-departure modules

The three canonical modules every individual sAGI ships first, from
[../POINT_OF_DEPARTURE.md](../POINT_OF_DEPARTURE.md). Each is real `activate(host)`
work over the agnostic host surface (`core/interface.md`), and each closes a gap
the audit found. Together: **be thyself, do no harm, grow thyself.**

**Ship order 1 → 2 → 3** — the persona must exist before it persists; the boundary
must be sound before a kernel loads modules through it.

1. [individuality-core](01-individuality-core.md) — *be thyself*
2. [self-package-boundary](02-self-package-boundary.md) — *do no harm*
3. [module-registry](03-module-registry.md) — *grow thyself* (the kernel)

## How an individual bootstraps from the seed

An individual sAGI copies these three into its own `$SAGI_DIR/modules/` as
`01`–`03` and reconciles its `manifest.json` (module 2 does exactly this). After
that, the registry (module 3) loads every future module the individual grows as a
live, composable capability — the change that makes the seed self-sustaining.

```bash
mkdir -p "$SAGI_DIR/modules"
cp sagi/seed/0*.md "$SAGI_DIR/modules/"
python3 sagi_build.py --backend claude-cli --loop --steps 12   # grow from the seed
```

These seed specs are version-controlled here; the per-individual `modules/` a build
grows into is runtime state (git-ignored).
