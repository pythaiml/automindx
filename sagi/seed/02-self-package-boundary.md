# self-package-boundary

*do no harm* — owns the individual's persistence boundary and keeps the
ledger truthful (closes the manifest-desync and thin-provenance gaps).

- **id:** `self-package-boundary`
- **deps:** `[individuality-core]`
- **inputs:** `SAGI_DIR`
- **outputs:** `{ syncManifest(), stamp() }`

## activate(host)

Reconcile `manifest.json` against the `.md` files actually present in `modules/`:
re-derive `step` from filename order, re-add orphan files, and rewrite
`version = 0.0.{len}` (mirroring `sagi_build.py`). Stamp every manifest entry and
`.history` line with the individual `id` and `backend` (today those reach only
`build.jsonl`). Assert every write resolves inside `MODULES_DIR`
(as `app/api/sagi/route.ts` already enforces).

## Advances self-building

Makes `len(modules)` — the value the whole loop counts on — trustworthy across the
CLI host, the console host, and crashes, and guarantees one individual never writes
into another's package.
