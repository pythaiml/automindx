# sagi/modules/53-dataset-manager.py — tier-6 · "curate thy data"
#
# GAP CLOSED: knowledge-grounding ingests single references, but there's no first-class notion of a
# DATASET — a named, provenanced collection the individual can train/evaluate against. This manages
# datasets: register a named collection with provenance, fetch it, and list them. Items persist under
# datasets/ (Store-confined) with a manifest; each carries where it came from.
#
# GROUNDED REUSE: knowledge-grounding.ingest (per-item provenance), host.store, host.memory.commit.
from __future__ import annotations

import json
import os

MODULE_ID = "dataset-manager"
DEPS = ["knowledge-grounding", "memory-recall"]
MOTTO = "curate thy data"

_DIR = "datasets"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _index():
        return host.store.read_json(os.path.join(_DIR, "index.json"), default={}) or {}

    def register(name, items, provenance="operator"):
        """Register a named dataset (list of records) with provenance. Persisted under datasets/."""
        rel = os.path.join(_DIR, f"{name}.json")
        host.store.write(rel, json.dumps({"provenance": provenance, "items": items}, indent=2))
        idx = _index()
        idx[name] = {"file": rel, "n": len(items), "provenance": provenance}
        host.store.write_json(os.path.join(_DIR, "index.json"), idx)
        # record provenance into knowledge memory
        kg = _sib(host, "knowledge-grounding")
        if kg:
            kg["ingest"](f"dataset://{name}", f"dataset {name}: {len(items)} items from {provenance}")
        host.log("dataset_register", name=name, n=len(items))
        return {"name": name, "n": len(items)}

    def get(name):
        """Fetch a dataset by name: {provenance, items}."""
        raw = host.store.read(os.path.join(_DIR, f"{name}.json"))
        if raw is None:
            return {"error": f"no dataset {name}"}
        try:
            return json.loads(raw)
        except Exception:
            return {"error": "corrupt dataset"}

    def list_datasets():
        return _index()

    host.log("module", step="tier6-53", id=MODULE_ID)
    return {"register": register, "get": get, "list_datasets": list_datasets}
