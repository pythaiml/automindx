# sagi/modules/37-multimodal-io.py — tier-4 · "perceive broadly"
#
# GAP CLOSED: every input/output so far is plain text. This adds typed modality adapters so the
# individual can ingest and emit structured forms (json, csv/table, and image/binary-by-reference)
# uniformly, each normalised to a text record with provenance so it flows into memory/recall. It does
# NOT pull in heavy media libraries (stdlib only): images/binaries are handled as referenced
# artifacts (path/url + metadata), not decoded pixels — awareness without new dependencies.
#
# GROUNDED REUSE: tool-registry (adapters registered as tools), knowledge-grounding.ingest
# (provenance for an ingested artifact), host.store (Store-confined). STDLIB only (json, csv, io).
from __future__ import annotations

import csv
import io
import json

MODULE_ID = "multimodal-io"
DEPS = ["tool-registry", "knowledge-grounding"]
MOTTO = "perceive broadly"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    modalities = {}   # name -> {"decode": data->text, "encode": record->str}

    def register_modality(name, decode, encode=None):
        """Register a typed adapter: decode(data)->normalised text/record; encode(record)->str."""
        modalities[name] = {"decode": decode, "encode": encode or (lambda r: str(r))}
        return name

    def ingest(modality, data, ref=None):
        """Normalise `data` of the given modality to a text record, with provenance via knowledge-grounding."""
        m = modalities.get(modality)
        if not m:
            return {"ok": False, "error": f"unknown modality {modality}"}
        try:
            text = m["decode"](data)
        except Exception as e:
            return {"ok": False, "error": f"decode failed: {str(e)[:120]}"}
        kg = _sib(host, "knowledge-grounding")
        if kg and ref:
            kg["ingest"](ref, text if isinstance(text, str) else json.dumps(text))
        return {"ok": True, "modality": modality, "record": text}

    def emit(modality, record):
        """Encode a record back into a modality's wire form."""
        m = modalities.get(modality)
        if not m:
            return {"ok": False, "error": f"unknown modality {modality}"}
        try:
            return {"ok": True, "encoded": m["encode"](record)}
        except Exception as e:
            return {"ok": False, "error": str(e)[:120]}

    # --- grounded built-in modalities (stdlib) ---
    register_modality("text", lambda d: str(d), lambda r: str(r))
    register_modality("json", lambda d: json.loads(d) if isinstance(d, str) else d,
                      lambda r: json.dumps(r, indent=2))
    def _csv_decode(d):
        rows = list(csv.reader(io.StringIO(d if isinstance(d, str) else "")))
        return {"rows": rows, "n": len(rows)}
    register_modality("table", _csv_decode,
                      lambda r: "\n".join(",".join(map(str, row)) for row in (r.get("rows") if isinstance(r, dict) else r)))
    # image/binary handled by reference (metadata), never decoded pixels — no new deps
    register_modality("artifact", lambda d: {"ref": d, "kind": "binary/reference", "note": "not decoded (stdlib-only)"},
                      lambda r: str((r or {}).get("ref", r)))

    host.log("module", step="tier4-37", id=MODULE_ID, modalities=len(modalities))
    return {"register_modality": register_modality, "ingest": ingest, "emit": emit}
