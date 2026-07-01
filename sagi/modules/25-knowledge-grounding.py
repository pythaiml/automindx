# sagi/modules/25-knowledge-grounding.py — tier-3 · "ground thyself"
#
# GAP CLOSED: recall today only surfaces what sAGI generated about itself — it has no way to ingest
# EXTERNAL references with provenance, so reasoning can't be grounded in real sources. This ingests
# a reference (provided text, or fetched via a tool-registry tool) into the individual's memory with
# provenance, gated by policy-guard, and lets recall cite sources. Ingested notes live under
# knowledge/ (Store-confined) and are gitmind-committed so memory-recall finds them.
#
# GROUNDED REUSE: policy-guard.check (gate ingest), tool-registry.invoke (optional fetch),
# memory-recall.recall (cite), host.memory.commit (make the source part of memory).
from __future__ import annotations

import hashlib
import os
import time

MODULE_ID = "knowledge-grounding"
DEPS = ["memory-recall", "inference-router", "tool-registry", "policy-guard"]
MOTTO = "ground thyself"

_DIR = "knowledge"


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    def _sources():
        return host.store.read_json("sources.json", default=[]) or []

    def ingest(ref, text=None):
        """Ingest an external reference (with text, or fetched via a 'fetch' tool). Policy-gated."""
        guard = _sib(host, "policy-guard")
        if guard:
            d = guard["check"]("ingest", {"ref": ref})
            if not d.get("allow", True):
                return {"ok": False, "refused": d.get("reason")}
        if text is None:
            tr = _sib(host, "tool-registry")
            if tr and any(t["name"] == "fetch" for t in tr["list_tools"]()):
                r = tr["invoke"]("fetch", {"url": ref})
                text = (r.get("result") if isinstance(r, dict) else None) or ""
            else:
                text = ""                                   # offline: provenance recorded without body
        key = hashlib.sha256((ref + "\0" + (text or "")).encode()).hexdigest()[:16]
        rel = os.path.join(_DIR, f"{key}.md")
        body = f"# source: {ref}\n\n{text}\n"
        host.store.write(rel, body)                          # Store-confined
        srcs = _sources()
        if not any(s.get("key") == key for s in srcs):
            srcs.append({"key": key, "ref": ref, "file": rel, "chars": len(text or ""), "ts": int(time.time())})
            host.store.write_json("sources.json", srcs)
        if getattr(host, "memory", None) is not None:
            try:
                host.memory.commit(moment={"event": "ingest", "ref": ref, "file": rel}, message=f"ingest {ref}")
            except Exception:
                pass
        host.log("ingest", ref=ref, key=key, chars=len(text or ""))
        return {"ok": True, "key": key, "file": rel}

    def cite(query, k=5):
        """Recall grounded in ingested sources — returns hits with their provenance ref when known."""
        recall = _sib(host, "memory-recall")
        hits = recall["recall"](query, k) if recall else []
        by_file = {s["file"]: s["ref"] for s in _sources()}
        for h in hits:
            f = h.get("file")
            if f in by_file:
                h["source"] = by_file[f]
        return hits

    def sources():
        return _sources()

    host.log("module", step="tier3-25", id=MODULE_ID, sources=len(_sources()))
    return {"ingest": ingest, "cite": cite, "sources": sources}
