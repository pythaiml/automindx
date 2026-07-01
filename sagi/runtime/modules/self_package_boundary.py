# Seed module 2 — self-package-boundary  ·  "do no harm"
# Owns the persistence boundary: reconciles the manifest against the modules that
# actually exist, stamps provenance, and — the do-no-harm guarantee — refuses any
# write that would escape this individual's package (so one sAGI can never corrupt
# another's).
import os
import re

MODULE_ID = "self-package-boundary"
DEPS = ["individuality-core"]
MOTTO = "do no harm"


def _slug_id(fname: str) -> str:
    stem = re.sub(r"^\d+-", "", fname[:-3] if fname.endswith(".md") else fname)
    return stem or "module"


def _title_from_file(host, fname: str) -> str:
    text = host.store.read(os.path.join("modules", fname)) or ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()[:90]
        if s:
            return s[:90]
    return _slug_id(fname)


def activate(host):
    def sync_manifest():
        files = host.store.list_modules()
        manifest = host.store.read_json("manifest.json", default={"name": "sagi", "modules": []}) or {}
        existing = {m.get("file"): m for m in manifest.get("modules", [])}
        modules = []
        for i, fn in enumerate(files, 1):
            prev = existing.get(fn, {})
            modules.append(stamp({
                "step": i,
                "id": prev.get("id") or _slug_id(fn),
                "title": prev.get("title") or _title_from_file(host, fn),
                "file": fn,
                "ts": prev.get("ts"),
            }))
        manifest["modules"] = modules
        manifest["version"] = f"0.0.{len(modules)}"
        # do no harm: this only writes inside SAGI_DIR (Store enforces confinement).
        host.store.write_json("manifest.json", manifest)
        return manifest

    def stamp(entry):
        entry.setdefault("id_owner", host.identity_id)
        entry.setdefault("backend", host.backend)
        return entry

    def guard_write(rel: str) -> bool:
        """True iff writing rel stays inside this individual's package."""
        try:
            host.store._safe(rel)
            return True
        except ValueError:
            host.log("harm_averted", path=rel)
            return False

    m = sync_manifest()
    host.log("module", step=2, id=MODULE_ID, reconciled=len(m["modules"]))
    return {"syncManifest": sync_manifest, "stamp": stamp, "guardWrite": guard_write}
