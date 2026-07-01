# tests/test_sagi_runtime.py — the three seed modules, running over the host surface.
#   python3 -m pytest tests/test_sagi_runtime.py -q
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagi.runtime import Host, boot
from sagi.runtime.modules import module_registry


def _seed_dir(with_modules=True):
    root = tempfile.mkdtemp()
    ident = {"id": "ada-sagi", "name": "Ada", "baseId": "sagi",
             "individual": "You value terse constructive proofs.", "backend": "ollama"}
    open(os.path.join(root, "identity.json"), "w").write(json.dumps(ident))
    os.makedirs(os.path.join(root, "modules"))
    if with_modules:
        open(os.path.join(root, "modules", "01-alpha.md"), "w").write("# Alpha\n\nfirst")
        open(os.path.join(root, "modules", "02-beta.md"), "w").write("# Beta\n\nsecond")
    return root


def test_individuality_core_speaks_as_the_individual():
    host, h = boot(_seed_dir())
    who = h["individuality-core"]["whoami"]()
    assert who["id"] == "ada-sagi" and who["name"] == "Ada" and who["baseId"] == "sagi"
    prompt = h["individuality-core"]["prompt"]
    assert "terse constructive proofs" in prompt          # individuality layered on
    assert "self-building" in prompt.lower()              # the sAGI base template
    assert host._prompt_prefix == prompt                   # every call_model now speaks as Ada


def test_self_package_boundary_reconciles_manifest():
    host, h = boot(_seed_dir())
    m = h["self-package-boundary"]["syncManifest"]()
    assert m["version"] == "0.0.2"
    assert [x["file"] for x in m["modules"]] == ["01-alpha.md", "02-beta.md"]
    assert m["modules"][0]["title"] == "Alpha"            # title read from the file
    assert m["modules"][0]["id_owner"] == "ada-sagi"      # provenance stamped
    # the manifest was actually written to disk
    on_disk = json.load(open(os.path.join(host.root, "manifest.json")))
    assert len(on_disk["modules"]) == 2


def test_do_no_harm_guard_refuses_escaping_writes():
    host, h = boot(_seed_dir())
    guard = h["self-package-boundary"]["guardWrite"]
    assert guard("modules/03-ok.md") is True
    assert guard("../../etc/evil") is False               # do no harm: never escape the package


def test_module_registry_lists_seeds_and_live_registers():
    host, h = boot(_seed_dir())
    reg = h["module-registry"]
    assert set(reg["list"]()) == {"individuality-core", "self-package-boundary", "module-registry"}
    host.emit("module.persisted", {"id": "03-grown", "file": "03-grown.md"})  # no restart
    assert "03-grown" in reg["list"]()
    assert reg["get"]("03-grown")["meta"]["persisted"] is True


def test_registry_activates_in_topological_deps_order():
    host = Host(_seed_dir(with_modules=False))
    order = []
    loaders = {
        "c": {"deps": ["b"], "activate": lambda h: order.append("c")},
        "a": {"deps": [], "activate": lambda h: order.append("a")},
        "b": {"deps": ["a"], "activate": lambda h: order.append("b")},
    }
    reg = module_registry.activate(host)
    reg["activate_all"](loaders)
    assert order == ["a", "b", "c"]                        # deps before dependents


def test_registry_refuses_dependency_cycles():
    host = Host(_seed_dir(with_modules=False))
    reg = module_registry.activate(host)
    cyclic = {"x": {"deps": ["y"], "activate": lambda h: None},
              "y": {"deps": ["x"], "activate": lambda h: None}}
    try:
        reg["activate_all"](cyclic)
        assert False, "expected a cycle error"
    except ValueError as e:
        assert "cycle" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
