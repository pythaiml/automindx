# tests/test_spawn.py — every sAGI can spawn another sAGI (sub | sov).
#   python3 -m pytest tests/test_spawn.py -q
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagi.runtime.spawn import spawn
from sagi.runtime.gitmind import GitMind


def _parent():
    r = tempfile.mkdtemp()
    os.makedirs(os.path.join(r, "modules"))
    open(os.path.join(r, "identity.json"), "w").write(json.dumps({"id": "root-sagi", "baseId": "sagi"}))
    return r


def test_spawn_sub_is_nested_and_subordinate():
    p = _parent()
    res = spawn(p, "Scout", mode="sub", individual="You explore.")
    assert res["mode"] == "sub" and res["sovereign"] is False
    assert res["dir"] == os.path.join(p, "children", "scout")
    ident = json.load(open(os.path.join(res["dir"], "identity.json")))
    assert ident["grown_from"] == "root-sagi" and ident["mode"] == "sub"
    assert ident["individual"] == "You explore."
    # child is a real, booted individual (has its own gitmind)
    assert os.path.isdir(os.path.join(res["dir"], ".gitmind"))


def test_spawn_sov_is_sovereign_peer():
    p = _parent()
    res = spawn(p, "Athena", mode="sov")
    assert res["mode"] == "sov" and res["sovereign"] is True
    assert res["dir"] == os.path.join(p, "sovereign", "athena")
    ident = json.load(open(os.path.join(res["dir"], "identity.json")))
    assert ident["sovereign"] is True and ident["grown_from"] == "root-sagi"


def test_lineage_and_parent_milestone_recorded():
    p = _parent()
    spawn(p, "Scout", mode="sub")
    spawn(p, "Athena", mode="sov")
    lineage = json.load(open(os.path.join(p, "lineage.json")))
    ids = {c["id"]: c["mode"] for c in lineage["children"]}
    assert ids == {"scout": "sub", "athena": "sov"}
    # each spawn is a GLOBAL gitmind milestone on the parent
    spawns = [c for c in GitMind(p).global_log() if c["moment"].get("event") == "spawn"]
    assert {c["moment"]["child"] for c in spawns} == {"scout", "athena"}
    # and a .history event
    hist = open(os.path.join(p, ".history", "build.jsonl")).read()
    assert hist.count('"event": "spawn"') == 2


def test_spawn_is_recursive_child_spawns_grandchild():
    p = _parent()
    child = spawn(p, "Scout", mode="sub")
    grand = spawn(child["dir"], "Recon", mode="sub")     # a sAGI spawning a sAGI
    assert grand["parent"] == "scout"
    assert os.path.isdir(os.path.join(child["dir"], "children", "recon"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
