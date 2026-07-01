# tests/test_gitmind.py — the internal git-like memory tree.
#   python3 -m pytest tests/test_gitmind.py -q
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagi.runtime.gitmind import GitMind


def _root():
    r = tempfile.mkdtemp()
    os.makedirs(os.path.join(r, "modules"))
    return r


def _write(root, fn, content):
    open(os.path.join(root, "modules", fn), "w").write(content)


def test_commit_chain_and_dedup():
    r = _root()
    _write(r, "01-a.md", "# A\nv1")
    gm = GitMind(r)
    c1 = gm.commit(moment={"event": "boot"}, ts=100)
    assert gm.head() == c1
    # no change -> no new commit
    assert gm.commit(ts=101) == c1
    # change a module -> new commit chained to c1
    _write(r, "01-a.md", "# A\nv2")
    c2 = gm.commit(moment={"event": "module.persisted", "id": "01-a"}, ts=200)
    assert c2 != c1
    log = gm.log()
    assert [x["commit"] for x in log] == [c2, c1]           # newest first
    assert gm.read_commit(c2)["parent"] == c1


def test_tree_reconstructs_memory_at_a_commit():
    r = _root()
    _write(r, "01-a.md", "# A\nv1")
    gm = GitMind(r)
    c1 = gm.commit(ts=100)
    _write(r, "01-a.md", "# A\nv2")
    _write(r, "02-b.md", "# B\nnew")
    c2 = gm.commit(ts=200)
    t1 = gm.tree(c1)
    t2 = gm.tree(c2)
    assert t1 == {"01-a.md": "# A\nv1"}                     # past memory intact
    assert t2["01-a.md"] == "# A\nv2" and t2["02-b.md"] == "# B\nnew"


def test_access_memory_from_a_history_moment():
    r = _root()
    _write(r, "01-a.md", "v1")
    gm = GitMind(r)
    gm.commit(ts=100)
    _write(r, "01-a.md", "v2")
    gm.commit(ts=300)
    # a .history moment at ts=150 sees the v1 memory (latest commit <= 150)
    assert gm.at_moment(150)["ts"] == 100
    assert gm.tree_at_moment(150) == {"01-a.md": "v1"}
    assert gm.tree_at_moment(999) == {"01-a.md": "v2"}


def test_local_vs_global_commit_tiers():
    r = _root()
    _write(r, "01-a.md", "v1")
    gm = GitMind(r)
    g0 = gm.global_commit(moment={"event": "genesis"}, ts=100)   # milestone (horizontal)
    assert gm.global_head() == g0
    _write(r, "01-a.md", "v2")
    l1 = gm.commit(ts=200)                                        # local timeline (vertical)
    _write(r, "02-b.md", "big")
    g1 = gm.global_commit(moment={"event": "expansion"}, ts=300)  # massive-upgrade milestone
    assert gm.global_head() == g1
    # HEAD walks everything; GLOBAL walks only milestones.
    assert [c["commit"] for c in gm.log()] == [g1, l1, g0]
    assert [c["commit"] for c in gm.global_log()] == [g1, g0]     # global chain skips locals
    assert [c["commit"] for c in gm.log(scope="global")] == [g1, g0]
    assert gm.read_commit(g1)["global_parent"] == g0


def test_boot_grows_memory_and_snapshots_persisted_moments():
    import json
    from sagi.runtime import boot
    r = _root()
    open(os.path.join(r, "identity.json"), "w").write(json.dumps({"id": "x", "baseId": "sagi"}))
    _write(r, "01-a.md", "# A")
    host, _ = boot(r)
    assert host.memory is not None and host.memory.head() is not None   # seed snapshot
    n0 = len(host.memory.log())
    _write(r, "02-b.md", "# B")
    host.emit("module.persisted", {"id": "02-b", "file": "02-b.md"})    # auto-commit
    assert len(host.memory.log()) == n0 + 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
