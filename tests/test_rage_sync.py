# tests/test_rage_sync.py — saving gitmind trees into RAGE (SQLite fallback here).
#   python3 -m pytest tests/test_rage_sync.py -q
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagi.runtime.gitmind import GitMind
from sagi.runtime.rage_sync import RageSync
from services.memory_service import MemoryService


def _root():
    r = tempfile.mkdtemp()
    os.makedirs(os.path.join(r, "modules"))
    return r


def _mem():
    return MemoryService(os.path.join(tempfile.mkdtemp(), "m.db"))


def test_save_all_and_semantic_recall():
    r = _root()
    open(os.path.join(r, "modules", "01-a.md"), "w").write("# Alpha\nthe mitochondria is the powerhouse")
    open(os.path.join(r, "modules", "02-b.md"), "w").write("# Beta\nquicksort partitions around a pivot")
    gm = GitMind(r)
    gm.commit(ts=100)
    sync = RageSync(gm, memory=_mem(), session="s")
    saved = sync.save_all()
    assert saved == 2
    # SQLite fallback = substring recall (pgvector/RAGE does true semantic search).
    hits = sync.search("mitochondria")
    assert hits and "powerhouse" in hits[0]["text"]
    assert hits[0]["file"] == "01-a.md" and hits[0]["commit"]      # tagged with its tree moment


def test_save_dedups_identical_forms_across_commits():
    r = _root()
    open(os.path.join(r, "modules", "01-a.md"), "w").write("stable content")
    gm = GitMind(r)
    gm.commit(ts=100)
    open(os.path.join(r, "modules", "02-b.md"), "w").write("added later")
    gm.commit(ts=200)
    sync = RageSync(gm, memory=_mem(), session="s")
    # 01-a is unchanged across both commits -> saved once; 02-b -> once. Total 2, not 3.
    assert sync.save_all() == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
