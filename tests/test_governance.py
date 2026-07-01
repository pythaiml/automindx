# tests/test_governance.py — edit governance: self + sub editable, sov self-only.
#   python3 -m pytest tests/test_governance.py -q
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagi.runtime.governance import can_edit, governed_write
from sagi.runtime.spawn import spawn


def _parent():
    r = tempfile.mkdtemp()
    os.makedirs(os.path.join(r, "modules"))
    open(os.path.join(r, "identity.json"), "w").write(json.dumps({"id": "root", "baseId": "sagi"}))
    return r


def test_a_sagi_can_edit_itself_and_its_sub_children():
    p = _parent()
    sub = spawn(p, "Scout", mode="sub")["dir"]
    assert can_edit(p, p) is True                     # edit self
    assert can_edit(p, sub) is True                   # edit subordinate child
    grand = spawn(sub, "Recon", mode="sub")["dir"]
    assert can_edit(p, grand) is True                 # edit sub-of-sub


def test_only_the_sovereign_can_edit_a_sov_sagi():
    p = _parent()
    sov = spawn(p, "Athena", mode="sov")["dir"]
    assert can_edit(p, sov) is False                  # parent may NOT edit a sovereign
    assert can_edit(sov, sov) is True                 # the sovereign edits itself
    sov_sub = spawn(sov, "Aide", mode="sub")["dir"]
    assert can_edit(sov, sov_sub) is True             # a sovereign governs its own subtree
    assert can_edit(p, sov_sub) is False              # parent can't reach across the sovereignty boundary


def test_governance_is_downward_only_and_not_lateral():
    p = _parent()
    a = spawn(p, "A", mode="sub")["dir"]
    b = spawn(p, "B", mode="sub")["dir"]
    assert can_edit(a, p) is False                    # a child cannot edit its parent
    assert can_edit(a, b) is False                    # siblings cannot edit each other


def test_governed_write_enforces_permission_and_confinement():
    p = _parent()
    sub = spawn(p, "Scout", mode="sub")["dir"]
    sov = spawn(p, "Athena", mode="sov")["dir"]
    dest = governed_write(p, sub, "modules/99-parent-edit.md", "# from parent")  # allowed
    assert os.path.isfile(dest)
    try:
        governed_write(p, sov, "modules/x.md", "nope")                            # denied
        assert False, "expected PermissionError editing a sovereign"
    except PermissionError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
