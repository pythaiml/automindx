# tests/test_services.py — coverage for the codephreak service layer (audit #7).
#   python3 -m pytest -q      (or: python3 tests/test_services.py)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile

from services.config import settings
from services.memory_service import MemoryService
from services.inference_orchestrator import InferenceOrchestrator


def _tmp_db():
    return os.path.join(tempfile.mkdtemp(), "t.db")


def test_memory_roundtrip_and_persistence():
    db = _tmp_db()
    m = MemoryService(db)
    m.append("s1", "user", {"text": "hello"})
    m.append("s1", "assistant", {"text": "hi"})
    got = m.retrieve("s1")
    assert [t["role"] for t in got] == ["user", "assistant"]        # oldest-first
    assert got[0]["text"] == "hello"
    # persistence: a fresh instance on the same file sees the data
    assert MemoryService(db).retrieve("s1")[1]["text"] == "hi"


def test_memory_isolation_and_clear():
    m = MemoryService(_tmp_db())
    m.append("a", "user", {"text": "x"})
    m.append("b", "user", {"text": "y"})
    assert set(m.sessions()) == {"a", "b"}
    m.clear("a")
    assert m.retrieve("a") == []
    assert m.retrieve("b")[0]["text"] == "y"


def test_sanitize_strips_control_chars():
    clean = InferenceOrchestrator._sanitize("hello\x00\x07 world\x1b[31m")
    assert clean == "hello world[31m"
    assert "\x00" not in clean and "\x1b" not in clean
    assert InferenceOrchestrator._sanitize("   ") == ""


def test_orchestrator_run_with_mocked_model():
    orch = InferenceOrchestrator()
    orch.mem = MemoryService(_tmp_db())
    orch.model.predict = lambda messages, think=False: "mocked reply"  # no Ollama needed
    out = orch.run("what is 2+2?", "sess")
    assert out["status"] == "ok" and out["response"] == "mocked reply"
    # the turn was persisted for continuity
    hist = orch.mem.retrieve("sess")
    assert hist[0]["text"] == "what is 2+2?" and hist[1]["text"] == "mocked reply"


def test_config_defaults():
    assert settings.max_context >= 1 and settings.ollama_host.startswith("http")


def test_health_snapshot():
    orch = InferenceOrchestrator()
    orch.model.ping = lambda: True
    h = orch.health()
    assert h["status"] == "ok" and h["ollama"] is True and h["capacity"] >= 1


def test_overload_guard_rejects_at_capacity():
    import services.inference_orchestrator as io
    orch = InferenceOrchestrator()
    orch.mem = MemoryService(_tmp_db())
    orch.model.predict = lambda messages, think=False: "ok"
    # Drain the semaphore to simulate a full box, then a request must be rejected fast.
    held = [io._INFLIGHT.acquire(blocking=False) for _ in range(io._MAX_CONCURRENCY)]
    try:
        out = orch.run("hello", "s")
        assert out["status"] == "busy"
    finally:
        for h in held:
            if h:
                io._INFLIGHT.release()


def test_memory_factory_defaults_to_sqlite():
    from services.memory import get_memory
    assert type(get_memory()).__name__ == "MemoryService"  # compatible default


def test_sqlite_keyword_search():
    m = MemoryService(_tmp_db())
    m.append("s", "user", {"text": "the mitochondria is the powerhouse"})
    m.append("s", "assistant", {"text": "unrelated"})
    hits = m.search("mitochondria", "s")
    assert len(hits) == 1 and "powerhouse" in hits[0]["text"]


def test_rage_backend_falls_back_when_unavailable(monkeypatch=None):
    # Requesting pgvector without the deps/DB must not crash — falls back to SQLite.
    os.environ["AUTOMINDX_MEMORY_BACKEND"] = "pgvector"
    import importlib
    import services.config as cfg
    importlib.reload(cfg)
    import services.memory as mem
    importlib.reload(mem)
    assert type(mem.get_memory()).__name__ == "MemoryService"
    os.environ.pop("AUTOMINDX_MEMORY_BACKEND", None)
    importlib.reload(cfg); importlib.reload(mem)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
    sys.exit(0)
