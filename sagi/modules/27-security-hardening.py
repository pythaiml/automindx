# sagi/modules/27-security-hardening.py — tier-3 · "shield thyself"
#
# GAP CLOSED: module-verifier proves a module ACTIVATES, and self-package-boundary confines writes,
# but nothing statically screens a *synthesized* module for dangerous constructs BEFORE it is
# activated, rate-limits runaway tool/build invocation, or signs savepoints for integrity. This adds
# that shield: scan() flags eval/exec/os.system/subprocess/socket/network/raw-open-for-write before
# activation; ratelimit() caps invocations per key; sign() produces a content signature.
#
# GROUNDED REUSE: module-verifier (the activate-contract gate this precedes), policy-guard (denials),
# host.store (read source, Store-confined). STDLIB only.
from __future__ import annotations

import hashlib
import os
import re
import time

MODULE_ID = "security-hardening"
DEPS = ["policy-guard", "module-verifier"]
MOTTO = "shield thyself"

# constructs a grown/synthesized module should not contain (defense-in-depth over the boundary)
_DANGER = [
    (re.compile(r"\bos\.system\s*\("), "os.system"),
    (re.compile(r"\bsubprocess\.(?:run|Popen|call|check_output)\s*\("), "subprocess exec"),
    (re.compile(r"(?<![.\w])eval\s*\("), "eval()"),
    (re.compile(r"(?<![.\w])exec\s*\("), "exec()"),
    (re.compile(r"\b__import__\s*\("), "__import__"),
    (re.compile(r"\bsocket\.socket\s*\("), "raw socket"),
    (re.compile(r"\b(?:urllib\.request|requests|http\.client)\b"), "network client"),
    (re.compile(r"\bshutil\.rmtree\s*\("), "recursive delete"),
    (re.compile(r"\bopen\s*\([^)]*['\"][rwa+b]*w"), "raw open-for-write (bypasses Store)"),
]


def _sib(host, mid):
    reg = getattr(host, "registry", None)
    return (reg["get"](mid) or {}).get("handle") if reg else None


def activate(host):
    _calls = {}   # key -> [timestamps] for rate limiting

    def scan(name_or_source):
        """Static-scan a module (by filename in modules/, or raw source) for dangerous constructs."""
        src = name_or_source
        if isinstance(name_or_source, str) and name_or_source.endswith(".py") and "\n" not in name_or_source:
            src = host.store.read(os.path.join("modules", name_or_source)) or ""
        findings = [label for rx, label in _DANGER if rx.search(src or "")]
        safe = not findings
        if not safe:
            host.log("security_scan", safe=False, findings=findings)
        return {"safe": safe, "findings": findings}

    def ratelimit(key, max_per_min=30):
        """True if this call is within the per-key rate budget; False if it should be throttled."""
        now = time.time()
        window = [t for t in _calls.get(key, []) if now - t < 60]
        window.append(now)
        _calls[key] = window
        allowed = len(window) <= max_per_min
        if not allowed:
            host.log("ratelimit_throttle", key=key, count=len(window))
        return allowed

    def sign(path_or_text):
        """A content signature (sha256 over identity + content) for savepoint/file integrity."""
        content = path_or_text
        if isinstance(path_or_text, str) and "\n" not in path_or_text:
            maybe = host.store.read(path_or_text)
            if maybe is not None:
                content = maybe
        sig = hashlib.sha256((getattr(host, "identity_id", "sagi") + "\0" + (content or "")).encode()).hexdigest()
        return {"signature": sig, "identity": getattr(host, "identity_id", "sagi")}

    host.log("module", step="tier3-27", id=MODULE_ID)
    return {"scan": scan, "ratelimit": ratelimit, "sign": sign}
