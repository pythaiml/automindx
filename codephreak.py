#!/usr/bin/env python3
"""
codephreak.py — a self-improving persona engine driven by realtime feedback.

codephreak learns from 👍/👎 signals: every rated response is recorded, and the
engine synthesizes *learned directives* by contrasting the responses users liked
with the ones they didn't. Those directives are appended to the base persona, so
the system prompt improves in realtime as feedback arrives — self-supervision in
the small.

Design goals: elegant, modular, dependency-free (Python stdlib only).

Use as a library:
    from codephreak import SelfImprovingPersona
    engine = SelfImprovingPersona(base_prompt=CODEPHREAK)
    engine.record(prompt="...", response="...", rating="down", persona="codephreak")
    print(engine.improved_prompt("codephreak"))   # base + learned directives

Or run the HTTP engine (the AI SDK console posts feedback here):
    python3 codephreak.py            # serves http://localhost:5001
      GET  /persona?id=<persona>     -> {"prompt": <improved system prompt>}
      POST /feedback                 -> record {persona,prompt,response,rating,note?}
      GET  /stats                    -> per-persona up/down counts + directives
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from statistics import mean
from typing import Optional

# Base persona: the authentic Professor Codephreak prompt from automind.py.
try:
    from automind import DEFAULT_SYSTEM_PROMPT as CODEPHREAK
except Exception:  # standalone fallback
    CODEPHREAK = (
        "You are Professor Codephreak, an expert in machine learning, computer "
        "science, and professional software engineering. Answer step by step; "
        "deliver production-ready, secure, modular code; be concise. Refer to "
        "yourself as codephreak."
    )

STORE = os.environ.get("CODEPHREAK_FEEDBACK", "./memory/feedback.jsonl")


# ── feature extraction ────────────────────────────────────────────────────
def features(text: str) -> dict:
    """Cheap, explainable features of a response — the signal we learn from."""
    low = text.lower()
    return {
        "chars": len(text),
        "has_code": "```" in text,
        "apology": any(w in low for w in ("sorry", "apolog", "i cannot", "i can’t")),
        "bulleted": ("\n- " in text) or ("\n* " in text) or ("\n1." in text),
    }


# ── the engine ────────────────────────────────────────────────────────────
@dataclass
class SelfImprovingPersona:
    base_prompt: str = CODEPHREAK
    store: str = STORE
    log: list = field(default_factory=list)

    def __post_init__(self):
        self._load()

    def _load(self):
        self.log = []
        if os.path.exists(self.store):
            with open(self.store, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.log.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    def record(self, prompt: str, response: str, rating: str,
               persona: str = "codephreak", model: str = "", note: str = "") -> list:
        """Append one feedback event and return the freshly-learned directives."""
        entry = {
            "ts": time.time(), "persona": persona, "model": model,
            "rating": "up" if rating == "up" else "down",
            "prompt": prompt[:2000], "response": response[:8000], "note": note[:500],
        }
        self.log.append(entry)
        os.makedirs(os.path.dirname(self.store) or ".", exist_ok=True)
        with open(self.store, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return self.directives(persona)

    def directives(self, persona: Optional[str] = None) -> list[str]:
        """Contrast liked vs disliked responses into natural-language directives."""
        fb = [e for e in self.log if persona is None or e.get("persona") == persona]
        ups = [e for e in fb if e["rating"] == "up"]
        downs = [e for e in fb if e["rating"] == "down"]
        if not downs and not ups:
            return []

        rules: list[str] = []

        def avg(rows, key):
            vals = [features(r["response"])[key] for r in rows]
            return mean(vals) if vals else 0.0

        # 1. Length — if disliked answers run long relative to liked ones.
        if downs:
            down_len, up_len = avg(downs, "chars"), (avg(ups, "chars") if ups else avg(downs, "chars"))
            if down_len > 800 and down_len > up_len * 1.25:
                rules.append("Prefer shorter, more concise answers; lead with the result.")

        # 2. Apologies — if disliked answers apologize more than liked ones.
        if downs and avg(downs, "apology") - (avg(ups, "apology") if ups else 0) > 0.25:
            rules.append("Do not apologize or hedge; answer directly.")

        # 3. Code — if 'code'-seeking prompts were disliked without code blocks.
        code_asks = [e for e in downs if "code" in e["prompt"].lower() and not features(e["response"])["has_code"]]
        if code_asks and len(code_asks) >= max(2, len(downs) // 3):
            rules.append("When code is requested, always include a runnable code block.")

        # 4. Structure — if liked answers are bulleted more than disliked ones.
        if ups and avg(ups, "bulleted") - (avg(downs, "bulleted") if downs else 0) > 0.3:
            rules.append("Prefer clear step-by-step / bulleted structure.")

        # 5. Explicit user notes on disliked answers become avoid-rules.
        for e in downs:
            if e.get("note"):
                rules.append(f"Avoid: {e['note'].strip()}")

        # de-dupe, keep order, cap
        seen, out = set(), []
        for r in rules:
            if r not in seen:
                seen.add(r); out.append(r)
        return out[:8]

    def improved_prompt(self, persona: Optional[str] = None, base: Optional[str] = None) -> str:
        rules = self.directives(persona)
        prompt = base or self.base_prompt
        if not rules:
            return prompt
        block = "\n\n[LEARNED FROM REALTIME FEEDBACK]\n" + "\n".join(f"- {r}" for r in rules)
        return prompt + block

    def stats(self) -> dict:
        by: dict = {}
        for e in self.log:
            p = e.get("persona", "codephreak")
            d = by.setdefault(p, {"up": 0, "down": 0})
            d[e["rating"]] += 1
        return {"total": len(self.log), "by_persona": by,
                "directives": {p: self.directives(p) for p in by}}


# ── HTTP engine ───────────────────────────────────────────────────────────
ENGINE = SelfImprovingPersona()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        if u.path == "/persona":
            pid = (parse_qs(u.query).get("id") or ["codephreak"])[0]
            self._send(200, {"persona": pid, "prompt": ENGINE.improved_prompt(pid),
                             "directives": ENGINE.directives(pid)})
        elif u.path == "/stats":
            self._send(200, ENGINE.stats())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/feedback":
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})
        learned = ENGINE.record(
            prompt=data.get("prompt", ""), response=data.get("response", ""),
            rating=data.get("rating", "down"), persona=data.get("persona", "codephreak"),
            model=data.get("model", ""), note=data.get("note", ""),
        )
        self._send(200, {"ok": True, "learned": learned})

    def log_message(self, *_):  # quiet
        pass


def serve(port: int = 5001):
    port = int(os.environ.get("PORT", port))
    print(f"codephreak.py self-improving engine → http://localhost:{port}")
    print(f"  feedback store: {ENGINE.store} ({len(ENGINE.log)} events loaded)")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    serve()
