#!/usr/bin/env python3
"""
sagi_build.py — headless self-builder for the agnostic `sagi/` package.

Proves sagi/ is orchestrator-agnostic: the automindX console is one host, this
CLI is another. Both drive the same contract — ask a chosen model for the next
module (grown from the Savante persona), then persist it to sagi/modules/ and
update sagi/manifest.json. Same result, no browser.

Deps: requests (+ a running Ollama). Reuses the aGLM PODA loop from aglm/.

    python3 sagi_build.py                 # build 3 modules with the default model
    python3 sagi_build.py --steps 5 --model gpt-oss:120b-cloud
    python3 sagi_build.py --backend claude-cli   # build with Claude via your CLI subscription
    python3 sagi_build.py --backend claude-api   # build with the Anthropic API (ANTHROPIC_API_KEY)
    python3 sagi_build.py --loop          # autonomous loop via aglm.AutonomousLoop

Backends (agnostic — sAGI doesn't care which mind drives it):
    ollama      (default) a running Ollama daemon
    claude-cli  the host `claude` binary in headless mode — uses your Claude
                subscription, no API key ("log into the terminal, call claude")
    claude-api  the Anthropic Messages API with ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

import requests

SAGI_DIR = os.environ.get("SAGI_DIR", os.path.join(os.path.dirname(__file__), "sagi"))
MODULES_DIR = os.path.join(SAGI_DIR, "modules")
MANIFEST = os.path.join(SAGI_DIR, "manifest.json")
# sAGI is a read-write sandbox layer that morphs from Savante; every run and every
# module it grows is appended to .history so the whole self-build can be replayed.
HISTORY_DIR = os.path.join(SAGI_DIR, ".history")
HISTORY_LOG = os.path.join(HISTORY_DIR, "build.jsonl")
GOAL_FILE = os.path.join(SAGI_DIR, "goal.txt")
CURRENT_GOAL = ""   # set from --goal or goal.txt; steers each module choice


def read_goal() -> str:
    try:
        with open(GOAL_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


# Ataraxia circuit breaker — for continuous builds, perfection is Ataraxia: stop
# when the architecture reaches "done" / "ready" (converged, nothing new) or a
# "time" bound, rather than over-building past perfection.
ATARAXIA_PATIENCE = int(os.environ.get("SAGI_ATARAXIA_PATIENCE", "3"))   # consecutive no-new → done
ATARAXIA_SECONDS = float(os.environ.get("SAGI_ATARAXIA_SECONDS", "0"))   # >0 → time breaker
_DONE_RE = re.compile(
    r"\b(ataraxia|no further modules?|nothing (?:left|more) to build|"
    r"the system is complete|architecture (?:is )?complete|feature[- ]complete|done and ready)\b", re.I)


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def log_history(event: dict) -> None:
    try:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        with open(HISTORY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": int(time.time()), **event}) + "\n")
    except OSError:
        pass
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("CODEPHREAK_MODEL", "gpt-oss:120b-cloud")
# Model backend: 'ollama' (default) | 'claude-cli' (host `claude` CLI, uses your
# Claude subscription — no API key) | 'claude-api' (Anthropic API + ANTHROPIC_API_KEY).
BACKEND = os.environ.get("SAGI_BACKEND", "ollama")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")

SAVANTE = (
    "You are Savante — sAGI, a scientific savant general intelligence within "
    "AUTOMINDX. Reason with rigor across domains; be precise and calibrated.\n\n"
    "[sAGI MODE] Build sAGI — a self-building, agnostic, modular scientific "
    "general intelligence — one module per step."
)

# The dynamic self-audit prompt (21-self-prompt-auditor.py persists it here on every sAGI update).
# When present it REPLACES the static SAVANTE so the headless backends inject the same live,
# self-adjusting prompt the in-process host does; SAVANTE stays the guaranteed fallback.
SYS_PROMPT_FILE = os.path.join(SAGI_DIR, "system_prompt.md")


def _system() -> str:
    """The system prompt to inject into the model: the dynamic self-audit if grown, else SAVANTE."""
    try:
        with open(SYS_PROMPT_FILE, encoding="utf-8") as f:
            dyn = f.read().strip()
            if dyn:
                return dyn
    except OSError:
        pass
    return SAVANTE


def read_manifest() -> dict:
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"name": "sagi", "modules": []}


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (title or "module").lower()).strip("-")[:48]
    return s or "module"


def _ask_ollama(model: str, prompt: str) -> str:
    """One non-streaming chat turn against Ollama."""
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={"model": model, "messages": [
            {"role": "system", "content": _system()},
            {"role": "user", "content": prompt},
        ], "stream": False, "think": False, "options": {"num_predict": 400}},
        timeout=300,
    )
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()


def _ask_claude_cli(model: str, prompt: str) -> str:
    """Call Claude through the host's `claude` CLI in headless print mode.

    This is the 'log into the terminal, call claude' path: it uses whatever Claude
    subscription the CLI is already signed in with — no API key. Requires the
    `claude` binary on PATH (https://claude.com/claude-code).
    """
    if not shutil.which("claude"):
        raise RuntimeError("`claude` CLI not found on PATH — install Claude Code, or use --backend ollama")
    cmd = ["claude", "-p", prompt, "--append-system-prompt", _system()]
    if model and model.startswith("claude"):
        cmd += ["--model", model]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {(out.stderr or out.stdout).strip()[:200]}")
    return out.stdout.strip()


def _ask_claude_api(model: str, prompt: str) -> str:
    """Call Claude via the Anthropic Messages API (needs ANTHROPIC_API_KEY)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — export it, or use --backend claude-cli")
    mid = model if (model or "").startswith("claude") else ANTHROPIC_MODEL
    body = json.dumps({
        "model": mid, "max_tokens": 1024, "system": _system(),
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    return "".join(b.get("text", "") for b in data.get("content", [])).strip()


def ask_model(model: str, prompt: str, backend: str = None) -> str:
    """Dispatch one turn to the chosen backend (Ollama, Claude CLI, or Claude API)."""
    b = backend or BACKEND
    if b == "claude-cli":
        return _ask_claude_cli(model, prompt)
    if b == "claude-api":
        return _ask_claude_api(model, prompt)
    return _ask_ollama(model, prompt)


def build_step(model: str, backend: str = None) -> dict:
    """One PODA-style step: propose → specify → persist."""
    manifest = read_manifest()
    built = "; ".join(m["title"] for m in manifest["modules"]) or "(none)"
    prior_norm = {_norm_title(m["title"]) for m in manifest["modules"]}
    prompt = (
        f"Modules built so far: {built}.\n"
        "Propose and specify the NEXT single module (do not repeat one already "
        "built): a short Title line, then a concise spec (purpose · interface · "
        "how it plugs into an agnostic core · how it advances self-building). "
        "Keep it modular and includable in any project (including as a Tauri app)."
    )
    if CURRENT_GOAL:
        prompt = f"Overarching goal: {CURRENT_GOAL}. Choose the next module that best advances this goal.\n" + prompt
    text = ask_model(model, prompt, backend)
    title = next((l for l in text.splitlines() if l.strip()), "Module")
    title = title.replace("**", "")
    title = re.sub(r"^#+\s*|^title:\s*|^module\s*\d*\s*[:\-–]\s*", "", title, flags=re.I).strip("* ").strip()[:90] or "Module"

    step = len(manifest["modules"]) + 1
    os.makedirs(MODULES_DIR, exist_ok=True)
    fname = f"{step:02d}-{slug(title)}.md"
    with open(os.path.join(MODULES_DIR, fname), "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{text}\n")
    manifest["modules"] = [m for m in manifest["modules"] if m["file"] != fname]
    manifest["modules"].append({"step": step, "title": title, "file": fname, "ts": int(time.time())})
    manifest["version"] = f"0.0.{len(manifest['modules'])}"
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    log_history({"event": "module", "step": step, "title": title, "file": fname, "backend": backend or BACKEND})
    print(f"  ✓ step {step}: {title}  → sagi/modules/{fname}")
    return {"success": True, "step": step, "title": title,
            "redundant": _norm_title(title) in prior_norm,     # nothing new → converging
            "done": bool(_DONE_RE.search(text))}                # the build signalled completion


def build(model: str, steps: int, backend: str = None) -> None:
    print(f"sAGI headless build · backend={backend or BACKEND} · model={model} · {steps} step(s) → {SAGI_DIR}")
    log_history({"event": "run_start", "backend": backend or BACKEND, "model": model, "steps": steps})
    done = 0
    calm = 0            # consecutive no-new (redundant) steps
    t0 = time.time()
    for _ in range(steps):
        try:
            r = build_step(model, backend)
            done += 1
        except Exception as e:
            print(f"  ✗ step failed: {e}")
            log_history({"event": "error", "detail": str(e)[:200]})
            break
        calm = calm + 1 if r.get("redundant") else 0
        reason = ataraxia(r, calm, t0)
        if reason:
            log_history({"event": "ataraxia", "reason": reason, "built": done})
            print(f"  ⊙ Ataraxia reached ({reason}) — sAGI has converged to perfection; circuit breaker tripped.")
            break
    log_history({"event": "run_end", "built": done})


def ataraxia(result: dict, calm: int, t0: float):
    """Return the trip reason if perfection (Ataraxia) is reached, else None."""
    if result.get("done"):
        return "ready"                                  # the build signalled it is done/ready
    if calm >= ATARAXIA_PATIENCE:
        return "done"                                   # converged — nothing new to build
    if ATARAXIA_SECONDS and (time.time() - t0) > ATARAXIA_SECONDS:
        return "time"                                   # time bound reached
    return None


async def build_loop(model: str, steps: int, interval: float = 2.0, backend: str = None) -> None:
    """Drive the build with the migrated aGLM PODA loop (agnostic orchestrator)."""
    from aglm import AGLMCore, AutonomousLoop, Decision, PerceptionContext

    remaining = {"n": steps}
    state = {"calm": 0, "t0": time.time(), "ataraxia": None}   # Ataraxia circuit breaker

    async def perceive() -> PerceptionContext:
        return PerceptionContext(facts={"built": len(read_manifest()["modules"])}, source="sagi")

    async def decide(ctx, beliefs) -> Decision:
        if state["ataraxia"] or remaining["n"] <= 0:
            return Decision(action="stop")
        return Decision(action="build")

    async def act(d: Decision) -> dict:
        if d.action == "stop":
            return {"success": True, "done": True}
        remaining["n"] -= 1
        r = build_step(model, backend)
        state["calm"] = state["calm"] + 1 if r.get("redundant") else 0
        state["ataraxia"] = ataraxia(r, state["calm"], state["t0"])
        if state["ataraxia"]:
            log_history({"event": "ataraxia", "reason": state["ataraxia"]})
            print(f"  ⊙ Ataraxia reached ({state['ataraxia']}) — circuit breaker tripped.")
        return r

    core = AGLMCore(perceive=perceive, decide=decide, act=act, agent_id="sagi.builder")
    loop = AutonomousLoop(core, interval_seconds=interval)
    await loop.start()
    while remaining["n"] > 0 and not state["ataraxia"]:
        await asyncio.sleep(interval)
    await loop.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--backend", default=BACKEND, choices=["ollama", "claude-cli", "claude-api"],
                    help="model backend (claude-cli uses your Claude subscription via the terminal)")
    ap.add_argument("--goal", default="", help="overarching goal that steers each module choice")
    ap.add_argument("--set-goal", default=None, help="persist a standing goal to sagi/goal.txt and exit")
    ap.add_argument("--loop", action="store_true", help="drive via aglm.AutonomousLoop")
    a = ap.parse_args()
    if a.set_goal is not None:
        os.makedirs(SAGI_DIR, exist_ok=True)
        with open(GOAL_FILE, "w", encoding="utf-8") as f:
            f.write(a.set_goal.strip() + "\n")
        log_history({"event": "goal", "goal": a.set_goal.strip()})
        print(f"sAGI goal set → {a.set_goal.strip() or '(cleared)'}")
        raise SystemExit(0)
    CURRENT_GOAL = a.goal.strip() or read_goal()
    if CURRENT_GOAL:
        print(f"sAGI goal: {CURRENT_GOAL}")
    # If a Claude backend is chosen but the model is still the Ollama default, use a Claude model.
    model = a.model
    if a.backend.startswith("claude") and not model.startswith("claude"):
        model = ANTHROPIC_MODEL
    if a.loop:
        asyncio.run(build_loop(model, a.steps, backend=a.backend))
    else:
        build(model, a.steps, backend=a.backend)
