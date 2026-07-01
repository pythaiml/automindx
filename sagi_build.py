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
    python3 sagi_build.py --loop          # autonomous loop via aglm.AutonomousLoop
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time

import requests

SAGI_DIR = os.environ.get("SAGI_DIR", os.path.join(os.path.dirname(__file__), "sagi"))
MODULES_DIR = os.path.join(SAGI_DIR, "modules")
MANIFEST = os.path.join(SAGI_DIR, "manifest.json")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("CODEPHREAK_MODEL", "gpt-oss:120b-cloud")

SAVANTE = (
    "You are Savante — sAGI, a scientific savant general intelligence within "
    "AUTOMINDX. Reason with rigor across domains; be precise and calibrated.\n\n"
    "[sAGI MODE] Build sAGI — a self-building, agnostic, modular scientific "
    "general intelligence — one module per step."
)


def read_manifest() -> dict:
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"name": "sagi", "modules": []}


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (title or "module").lower()).strip("-")[:48]
    return s or "module"


def ask_model(model: str, prompt: str) -> str:
    """One non-streaming chat turn against Ollama."""
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={"model": model, "messages": [
            {"role": "system", "content": SAVANTE},
            {"role": "user", "content": prompt},
        ], "stream": False, "think": False, "options": {"num_predict": 400}},
        timeout=300,
    )
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()


def build_step(model: str) -> dict:
    """One PODA-style step: propose → specify → persist."""
    manifest = read_manifest()
    built = "; ".join(m["title"] for m in manifest["modules"]) or "(none)"
    prompt = (
        f"Modules built so far: {built}.\n"
        "Propose and specify the NEXT single module (do not repeat one already "
        "built): a short Title line, then a concise spec (purpose · interface · "
        "how it plugs into an agnostic core · how it advances self-building). "
        "Keep it modular and includable in any project (including as a Tauri app)."
    )
    text = ask_model(model, prompt)
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
    print(f"  ✓ step {step}: {title}  → sagi/modules/{fname}")
    return {"success": True, "step": step, "title": title}


def build(model: str, steps: int) -> None:
    print(f"sAGI headless build · model={model} · {steps} step(s) → {SAGI_DIR}")
    for _ in range(steps):
        try:
            build_step(model)
        except Exception as e:
            print(f"  ✗ step failed: {e}")
            break


async def build_loop(model: str, steps: int, interval: float = 2.0) -> None:
    """Drive the build with the migrated aGLM PODA loop (agnostic orchestrator)."""
    from aglm import AGLMCore, AutonomousLoop, Decision, PerceptionContext

    remaining = {"n": steps}

    async def perceive() -> PerceptionContext:
        return PerceptionContext(facts={"built": len(read_manifest()["modules"])}, source="sagi")

    async def decide(ctx, beliefs) -> Decision:
        return Decision(action="build" if remaining["n"] > 0 else "stop")

    async def act(d: Decision) -> dict:
        if d.action == "stop":
            return {"success": True, "done": True}
        remaining["n"] -= 1
        return build_step(model)

    core = AGLMCore(perceive=perceive, decide=decide, act=act, agent_id="sagi.builder")
    loop = AutonomousLoop(core, interval_seconds=interval)
    await loop.start()
    while remaining["n"] > 0:
        await asyncio.sleep(interval)
    await loop.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--loop", action="store_true", help="drive via aglm.AutonomousLoop")
    a = ap.parse_args()
    if a.loop:
        asyncio.run(build_loop(a.model, a.steps))
    else:
        build(a.model, a.steps)
