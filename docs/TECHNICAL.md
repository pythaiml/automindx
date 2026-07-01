# automindx — Technical Reference

automindx (codename **codephreak**) is Professor Codephreak's local-language-model
environment. This document describes how the pieces fit together and which
entrypoint to use. For the audit that produced the current code see
[AUDIT.md](AUDIT.md); for the project narrative see [README.md](../README.md).

## 1. Two ways to run

| Path | Entrypoint | Loads | Needs |
|---|---|---|---|
| **Modern (recommended)** | `ollama_codephreak.py` | **instantly** | a running Ollama daemon |
| Legacy GGML (CPU) | `hfapp.py` | after model download | `llama-cpp-python` + ~4 GB model |
| Legacy transformers | `hfUIUX.py` / `uiux.py` | after model load | `torch` + `transformers` + RAM/GPU |

The legacy paths load a multi-GB model **in-process before the UI renders**, so on
modest hardware the window may never appear. `ollama_codephreak.py` keeps the model
in the Ollama daemon, so the Gradio UI loads immediately and degrades gracefully
when no model is present.

## 2. Repository map (current)

```
automindX/
  codephreak-console/   Next.js + Vercel AI SDK console (the flagship UI)
      app/api/chat        native-Ollama bridge: tool-calling (real filesystem access)
                          + subtask decomposition on the tool-call limit
      app/ Substrate · SagiVisual · SagiBackground · SagiFace · ClaudeTerminal
                          living WebGL substrate · sAGI constellation · evolving
                          geometry · sensorium (oscilloscope/TTS · mic · cameras) ·
                          interactive PTY terminal (call `claude` from your subscription)
      lib/persona.ts      expandable personas: base template + individuality layer,
                          per-persona avatars (Codephreak, Savante, sAGI, jAImla, …)
      pty-server.js       local-only node-pty ⇄ WebSocket bridge for the terminal
  services/             decoupled service layer — see SERVICES.md
      Memory (SQLite) · RageMemory (pgvector/pgvectorscale) · get_memory() ·
      ModelService (lazy Ollama) · InferenceOrchestrator · ModelRegistry ·
      secrets (keyring→env) · self_audit (codephreak reads the real source)
  aglm/                 the Autonomous General Learning Model package (PODA cycle)
  sagi/                 the self-building sAGI package (agnostic, modular)
      POINT_OF_DEPARTURE.md · seed/ (the 3 seed modules) · core/interface.md
      runtime/          executable: Host {log,store,callModel,on,emit} · the three
                        seed modules (be thyself · do no harm · grow thyself) ·
                        gitmind (git-like memory tree) · rage_sync (save trees→RAGE)
  4096/                 the 4096-token issue: context4096.py, chunk4096.py + docs
  scripts/              automindx.install, run_codephreak_console.sh
  automind.py           thin façade → services (chat) + persona/format helpers
  codephreak.py         self-improving persona engine (realtime 👍/👎 → directives)
  ollama_codephreak.py  modern Gradio chat (model picker + token counter)
  uiux.py hfUIUX.py hfapp.py llama_model.py memory.py   legacy transformers/GGML path
```

- **automind.py** — thin façade: `chat(user_input, session_id)` delegates to the
  service layer; also exports the codephreak persona (`DEFAULT_SYSTEM_PROMPT`) and
  `format_to_llama_chat_style`.
- **services/** — the production layer (memory · model · orchestrator · registry ·
  secrets · self-audit). See [SERVICES.md](SERVICES.md).
- **sagi/runtime/** — the point of departure made executable + the **gitmind** memory
  tree saved into **RAGE**. See [../sagi/POINT_OF_DEPARTURE.md](../sagi/POINT_OF_DEPARTURE.md)
  and [savethetrees.md](savethetrees.md).
- **llama_model.py** *(was `aglm.py`)* — `LlamaModel`: local dir or HF id, `device_map="auto"`.
- **memory.py** — legacy JSON conversation memory; the service layer uses SQLite/pgvector.
- **4096/context4096.py** — token-aware sliding window for the original 4096-token model.

> **Reality note.** Models run in the **Ollama daemon** (local + free `:cloud`), not an
> in-process torch checkpoint; self-improvement is prompt-space (feedback → learned
> directives). The service layer, registry, secrets, self-audit, and sAGI runtime are
> real and tested (`python3 -m pytest`).

## 3. The modern entrypoint (`ollama_codephreak.py`)

- Talks to Ollama's native `/api/chat` (streaming) at `OLLAMA_HOST`
  (default `http://localhost:11434`), with `think=False` for snappy CPU replies.
- `list_models()` pulls the live model list (`/api/tags`); the dropdown offers
  local and `:cloud` models (`AUTOMINDX_MODEL` sets the default, `qwen3:0.6b`).
- Prepends `DEFAULT_SYSTEM_PROMPT` so every reply is "codephreak".
- **Token counter**: from the final stream object's `prompt_eval_count`,
  `eval_count`, and `eval_duration` → `🪙 N tokens — P prompt + C completion · R tok/s`.
- Graceful errors: daemon offline, cloud-subscription 403, model-not-found.
- Persists each exchange via `memory.save_conversation_memory`.
- Version-robust Gradio: a `Blocks` chatbot (works on gradio 3.x and 4.x) instead
  of `ChatInterface(additional_inputs=…)`, which 3.37 does not support.

```bash
pip install -r requirements-ollama.txt   # gradio + requests, no torch
ollama serve && ollama pull qwen3:0.6b
python3 ollama_codephreak.py             # → http://localhost:7860
```

## 4. Relationship to AGLM

codephreak is **AGLM** — an augmentation layer, not a model. The full AGLM
cognitive console (Socratic, Logic, Nonmonotonic, Epistemic, BDI, MASTERMIND,
Autonomize, Memory, Prediction) and an AI-SDK streaming participant UI live at
**https://github.com/GATERAGE/aglm**. automindx is the deployment environment that
gives that augmentation a persistent, persona-driven chat surface.

## 5. Docker

`Dockerfile` builds the legacy heavy stack and runs `hfapp.py` (the self-contained
GGML demo that downloads its model on first run) on port 7860. For the Ollama path,
run Ollama on the host and `ollama_codephreak.py` with `requirements-ollama.txt`.
