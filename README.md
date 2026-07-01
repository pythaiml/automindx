<div align="center">

# automindx — IAML

### Intelligent Autonomous Machine Learning · *"I Am Machine Learning"*

**Professor Codephreak** — an expert in machine learning, computer science, and
secure, modular programming — as a local, persona-driven language-model
environment. Project codename: **codephreak**.

[![Loads instantly](https://img.shields.io/badge/UI-loads%20instantly-4b9)](docs/TECHNICAL.md)
[![Models](https://img.shields.io/badge/models-Ollama%20local%20%2B%20cloud-000)](docs/TECHNICAL.md)
[![Gradio](https://img.shields.io/badge/UI-Gradio-f97316)](https://gradio.app)
[![audit](https://img.shields.io/badge/audit-AUDIT.md-blue)](docs/AUDIT.md)

</div>

---

## ⚡ Quickstart — the modern, actually-loads path (Ollama)

The classic `uiux.py` / `hfUIUX.py` load a multi-GB model **before** the UI
renders, so on modest hardware the window may never appear. `ollama_codephreak.py`
inverts that: the model runs in the **Ollama** daemon and the Gradio UI loads
instantly (degrading gracefully when no model is present). It ships a model picker
(local + `:cloud`), live streaming, and a **token counter** (prompt + completion +
tok/s).

```bash
pip install -r requirements-ollama.txt   # gradio + requests only — no torch
ollama serve                             # in another terminal
ollama pull qwen3:0.6b                    # small local model for modest hardware
python3 ollama_codephreak.py             # → http://localhost:7860
```

> **codephreak is AGLM** — an augmentation layer, not a model. The full AGLM
> cognitive console (Socratic, Logic, BDI, MASTERMIND, …) and an AI-SDK streaming
> participant UI live at **[GATERAGE/aglm](https://github.com/GATERAGE/aglm)**.

📚 **[docs/NAV.md](docs/NAV.md)** — full documentation index &nbsp;·&nbsp;
📖 **[TECHNICAL.md](docs/TECHNICAL.md)** — module map & entrypoints &nbsp;·&nbsp;
🔧 **[AUDIT.md](docs/AUDIT.md)** — audit findings & fixes

---

## Entrypoints

| Path | File | Loads | Requires |
|---|---|---|---|
| **AI SDK console (definitive)** | `codephreak-console/` + `codephreak.py` | **instantly** | Ollama + Node (see [CODEPHREAK_CONSOLE.md](docs/CODEPHREAK_CONSOLE.md)) |
| Modern Gradio | `ollama_codephreak.py` | **instantly** | a running Ollama daemon |
| Legacy GGML (CPU) | `hfapp.py` | after model download | `llama-cpp-python` + ~4 GB model |
| Legacy transformers | `uiux.py` / `hfUIUX.py` | after model load | `torch` + `transformers` |

The **[Professor Codephreak AI SDK console](docs/CODEPHREAK_CONSOLE.md)** is the
flagship UI: streaming chat on the Vercel AI SDK v7 with gpt-oss by default,
reasoning, a **live token / tok-s** counter, advanced **and** scientific sampling,
a `.persona` creator (Codephreak, automindX, jAImla, Savante, …), `.history`,
export/copy, ❤/💔 feedback into the self-improving `codephreak.py`, and a living
WebGL substrate that fluxes while thinking. codephreak also has **read-only
filesystem access** (tool-calling: `list_files`/`read_file`/`grep`) so it answers
about the *actual* code, and **decomposes large tasks into sub-tasks** completed
one at a time.

```bash
./scripts/run_codephreak_console.sh      # engine :5001 + console :3100
```

## Modules

> Documentation: **codephreak = `uiux.py` + `memory.py` + `automind.py` + `aglm.py`**

| Module | Responsibility |
|---|---|
| **`services/`** | The decoupled service layer (see **[SERVICES.md](docs/SERVICES.md)**): `MemoryService` (SQLite) · `RageMemory` (pgvector/pgvectorscale) · `get_memory()` factory · `ModelService` (lazy Ollama client) · `InferenceOrchestrator` (sanitize → RAG recall → infer → persist → JSON logs) · `ModelRegistry` (versioned config + digest integrity) · `secrets` (keyring→env) · **`self_audit`** (codephreak reads the real source). |
| `automind.py` | Thin façade — `chat(user_input, session_id)` delegates to the service layer; also exports the codephreak persona + `format_to_llama_chat_style`. |
| `codephreak.py` | Self-improving persona engine — learns directives from realtime 👍/👎 feedback (`GET /healthz`, `POST /feedback`). |
| `memory.py` | Legacy JSON conversation memory (`./memory/*.json`); the service layer uses SQLite/pgvector. |
| `llama_model.py` | `LlamaModel` — loads a local dir **or** a Hugging Face id (`device_map="auto"`). *(was `aglm.py`)* |
| `ollama_codephreak.py` | Modern Ollama-backed Gradio chat with model picker + token counter. |
| `aglm/` | The **Autonomous General Learning Model** package (PODA cycle · beliefs · autonomous loop), migrated from [GATERAGE/aglm](https://github.com/GATERAGE/aglm). See [AGLM.md](docs/AGLM.md). |
| `automind_aglm.py` | Wires aGLM to codephreak's feedback so automindX self-refines its persona. |

> **Reality note.** automindX runs models in the **Ollama daemon** (local + free
> `:cloud`), not an in-process torch checkpoint. Self-improvement is prompt-space
> (feedback → learned directives), not weight-space training. The service layer,
> registry, secrets, and self-audit are real and tested (`python3 -m pytest`).

---

## Install

The [`scripts/automindx.install`](https://github.com/pythaiml/automindx/blob/main/scripts/automindx.install)
script sets up the full environment (tested on Ubuntu 22.04, Linux Mint 21.2, and
Mandrake Linux):

```bash
chmod +x scripts/automindx.install && ./scripts/automindx.install
```

## Legacy GGML run

Auto-downloads `llama2-7b-chat-codeCherryPop-qLoRA-GGML` and launches Professor
Codephreak with an agenda to build the automindx deployment environment.

```bash
python3 uiux.py \
  --model_name="TheBloke/llama2-7b-chat-codeCherryPop-qLoRA-GGML" \
  --model_type="ggml" \
  --file_name="llama-2-7b-chat-codeCherryPop.ggmlv3.q4_1.bin" \
  --save_history
```

> **Note.** The original model has a hard 4096-token context. The token-aware
> sliding window in `context4096.py` (wired into `hfUIUX.py`) fixes the old
> overflow crash; modern models via Ollama sidestep it entirely with a larger
> `num_ctx`. See **[CONTEXT.md](4096/CONTEXT.md)**.

---

## Links

- **[Professor Codephreak](https://github.com/Professor-Codephreak)** — the origin and the evolving author; also runs the team of OpenAI agents at **[gpt.pythai.net](https://gpt.pythai.net)**
- **[github.com/GATERAGE](https://github.com/GATERAGE)** — the organization
  - **[GATERAGE/aglm](https://github.com/GATERAGE/aglm)** — Autonomous General Learning Model (package + consoles)
  - **[GATERAGE/RAGE](https://github.com/GATERAGE/RAGE)** — retrieval substrate (memory)
  - **[GATERAGE/mastermind](https://github.com/GATERAGE/mastermind)** — strategic orchestrator; **[mastermind.pythai.net](https://mastermind.pythai.net)** is the warcouncil interaction to the boardroom
- Original codephreak (historical) — <https://github.com/Professor-Codephreak/automind>
- **mindX** — the production system aGLM was distilled from · docs: **[mindx.pythai.net/docs.html](https://mindx.pythai.net/docs.html)**
- **[rage.pythai.net](https://rage.pythai.net)** — the public broadcast layer ("public noise"), where mindX publishes in the voice of mindX via its **AuthorAgent**; Professor Codephreak has published there occasionally

> **RAGE remembers · aGLM decides · MASTERMIND orchestrates.**

*This repo is under active development — Professor Codephreak keeps evolving.*
