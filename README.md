<div align="center">

# automindx — IAML

### Intelligent Autonomous Machine Learning · *"I Am Machine Learning"*

**Professor Codephreak** — an expert in machine learning, computer science, and
secure, modular programming — as a local, persona-driven language-model
environment. Project codename: **codephreak**.

[![Loads instantly](https://img.shields.io/badge/UI-loads%20instantly-4b9)](TECHNICAL.md)
[![Models](https://img.shields.io/badge/models-Ollama%20local%20%2B%20cloud-000)](TECHNICAL.md)
[![Gradio](https://img.shields.io/badge/UI-Gradio-f97316)](https://gradio.app)
[![audit](https://img.shields.io/badge/audit-AUDIT.md-blue)](AUDIT.md)

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

📖 **[TECHNICAL.md](TECHNICAL.md)** — module map & entrypoints &nbsp;·&nbsp;
🔧 **[AUDIT.md](AUDIT.md)** — audit findings & fixes

---

## Entrypoints

| Path | File | Loads | Requires |
|---|---|---|---|
| **Modern (recommended)** | `ollama_codephreak.py` | **instantly** | a running Ollama daemon |
| Legacy GGML (CPU) | `hfapp.py` | after model download | `llama-cpp-python` + ~4 GB model |
| Legacy transformers | `uiux.py` / `hfUIUX.py` | after model load | `torch` + `transformers` |

## Modules

> Documentation: **codephreak = `uiux.py` + `memory.py` + `automind.py` + `aglm.py`**

| Module | Responsibility |
|---|---|
| `uiux.py` | Gradio interface: captures input, streams the model's response, manages memory. |
| `automind.py` | The codephreak persona and `format_to_llama_chat_style` — renders history into Llama-2 chat format. |
| `memory.py` | Conversation memory — saves, loads, and exports `./memory/*.json` (`instruction` / `response`). |
| `aglm.py` | `LlamaModel` — loads a local dir **or** a Hugging Face id (`device_map="auto"`) and generates. |
| `ollama_codephreak.py` | Modern Ollama-backed chat with model picker + token counter. |

---

## Install

The [`automindx.install`](https://github.com/pythaiml/automindx/blob/main/automindx.install)
script sets up the full environment (tested on Ubuntu 22.04, Linux Mint 21.2, and
Mandrake Linux):

```bash
chmod +x automindx.install && ./automindx.install
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

> **Note.** The legacy paths load the model in-process; inputs larger than 4096
> characters can crash the original GGML model. Prefer `ollama_codephreak.py`.

---

## Links

- **Professor Codephreak** — <https://github.com/Professor-Codephreak>
- **aGLM** (augmentation layer + consoles) — <https://github.com/GATERAGE/aglm>
- Original codephreak (historical) — <https://github.com/Professor-Codephreak/automind>

*This repo is under active development.*
