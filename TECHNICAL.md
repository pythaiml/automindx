# automindx — Technical Reference

automindx (codename **codephreak**) is Professor Codephreak's local-language-model
environment. This document describes how the pieces fit together and which
entrypoint to use. For the audit that produced the current code see
[AUDIT.md](AUDIT.md); for the project narrative see [README.md](README.md).

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

## 2. Module architecture

```
ollama_codephreak.py ─┐                         (modern path)
                      ├─► automind.py  (DEFAULT_SYSTEM_PROMPT, format_to_llama_chat_style)
uiux.py / hfUIUX.py ──┤                         (legacy transformers path)
                      ├─► aglm.py      (LlamaModel: transformers load + generate)
hfapp.py ─────────────┘                         (legacy GGML path, self-contained)
                      └─► memory.py    (save / load / export conversation JSON)
```

- **automind.py** — the codephreak persona (`DEFAULT_SYSTEM_PROMPT`) and
  `format_to_llama_chat_style(memory)`, which renders a `[[user, response], …]`
  history (last entry `[user, None]`) into Llama-2 `[INST]…[/INST]` chat format.
- **aglm.py** — `LlamaModel(model_name, models_folder="./models")`. Resolves a
  local model directory **or** a bare Hugging Face repo id, loads with
  `device_map="auto"` (CPU or GPU), and `generate_contextual_output(memory)`
  formats + tokenizes + generates.
- **memory.py** — `save_conversation_memory(memory)` writes
  `./memory/memory_<ts>.json` as `[{instruction, response}]`;
  `load_conversation_memory(path)` and `export_conversation(memory, file)` round-trip.
- **chunk4096.py / 4096chunk.md** — guard against inputs exceeding the 4096-token
  context of the original GGML model.

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
