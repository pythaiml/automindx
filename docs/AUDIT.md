# automindx — audit & fixes

A code audit of automindx (informed by the AGLM rebuild at
https://github.com/pythaiml/aglm). The theme of every issue below is the same one
that plagued the AGLM prototype: **the app tries to load a multi-GB model before
the UI renders**, and several entrypoints were broken outright. Fixes preserve the
project's identity (Professor Codephreak) and are additive where possible.

## Bugs found & fixed

| # | File | Problem | Fix |
|---|------|---------|-----|
| 1 | `hfapp.py` | Uses `os.path.exists` / `os.path` but **never imports `os`** → `NameError` at startup (this is the HF Spaces demo entrypoint). | Added `import os`. |
| 2 | `Dockerfile` | `CMD ["python", "app.py"]` — **no `app.py` exists** → container exits immediately. | Pointed entrypoint at `hfapp.py` (the self-contained GGML demo). |
| 3 | `memory.py` | `save_conversation_memory` **defined twice**; module **lacks `load_conversation_memory`** that `hfUIUX.py` imports → `ImportError`. | Removed the duplicate; added `load_conversation_memory` and `export_conversation`. |
| 4 | `hfUIUX.py` | `LlamaModel(MODEL_ID)` (missing arg), `format_to_llama_chat_style(history, user_input)` and `save_conversation_memory(history, file)` (both take 1 arg), double-formatting of the prompt, model built **at import time**, and `ChatInterface` fn returning a tuple instead of a string. | Lazy-load the model; corrected all call signatures; pass raw memory once; return the response string. Marked as the legacy heavy path. |
| 5 | `aglm.py` | `AutoModelForCausalLM.from_pretrained(model_path, device="cuda")` — `device` is **not a valid kwarg** and hard-codes CUDA → crashes on CPU. `main()` reads `dialog['user_input']`/`['model_response']` but memory saves `instruction`/`response`. | `device_map="auto"` (CPU/GPU), resolve a local dir **or** a HF repo id, default `models_folder`, and fixed the memory keys. |

## Improvement added

- **`ollama_codephreak.py`** — a modern entrypoint that talks to a running
  **Ollama** daemon instead of loading a model in-process. The Gradio UI loads
  instantly; local small models run on CPU and `:cloud` models are proxied to
  ollama.com. Features a model picker, live token streaming, a **token counter**
  (prompt + completion + tok/s), graceful errors (offline daemon, cloud
  subscription 403, model-not-found), and long-term memory persistence.
- **`requirements-ollama.txt`** — lightweight deps (gradio + requests) for the
  Ollama path; no torch/transformers/llama-cpp required.

## Still heavy (unchanged by design)

`uiux.py`, `hfUIUX.py`, and `hfapp.py` remain the original transformers /
llama.cpp paths for users who want fully-local, in-process inference. They are
now internally consistent, but still require the corresponding heavy
dependencies and memory. New users should prefer `ollama_codephreak.py`.

## Verification

- All Python modules compile (`python3 -m py_compile *.py`).
- `ollama_codephreak.py` boots and serves the Gradio UI (HTTP 200); a real
  generation returns text plus accurate token counts from Ollama
  (`prompt_eval_count` / `eval_count`).
