# The 4096 limitation — analysis, fix, and the modern workflow

## The problem

The original codephreak model — `llama2-7b-chat-codeCherryPop-qLoRA-GGML` — is a
Llama-2 model with a **hard 4096-token context window**. When the system prompt +
conversation history + new question exceed 4096 tokens, generation crashes
(`Requested tokens exceed context window of 4096`). The README warned: *"inputs
larger than 4096 characters will crash the input → response."*

## Why the old fix (`chunk4096.py`) didn't work

`chunk4096.py` sliced the input into **4096-character** windows and generated a
response per slice, then concatenated them. Two problems:

1. **Wrong unit.** The limit is 4096 **tokens** (~12,000–16,000 characters), so it
   chopped far too early — and a single 4096-char slice plus the system prompt
   could still overflow in tokens.
2. **Lost coherence.** Each slice was generated in isolation with no shared
   context, so concatenated answers were incoherent. It also ignored the system
   prompt and history entirely.

## The fix — `context4096.py` (token-aware sliding window)

The correct approach keeps the model within its window instead of chopping input:

- Count tokens with the **actual tokenizer**.
- Always keep the **system prompt** and the **newest turns** that fit within
  `n_ctx − reserve` (reserve leaves room for the response).
- Drop the **oldest** turns first. The latest user turn is always preserved.

```python
from context4096 import ContextWindow
window = ContextWindow(tokenizer, n_ctx=4096, reserve=512)
memory = window.fit(memory, DEFAULT_SYSTEM_PROMPT)   # never overflows
```

This is wired into `hfUIUX.py`. Result (verified with a real tokenizer): a
conversation that would overflow at ~12k tokens is trimmed to fit 3,584 tokens,
keeping the most recent turns and the current question — **no crash, coherent
context**. It degrades gracefully to a ~4-chars/token estimate when no tokenizer
is supplied.

> Note: the 4096 window is intrinsic to Llama-2. It can only be *extended* with
> RoPE scaling (`rope_freq_scale` in llama.cpp), which trades quality for length.
> Sliding-window management is the reliable fix for the original weights.

## The modern workflow (recommended)

The limitation is a Llama-2-era artifact. Modern models served through **Ollama**
have far larger contexts (8k–128k+):

| Model | Context |
|---|---|
| `qwen3` | 32k+ |
| `deepseek-r1` | 64k+ |
| `gpt-oss:120b-cloud` | large (cloud) |

The **[Codephreak AI SDK console](../docs/CODEPHREAK_CONSOLE.md)** and
`ollama_codephreak.py` use these models, and the console exposes **`num_ctx`**
(512–32,768) as a live Scientific setting — so you simply size the window to the
model. For most work, **move to a modern model** and set `num_ctx` appropriately;
keep `context4096.py` for running the authentic original codephreak weights.
