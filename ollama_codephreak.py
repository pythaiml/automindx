# ollama_codephreak.py
# Professor Codephreak — modern, actually-loads entrypoint for automindx.
#
# The original uiux.py / hfUIUX.py load a multi-GB transformers/llama.cpp model
# before the UI renders, which means the UI never appears on modest hardware.
# This entrypoint inverts that: it talks to a running Ollama daemon (which keeps
# the model warm in its own process), so the Gradio UI loads instantly and
# degrades gracefully when no model/daemon is present.
#
# Local small models (qwen3:0.6b, deepseek-r1:1.5b) run on CPU; ':cloud' models
# (gpt-oss:120b-cloud, glm-5.x:cloud) are proxied by the daemon to ollama.com.
#
#   pip install gradio requests          # see requirements-ollama.txt
#   ollama serve                         # in another terminal
#   python ollama_codephreak.py
#
# Env:
#   OLLAMA_HOST       (default http://localhost:11434)
#   AUTOMINDX_MODEL   (default qwen3:0.6b)

import os
import requests
import gradio as gr

from automind import DEFAULT_SYSTEM_PROMPT
from memory import save_conversation_memory

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AUTOMINDX_MODEL", "qwen3:0.6b")


def list_models():
    """Live list of Ollama models (local + cloud). Empty list if daemon is down."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=4)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        # smallest/local first is friendlier on modest hardware
        names.sort(key=lambda n: (n.endswith(":cloud"), n))
        return names
    except Exception:
        return []


def chat(message, history, model):
    """Stream a Professor Codephreak reply from Ollama, augmented by the persona."""
    model = model or DEFAULT_MODEL
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    for turn in history or []:
        # gradio passes history as [[user, assistant], ...]
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            if turn[0]:
                messages.append({"role": "user", "content": turn[0]})
            if turn[1]:
                messages.append({"role": "assistant", "content": turn[1]})
    messages.append({"role": "user", "content": message})

    try:
        with requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": model, "messages": messages, "stream": True, "think": False},
            stream=True,
            timeout=300,
        ) as r:
            if r.status_code != 200:
                yield _explain_error(r.status_code, r.text, model), ""
                return
            acc = ""
            stats = ""
            import json as _json
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = _json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                acc += (obj.get("message") or {}).get("content", "")
                if obj.get("done"):
                    stats = _format_tokens(obj)
                if acc:
                    yield acc, stats
                if obj.get("done"):
                    break
            if not acc:
                yield "[model returned no content — try a larger model or raise the token budget]", ""
            else:
                # persist the exchange to long-term memory (./memory/*.json)
                try:
                    save_conversation_memory([[message, acc]])
                except Exception:
                    pass
    except requests.exceptions.ConnectionError:
        yield (
            "⚠️ Cannot reach Ollama at "
            f"`{OLLAMA_HOST}`.\n\nStart it with `ollama serve`, then pull a model "
            "(`ollama pull qwen3:0.6b`) and reload."
        ), ""
    except Exception as e:
        yield f"⚠️ Error talking to the model: {e}", ""


def _format_tokens(done_obj):
    """Build a token-counter line from Ollama's final stream object."""
    prompt = done_obj.get("prompt_eval_count", 0) or 0
    completion = done_obj.get("eval_count", 0) or 0
    total = prompt + completion
    eval_ns = done_obj.get("eval_duration", 0) or 0
    rate = f" · {completion / (eval_ns / 1e9):.1f} tok/s" if eval_ns else ""
    return f"🪙 **{total} tokens** — {prompt} prompt + {completion} completion{rate}"


def _explain_error(status, body, model):
    if status == 403 and "subscription" in body.lower():
        return (
            f"☁️ **{model}** is a cloud model that requires an Ollama subscription "
            "(https://ollama.com/upgrade). Pick a local model, or use a free cloud "
            "model such as `gpt-oss:120b-cloud`."
        )
    if status == 404:
        return f"Model **{model}** not found. Pull it first: `ollama pull {model}`."
    return f"Ollama returned HTTP {status}: {body[:200]}"


def build_ui():
    """Version-robust Blocks chatbot (works on gradio 3.x and 4.x) with live
    token streaming and a model picker."""
    models = list_models()
    default = DEFAULT_MODEL if (not models or DEFAULT_MODEL in models) else models[0]
    with gr.Blocks(title="Professor Codephreak · automindx") as demo:
        gr.Markdown(
            "# Professor Codephreak — automindx\n"
            "Local & cloud language models via **Ollama**. AGLM augments the model "
            "with the codephreak persona and reasoning. The UI loads instantly; "
            "the model runs in the Ollama daemon."
        )
        model_dd = gr.Dropdown(
            choices=models or [DEFAULT_MODEL],
            value=default,
            label="Model (local · or :cloud)",
            allow_custom_value=True,
        )
        if not models:
            gr.Markdown(
                "> ⚠️ No Ollama models detected. Start `ollama serve` and "
                "`ollama pull qwen3:0.6b`, then reload."
            )
        chatbot = gr.Chatbot(height=440, label="codephreak")
        tokens = gr.Markdown("🪙 0 tokens", elem_id="token-counter")
        msg = gr.Textbox(placeholder="Ask Professor Codephreak…", show_label=False)
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")

        def user(message, chat_history):
            if not message.strip():
                return "", chat_history
            return "", (chat_history or []) + [[message, None]]

        def bot(chat_history, model):
            message = chat_history[-1][0]
            history = chat_history[:-1]
            chat_history[-1][1] = ""
            stats = "🪙 streaming…"
            for partial, s in chat(message, history, model):
                chat_history[-1][1] = partial
                stats = s or stats
                yield chat_history, stats

        for trigger in (msg.submit, send.click):
            trigger(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [chatbot, model_dd], [chatbot, tokens]
            )
        clear.click(lambda: (None, "🪙 0 tokens"), None, [chatbot, tokens], queue=False)

        gr.Examples(
            examples=[
                "How can I implement zero-knowledge proofs in Python?",
                "Explain the BDI agent model in three sentences.",
            ],
            inputs=msg,
        )
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="0.0.0.0", server_port=7860)
