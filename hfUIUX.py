# hfUIUX.py
# all hugging face version start with hf
#
# LEGACY HEAVY PATH: loads a full transformers model (microsoft/phi-2) in-process.
# This needs transformers + torch and several GB of RAM, and the UI does not
# appear until the model finishes loading. For an entrypoint that loads instantly
# on modest hardware, use ollama_codephreak.py instead.
import gradio as gr
from automind import format_to_llama_chat_style
from memory import save_conversation_memory
from aglm import LlamaModel

MODEL_ID = "microsoft/phi-2"

# Lazily instantiate so importing this module (and the Space build) doesn't block
# on a multi-GB model download — the previous module-level LlamaModel(MODEL_ID)
# crashed at import time (and passed the wrong number of arguments).
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = LlamaModel(MODEL_ID)  # models_folder defaults; HF id resolved directly
    return _llm


def chat(user_input, history=None):
    # gradio ChatInterface passes history as [[user, assistant], ...].
    history = history or []
    memory = [list(turn) for turn in history] + [[user_input, None]]
    # generate_contextual_output() already applies format_to_llama_chat_style;
    # pass the raw memory list (the old code double-formatted a pre-built prompt).
    response = get_llm().generate_contextual_output(memory)
    save_conversation_memory([[user_input, response]])
    return response


iface = gr.ChatInterface(
    fn=chat,
    title="Professor Codephreak (automindx demo)",
    description="Professor Codephreak (aka codephreak) is your expert in ML, computer science, open source, and secure programming. Free forever on Hugging Face Spaces."
)

if __name__ == "__main__":
    iface.launch()
