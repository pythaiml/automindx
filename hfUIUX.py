# hfUIUX.py
# all hugging face version start with hf
#
# LEGACY HEAVY PATH: loads a full transformers model (microsoft/phi-2) in-process.
# This needs transformers + torch and several GB of RAM, and the UI does not
# appear until the model finishes loading. For an entrypoint that loads instantly
# on modest hardware, use ollama_codephreak.py instead.
import os
import sys
import gradio as gr
from automind import format_to_llama_chat_style, DEFAULT_SYSTEM_PROMPT
from memory import save_conversation_memory
from llama_model import LlamaModel
# The 4096 context tooling lives in the 4096/ folder (grouped with its docs).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "4096"))
from context4096 import ContextWindow

MODEL_ID = "microsoft/phi-2"
N_CTX = 4096  # the original codephreak (Llama-2) hard context window

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
    llm = get_llm()
    # Token-aware sliding window: keep the system prompt + newest turns that fit
    # in 4096 tokens so long conversations never overflow the context (the proper
    # fix for the old character-based chunk4096.py). See context4096.py.
    window = ContextWindow(llm.tokenizer, n_ctx=N_CTX, reserve=512)
    memory = window.fit(memory, DEFAULT_SYSTEM_PROMPT)
    # generate_contextual_output() already applies format_to_llama_chat_style;
    # pass the raw memory list (the old code double-formatted a pre-built prompt).
    response = llm.generate_contextual_output(memory)
    save_conversation_memory([[user_input, response]])
    return response


iface = gr.ChatInterface(
    fn=chat,
    title="Professor Codephreak (automindx demo)",
    description="Professor Codephreak (aka codephreak) is your expert in ML, computer science, open source, and secure programming. Free forever on Hugging Face Spaces."
)

if __name__ == "__main__":
    iface.launch()
