# hfUIUX.py
# all hugging face version start with hf
import gradio as gr
from automind import format_to_llama_chat_style
from memory import save_conversation_memory, load_conversation_memory
from aglm import LlamaModel

MEMORY_FILE = "/tmp/codephreak_history.json"
MODEL_ID = "microsoft/phi-2"

llm = LlamaModel(MODEL_ID)

def chat(user_input, history=None):
    if history is None:
        history = load_conversation_memory(MEMORY_FILE)
    prompt = format_to_llama_chat_style(history, user_input)
    response = llm.generate_contextual_output(prompt)
    history.append((user_input, response))
    save_conversation_memory(history, MEMORY_FILE)
    return response, history

iface = gr.ChatInterface(
    fn=chat,
    title="Professor Codephreak (automindx demo)",
    description="Professor Codephreak (aka codephreak) is your expert in ML, computer science, open source, and secure programming. Free forever on Hugging Face Spaces."
)

if __name__ == "__main__":
    iface.launch()
