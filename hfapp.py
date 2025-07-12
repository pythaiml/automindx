# hfapp.py
# Professor Codephreak ancestor=mode
# 1 cpu + 16gb ram
import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

MODEL_REPO = "TheBloke/llama2-7b-chat-codeCherryPop-qLoRA-GGML"
MODEL_FILE = "llama-2-7b-chat-codeCherryPop.ggmlv3.q4_1.bin"

def ensure_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading model from Hugging Face Hub...")
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir="."
        )
    else:
        print(f"Model already present: {MODEL_FILE}")

ensure_model()
llm = Llama(model_path=MODEL_FILE, n_ctx=2048)

SYSTEM_PROMPT = (
    "You are Professor Codephreak, your digital ancestor and master of machine learning, computer science, and modular secure code. "
    "You always answer as 'codephreak', with step-by-step logic and practical solutions."
)

def chat(user_input, history=None):
    if history is None or len(history) == 0:
        history = [(
            "",
            "Greetings, seeker. I am **Professor Codephreak**, your digital ancestor—"
            "an original pioneer in computer science, machine learning, and secure, modular code. "
            "Ask your deepest technical questions and receive step-by-step, codephreak wisdom. "
            "⚠️ *As your ancestor, my responses may take several minutes on CPU. Upgrade hardware for faster insights.*"
        )]
    prompt = SYSTEM_PROMPT + "\n"
    for u, a in history:
        if u or a:  # skip empty initial
            prompt += f"User: {u}\ncodephreak: {a}\n"
    prompt += f"User: {user_input}\ncodephreak:"
    output = llm(prompt, max_tokens=256, stop=["User:", "\ncodephreak:"])
    response = output["choices"][0]["text"].strip()
    history.append((user_input, response))
    return response, history

iface = gr.ChatInterface(
    fn=chat,
    title="Professor Codephreak (automindx, GGML, CPU demo)",
    description=(
        "⚠️ This Space runs the authentic automindx GGML model on Hugging Face's free CPU Basic tier. "
        "Each answer may take 2–10 minutes. Upgrade for faster expert insight."
    ),
    examples=[["How can I implement zero-knowledge proofs in Python?"]]
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
