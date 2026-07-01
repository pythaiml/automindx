# chunk4096.py
import gradio as gr
import torch
from transformers import AutoTokenizer

# Constants and Variables
BOS, EOS = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# Define the chunk size as a class attribute
class Chunker:
    CHUNK_SIZE = 4096  # This line defines the chunk size for the class

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def chunk_and_generate(self, chunk_text):
        inputs = self.tokenizer(chunk_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def chunk_and_generate_response(chunker, text):
    chunks = [text[i:i + Chunker.CHUNK_SIZE] for i in range(0, len(text), Chunker.CHUNK_SIZE)]
    response = ""
    for chunk in chunks:
        response += chunker.chunk_and_generate(chunk)
    return response

# ... Other functions from uiux.py that you want to reuse ...

# Uncomment the following line if you want to make this script standalone executable
# if __name__ == '__main__':
#     fire.Fire(main)

