chunk4096.py
This script provides chunking functionality for handling input text longer than the model's limit of 4096 tokens. It uses the Chunker class to break down input text into smaller chunks and generate responses using a provided model and tokenizer.<br />

Dependencies<br />
Gradio: A Python library for building and sharing machine learning models through a user-friendly interface.<br />
PyTorch: An open-source machine learning framework.<br />
Fire: A library for creating command-line interfaces from Python functions.<br />
Transformers: A library for natural language processing and machine learning, including pre-trained language models.<br />
Enum: A built-in Python module for creating enumerated constants.<br />
Threading: A built-in Python module for concurrent execution.<br />
