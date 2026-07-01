FROM python:3.10-slim

WORKDIR /workspace

# Install system requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all source files
COPY . .

# Set environment variables for Hugging Face cache (optional, but helps with cold start)
ENV HF_HOME=/workspace/.cache/huggingface

# Expose Gradio default port
EXPOSE 7860

# Entrypoint — hfapp.py is the self-contained GGML (llama.cpp) demo that
# downloads its model on first run. (The previous "app.py" did not exist.)
CMD ["python", "hfapp.py"]
