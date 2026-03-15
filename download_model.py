"""
Download MiniLM-L6-v2 ONNX model for the chatbot.
Run this once on a new machine: python download_model.py
"""

import os
from huggingface_hub import hf_hub_download

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = "models/minilm"
ONNX_DIR = os.path.join(MODEL_DIR, "onnx")

os.makedirs(ONNX_DIR, exist_ok=True)

files = {
    "tokenizer.json": MODEL_DIR,
    "tokenizer_config.json": MODEL_DIR,
    "special_tokens_map.json": MODEL_DIR,
    "vocab.txt": MODEL_DIR,
    "config.json": MODEL_DIR,
    "onnx/model.onnx": MODEL_DIR,
}

print(f"Downloading MiniLM-L6-v2 ONNX model to {MODEL_DIR}/...")
for filename, local_dir in files.items():
    print(f"  {filename}...", end=" ", flush=True)
    subfolder = None
    local_name = filename
    if "/" in filename:
        subfolder = filename.rsplit("/", 1)[0]
        local_name = filename.rsplit("/", 1)[1]

    path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=local_name,
        subfolder=subfolder,
        local_dir=MODEL_DIR,
    )
    print("done")

print(f"\nModel downloaded to {MODEL_DIR}/")
print("You can now run: python app.py")
