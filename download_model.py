"""
Download all models for the chatbot.
Run this once on a new machine: python download_model.py

Downloads:
  1. MiniLM-L6-v2 ONNX (~87 MB) — semantic embedding
  2. TinyLlama 1.1B Q4_K_M GGUF (~638 MB) — text generation
"""

import os
import sys
from huggingface_hub import hf_hub_download

# ============================================================
# 1. MiniLM-L6-v2 (Semantic Encoder)
# ============================================================
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIR = "models/minilm"
ONNX_DIR = os.path.join(MINILM_DIR, "onnx")

os.makedirs(ONNX_DIR, exist_ok=True)

minilm_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "config.json",
    "onnx/model.onnx",
]

print("=" * 50)
print("  Model Downloader")
print("=" * 50)

# Check if MiniLM already exists
minilm_onnx_path = os.path.join(MINILM_DIR, "onnx", "model.onnx")
if os.path.exists(minilm_onnx_path):
    print(f"\n[1/2] MiniLM-L6-v2: already exists, skipping")
else:
    print(f"\n[1/2] Downloading MiniLM-L6-v2 (~87 MB) to {MINILM_DIR}/...")
    for filename in minilm_files:
        print(f"  {filename}...", end=" ", flush=True)
        subfolder = None
        local_name = filename
        if "/" in filename:
            subfolder = filename.rsplit("/", 1)[0]
            local_name = filename.rsplit("/", 1)[1]

        hf_hub_download(
            repo_id=MINILM_ID,
            filename=local_name,
            subfolder=subfolder,
            local_dir=MINILM_DIR,
        )
        print("done")
    print("  MiniLM ready!")

# ============================================================
# 2. TinyLlama 1.1B (Text Generation)
# ============================================================
TINYLLAMA_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
TINYLLAMA_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
TINYLLAMA_DIR = "models/tinyllama"

os.makedirs(TINYLLAMA_DIR, exist_ok=True)

tinyllama_path = os.path.join(TINYLLAMA_DIR, TINYLLAMA_FILE)
if os.path.exists(tinyllama_path):
    print(f"\n[2/2] TinyLlama 1.1B: already exists, skipping")
else:
    print(f"\n[2/2] Downloading TinyLlama 1.1B Q4_K_M (~638 MB) to {TINYLLAMA_DIR}/...")
    print(f"  {TINYLLAMA_FILE}...", end=" ", flush=True)
    hf_hub_download(
        repo_id=TINYLLAMA_ID,
        filename=TINYLLAMA_FILE,
        local_dir=TINYLLAMA_DIR,
    )
    print("done")
    print("  TinyLlama ready!")

# ============================================================
# Done
# ============================================================
print("\n" + "=" * 50)
print("  All models downloaded!")
print(f"  MiniLM:    {MINILM_DIR}/onnx/model.onnx")
print(f"  TinyLlama: {TINYLLAMA_DIR}/{TINYLLAMA_FILE}")
print("=" * 50)
print("\nRun: python app.py")
