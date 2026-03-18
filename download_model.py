"""
Download all models for the chatbot.
Run this once on a new machine: python download_model.py

Downloads:
  1. Multilingual-E5-Small ONNX (~130 MB) — multilingual semantic embedding
  2. TinyLlama 1.1B Q4_K_M GGUF (~638 MB) — text generation
  3. Cross-encoder reranker ONNX (~80 MB) — optional, improves search precision

Legacy: MiniLM-L6-v2 still supported if already downloaded.
"""

import os
from huggingface_hub import hf_hub_download

print("=" * 50)
print("  Model Downloader")
print("=" * 50)

# ============================================================
# 1. Multilingual-E5-Small (Semantic Encoder — replaces MiniLM)
# Supports 50+ languages including Bangla, same 384-dim output
# ============================================================
MULTILINGUAL_ID = "intfloat/multilingual-e5-small"
MULTILINGUAL_DIR = "models/multilingual-e5"
MULTILINGUAL_ONNX_DIR = os.path.join(MULTILINGUAL_DIR, "onnx")

os.makedirs(MULTILINGUAL_ONNX_DIR, exist_ok=True)

multilingual_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "onnx/model.onnx",
]

multilingual_onnx_path = os.path.join(MULTILINGUAL_DIR, "onnx", "model.onnx")
if os.path.exists(multilingual_onnx_path):
    print(f"\n[1/3] Multilingual-E5-Small: already exists, skipping")
else:
    print(f"\n[1/3] Downloading Multilingual-E5-Small (~130 MB) to {MULTILINGUAL_DIR}/...")
    for filename in multilingual_files:
        print(f"  {filename}...", end=" ", flush=True)
        subfolder = None
        local_name = filename
        if "/" in filename:
            subfolder = filename.rsplit("/", 1)[0]
            local_name = filename.rsplit("/", 1)[1]

        hf_hub_download(
            repo_id=MULTILINGUAL_ID,
            filename=local_name,
            subfolder=subfolder,
            local_dir=MULTILINGUAL_DIR,
        )
        print("done")
    print("  Multilingual-E5-Small ready!")

# ============================================================
# 1b. MiniLM-L6-v2 (Legacy — still download if not present)
# ============================================================
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIR = "models/minilm"
MINILM_ONNX_DIR = os.path.join(MINILM_DIR, "onnx")

os.makedirs(MINILM_ONNX_DIR, exist_ok=True)

minilm_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "config.json",
    "onnx/model.onnx",
]

minilm_onnx_path = os.path.join(MINILM_DIR, "onnx", "model.onnx")
if os.path.exists(minilm_onnx_path):
    print(f"\n[1b]  MiniLM-L6-v2 (legacy): already exists, skipping")
else:
    print(f"\n[1b]  Downloading MiniLM-L6-v2 (~87 MB) to {MINILM_DIR}/... (legacy fallback)")
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
    print(f"\n[2/3] TinyLlama 1.1B: already exists, skipping")
else:
    print(f"\n[2/3] Downloading TinyLlama 1.1B Q4_K_M (~638 MB) to {TINYLLAMA_DIR}/...")
    print(f"  {TINYLLAMA_FILE}...", end=" ", flush=True)
    hf_hub_download(
        repo_id=TINYLLAMA_ID,
        filename=TINYLLAMA_FILE,
        local_dir=TINYLLAMA_DIR,
    )
    print("done")
    print("  TinyLlama ready!")

# ============================================================
# 3. Cross-Encoder Reranker (Optional — improves precision)
# ============================================================
RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_DIR = "models/reranker"
RERANKER_ONNX_DIR = os.path.join(RERANKER_DIR, "onnx")

os.makedirs(RERANKER_ONNX_DIR, exist_ok=True)

reranker_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "onnx/model.onnx",
]

reranker_onnx_path = os.path.join(RERANKER_DIR, "onnx", "model.onnx")
if os.path.exists(reranker_onnx_path):
    print(f"\n[3/3] Cross-Encoder Reranker: already exists, skipping")
else:
    print(f"\n[3/3] Downloading Cross-Encoder Reranker (~80 MB) to {RERANKER_DIR}/...")
    for filename in reranker_files:
        print(f"  {filename}...", end=" ", flush=True)
        subfolder = None
        local_name = filename
        if "/" in filename:
            subfolder = filename.rsplit("/", 1)[0]
            local_name = filename.rsplit("/", 1)[1]

        try:
            hf_hub_download(
                repo_id=RERANKER_ID,
                filename=local_name,
                subfolder=subfolder,
                local_dir=RERANKER_DIR,
            )
            print("done")
        except Exception as e:
            print(f"skipped ({e})")

    if os.path.exists(reranker_onnx_path):
        print("  Reranker ready!")
    else:
        print("  Reranker ONNX not available — will skip reranking (optional feature)")

# ============================================================
# Done
# ============================================================
print("\n" + "=" * 50)
print("  All models downloaded!")
print(f"  Multilingual-E5: {MULTILINGUAL_DIR}/onnx/model.onnx")
print(f"  MiniLM (legacy): {MINILM_DIR}/onnx/model.onnx")
print(f"  TinyLlama:       {TINYLLAMA_DIR}/{TINYLLAMA_FILE}")
print(f"  Reranker:        {RERANKER_DIR}/onnx/model.onnx")
print("=" * 50)
print("\nRun: python app.py")
