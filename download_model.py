"""
Download all models for the chatbot.
Run: python download_model.py

Downloads:
  1. Multilingual-E5-Small ONNX (~130 MB) — multilingual semantic embedding
  2. MiniLM-L6-v2 ONNX (~87 MB) — legacy English embedding (fallback)
  3. LLM: TinyLlama 1.1B OR Phi-3 Mini 3.8B (user chooses)
  4. Cross-encoder reranker ONNX (~80 MB) — optional, improves search precision
"""

import os
import sys
from huggingface_hub import hf_hub_download


def download_hf_files(repo_id, files, local_dir):
    """Download a list of files from HuggingFace Hub."""
    for filename in files:
        print(f"  {filename}...", end=" ", flush=True)
        subfolder = None
        local_name = filename
        if "/" in filename:
            subfolder = filename.rsplit("/", 1)[0]
            local_name = filename.rsplit("/", 1)[1]
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=local_name,
                subfolder=subfolder,
                local_dir=local_dir,
            )
            print("done")
        except Exception as e:
            print(f"FAILED ({e})")


def download_embeddings():
    """Download both embedding models."""
    # Multilingual-E5-Small
    MULTILINGUAL_DIR = "models/multilingual-e5"
    os.makedirs(os.path.join(MULTILINGUAL_DIR, "onnx"), exist_ok=True)
    onnx_path = os.path.join(MULTILINGUAL_DIR, "onnx", "model.onnx")

    if os.path.exists(onnx_path):
        print("\n[1/4] Multilingual-E5-Small: already exists, skipping")
    else:
        print("\n[1/4] Downloading Multilingual-E5-Small (~130 MB)...")
        print("       50+ languages including Bangla — best for multilingual chatbot")
        download_hf_files("intfloat/multilingual-e5-small", [
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "config.json", "onnx/model.onnx",
        ], MULTILINGUAL_DIR)

    # MiniLM-L6-v2 (legacy fallback)
    MINILM_DIR = "models/minilm"
    os.makedirs(os.path.join(MINILM_DIR, "onnx"), exist_ok=True)
    minilm_path = os.path.join(MINILM_DIR, "onnx", "model.onnx")

    if os.path.exists(minilm_path):
        print("\n[2/4] MiniLM-L6-v2 (legacy): already exists, skipping")
    else:
        print("\n[2/4] Downloading MiniLM-L6-v2 (~87 MB)... (legacy fallback)")
        download_hf_files("sentence-transformers/all-MiniLM-L6-v2", [
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.txt", "config.json", "onnx/model.onnx",
        ], MINILM_DIR)


def choose_and_download_llm():
    """Let user choose which LLM to download."""
    print("\n" + "=" * 60)
    print("  Choose LLM Model for Text Generation")
    print("=" * 60)
    print()
    print("  [1] TinyLlama 1.1B  (Q4_K_M)")
    print("      Size: ~638 MB  |  RAM: ~2 GB")
    print("      Speed: Fast  |  Quality: Basic")
    print("      Best for: Low-end PCs, 4-8 GB RAM")
    print("      Answers are short, sometimes repetitive")
    print()
    print("  [2] Phi-3 Mini 3.8B  (Q3_K_M)")
    print("      Size: ~1.96 GB  |  RAM: ~4 GB")
    print("      Speed: Medium  |  Quality: Good (3x better than TinyLlama)")
    print("      Best for: 8-16 GB RAM PCs")
    print("      Smarter answers, understands context better")
    print()
    print("  [3] Both (download both, switch in config later)")
    print()
    print("  [0] Skip (don't download any LLM)")
    print()

    choice = input("  Your choice [1/2/3/0]: ").strip()

    # TinyLlama
    TINYLLAMA_DIR = "models/tinyllama"
    TINYLLAMA_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    os.makedirs(TINYLLAMA_DIR, exist_ok=True)
    tinyllama_path = os.path.join(TINYLLAMA_DIR, TINYLLAMA_FILE)

    # Phi-3
    PHI3_DIR = "models/phi3"
    PHI3_FILE = "Phi-3-mini-4k-instruct-Q3_K_M.gguf"
    os.makedirs(PHI3_DIR, exist_ok=True)
    phi3_path = os.path.join(PHI3_DIR, PHI3_FILE)

    if choice in ("1", "3"):
        if os.path.exists(tinyllama_path):
            print(f"\n[3/4] TinyLlama 1.1B: already exists, skipping")
        else:
            print(f"\n[3/4] Downloading TinyLlama 1.1B (~638 MB)...")
            print(f"  {TINYLLAMA_FILE}...", end=" ", flush=True)
            hf_hub_download(
                repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                filename=TINYLLAMA_FILE,
                local_dir=TINYLLAMA_DIR,
            )
            print("done")

    if choice in ("2", "3"):
        if os.path.exists(phi3_path):
            print(f"\n[3/4] Phi-3 Mini 3.8B: already exists, skipping")
        else:
            print(f"\n[3/4] Downloading Phi-3 Mini 3.8B (~1.96 GB)...")
            print(f"  {PHI3_FILE}...", end=" ", flush=True)
            hf_hub_download(
                repo_id="bartowski/Phi-3-mini-4k-instruct-GGUF",
                filename=PHI3_FILE,
                local_dir=PHI3_DIR,
            )
            print("done")

    if choice == "0":
        print("\n[3/4] LLM download skipped")

    if choice not in ("0", "1", "2", "3"):
        print(f"\n[3/4] Invalid choice '{choice}', skipping LLM download")

    # Print what's available
    print("\n  LLM Status:")
    print(f"    TinyLlama: {'READY' if os.path.exists(tinyllama_path) else 'not downloaded'}")
    print(f"    Phi-3:     {'READY' if os.path.exists(phi3_path) else 'not downloaded'}")


def download_reranker():
    """Download cross-encoder reranker (optional)."""
    RERANKER_DIR = "models/reranker"
    os.makedirs(os.path.join(RERANKER_DIR, "onnx"), exist_ok=True)
    onnx_path = os.path.join(RERANKER_DIR, "onnx", "model.onnx")

    if os.path.exists(onnx_path):
        print(f"\n[4/4] Cross-Encoder Reranker: already exists, skipping")
    else:
        print(f"\n[4/4] Downloading Cross-Encoder Reranker (~80 MB)...")
        print("       Improves search precision (optional but recommended)")
        download_hf_files("cross-encoder/ms-marco-MiniLM-L-6-v2", [
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "config.json", "onnx/model.onnx",
        ], RERANKER_DIR)

        if os.path.exists(onnx_path):
            print("  Reranker ready!")
        else:
            print("  Reranker ONNX not available — will skip reranking (optional)")


def print_summary():
    """Print final status of all models."""
    print("\n" + "=" * 60)
    print("  Download Complete — Model Status")
    print("=" * 60)

    models = [
        ("Multilingual-E5", "models/multilingual-e5/onnx/model.onnx", "Embedding (50+ languages)"),
        ("MiniLM-L6-v2", "models/minilm/onnx/model.onnx", "Embedding (English, legacy)"),
        ("TinyLlama 1.1B", "models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "LLM (638MB, basic)"),
        ("Phi-3 Mini 3.8B", "models/phi3/Phi-3-mini-4k-instruct-Q3_K_M.gguf", "LLM (1.96GB, smart)"),
        ("Reranker", "models/reranker/onnx/model.onnx", "Search precision (optional)"),
    ]

    for name, path, desc in models:
        status = "READY" if os.path.exists(path) else "---"
        icon = "+" if status == "READY" else " "
        print(f"  [{icon}] {name:20s} | {status:5s} | {desc}")

    print("=" * 60)
    print("\nRun: python app.py")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Mini Chatbot — Model Downloader")
    print("=" * 60)

    download_embeddings()
    choose_and_download_llm()
    download_reranker()
    print_summary()
