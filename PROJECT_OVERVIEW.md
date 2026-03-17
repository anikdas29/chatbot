# Mini Chatbot — Developer Overview

Fully offline NLP + ML chatbot with semantic search, local LLM generation, and self-learning.
No third-party AI API needed. Designed for companies to plug in their own Q&A dataset.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Backend** | Python 3.13, Flask 3.0 | REST API + Web UI server |
| **Semantic Search** | ONNX MiniLM-L6-v2 (384-dim) | Converts text to meaning vectors |
| **Vector Index** | FAISS IndexFlatIP | Sub-millisecond cosine similarity search over 6K+ vectors |
| **ML Classifier** | scikit-learn LogisticRegression + TF-IDF | Secondary classification signal |
| **LLM Generation** | TinyLlama 1.1B (Q4_K_M GGUF via ctransformers) | Hybrid RAG answer generation |
| **Spell Correction** | difflib (length-guarded, adaptive cutoff) | Typo fixing without destroying valid terms |
| **Database** | SQLite (indexed) | Sessions, conversation history, feedback, learning queue |
| **Frontend** | HTML + CSS + Vanilla JS | Dark-theme chat UI with feedback buttons |
| **Dataset** | 1,094 category JSON files (6,479 questions) | Two stores: general (1,029) + coding (65) |

---

## Models Used

### 1. MiniLM-L6-v2 (Semantic Encoder)
- **What:** Sentence-transformers model exported to ONNX format
- **Size:** ~87 MB (`models/minilm/`)
- **Output:** 384-dimensional normalized embedding vectors
- **Purpose:** Converts any text into a meaning vector. "feeling sad" and "i am depressed" produce nearly identical vectors even though they share no words.
- **Runtime:** ONNX Runtime (CPU) — no GPU needed
- **Files:** `model.onnx`, `tokenizer.json`, `vocab.txt`, `config.json`

### 2. TinyLlama 1.1B Chat (RAG Generator)
- **What:** 1.1 billion parameter LLM, quantized to 4-bit (Q4_K_M GGUF)
- **Size:** ~638 MB (`models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`)
- **Purpose:** Generates natural-sounding answers by reasoning over retrieved Q&A pairs from the dataset (Hybrid RAG). Instead of returning pre-written answers verbatim, the LLM synthesizes a fresh response grounded in actual dataset knowledge.
- **Runtime:** ctransformers (C++ backend, CPU inference)
- **Config:** `max_new_tokens=150`, `temperature=0.4`, `top_p=0.85`, `context_length=1024`
- **Optional:** Bot works without it (falls back to dataset answers with templates)

---

## Libraries / Dependencies

| Package | Version | Purpose |
|---|---|---|
| `flask` | >=3.0 | Web server + REST API |
| `scikit-learn` | >=1.3 | TF-IDF vectorizer + Logistic Regression classifier |
| `numpy` | >=1.24 | Array math, embedding operations |
| `onnxruntime` | >=1.16 | Run MiniLM ONNX model on CPU |
| `tokenizers` | >=0.15 | HuggingFace fast tokenizer for MiniLM |
| `huggingface-hub` | >=0.20 | Model download utility |
| `ctransformers` | >=0.2.27 | Load and run GGUF LLM (TinyLlama) |
| `faiss-cpu` | >=1.7 | Facebook AI Similarity Search — fast vector index |

Built-in: `json`, `sqlite3`, `re`, `difflib`, `logging`, `os`, `random`, `uuid`, `time`

---

## File Structure

```
chatbot/
├── chatbot.py                 # Core engine (1,865 lines) — all NLP/ML/RAG logic
├── app.py                     # Flask API server (167 lines)
├── requirements.txt           # Python dependencies
├── CLAUDE.md                  # AI assistant instructions for dataset management
├── PROJECT_OVERVIEW.md        # This file
│
├── models/
│   ├── minilm/                # ONNX MiniLM-L6-v2 (~87 MB)
│   │   ├── onnx/model.onnx
│   │   ├── tokenizer.json
│   │   └── vocab.txt, config.json, ...
│   └── tinyllama/             # TinyLlama 1.1B GGUF (~638 MB)
│       └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
│
├── category_wise_dataset/     # General knowledge (1,029 category JSON files)
├── coding_dataset/            # Programming topics (65 category JSON files)
│
├── templates/
│   └── index.html             # Chat Web UI
├── static/
│   └── style.css              # Dark theme styles
│
├── chatbot.db                 # SQLite database (sessions, feedback, learning)
├── chatbot.log                # Runtime logs
├── feedback.json              # Legacy feedback file
│
├── dataset.json               # Legacy flat dataset (superseded by category folders)
├── converter.py               # Dataset format converter (CSV/JSON/JSONL/TXT)
├── build_dataset.py           # Dataset builder from Kaggle CSV
├── build_clean.py             # Clean dataset builder from cached intents
├── download_model.py          # Model download helper
│
├── test_deep_analysis.py      # 60-question accuracy test (8 groups)
├── test_100q.py               # 100-question randomized test
└── test_500.py                # 500-question stress test
```

---

## Dataset Format

Each category is a separate JSON file in `category_wise_dataset/` or `coding_dataset/`:

```json
{
    "category": "python",
    "type": "general",
    "tags": ["python", "programming", "coding", "scripting"],
    "questions": [
        "what is python?",
        "tell me about python",
        "python ki?",
        "python programming language"
    ],
    "answers": [
        "Python is a high-level, interpreted programming language known for its simplicity.",
        "Python is one of the most popular languages, used in web dev, AI, data science, and more.",
        "Python holo ekta beginner-friendly language ja diye web, AI, ar data science kaj kora jay."
    ],
    "feelings": []
}
```

- **`tags`** (optional): Many-to-many mapping. A question about "python flask deployment" can match categories tagged with `python`, `flask`, and `deployment`.
- **`feelings`**: Tracks sentiment of user interactions (auto-populated).

**Stats:** 1,094 categories, 6,479 questions, across 2 dataset stores.

---

## How It Works — Full Pipeline

```
User types: "hwo to lern pytohn?"
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Text Cleaning                                  │
│  "hwo to lern pytohn?" → "hwo to lern pytohn"          │
│  (lowercase, strip punctuation)                         │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2: Follow-Up Detection                            │
│  Check if this is a follow-up like "tell me more" or    │
│  pronoun reference like "what about it?" — if so,       │
│  reuse the last discussed category from SQLite history. │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: Spell Correction (Length-Guarded)               │
│  "hwo to lern pytohn" → "hwo to learn python"           │
│                                                          │
│  Adaptive cutoff:                                        │
│    3-4 chars → 0.88 (protects: sql, api, npm, git)      │
│    5-6 chars → 0.82 (catches: pytohn → python)           │
│    7+ chars  → 0.78 (catches: javscript → javascript)    │
│  Length guard: rejects if candidate differs by >2 chars  │
│  Tries BOTH corrected AND original text (dual-path)      │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4: Multi-Category Detection (4 signals combined)   │
│                                                          │
│  Signal 1: FAISS Semantic Search                         │
│    Encode question → 384-dim vector via MiniLM            │
│    Search FAISS index (6,479 vectors) → top-15 matches   │
│    Aggregate by category → keep max score per category    │
│                                                          │
│  Signal 2: ML Classifier                                 │
│    TF-IDF + LogisticRegression → top-3 predictions       │
│    Probability per category                               │
│                                                          │
│  Signal 3: Intent Signal Detection                       │
│    22 regex patterns catch meaning ONNX misses            │
│    "i dont know what to do" → career, motivation          │
│    "keno hocche na" → debugging (Bangla patterns too)     │
│                                                          │
│  Signal 4: Tag-Based Discovery                           │
│    Extract words from question, look up in tag index      │
│    Find categories that tagged themselves with matches     │
│                                                          │
│  → All 4 signals feed into ConfidenceScorer               │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5: Multi-Factor Confidence Scorer                  │
│                                                          │
│  Weighted combination:                                    │
│    45% semantic + 25% ML + 15% intent + 10% agreement    │
│    + 5% score gap                                        │
│                                                          │
│  → Sigmoid normalization: 1/(1+exp(-12*(x-0.38)))        │
│  → Calibrated 5%-98% "sureness" score                    │
│                                                          │
│  Example outputs:                                        │
│    Exact match + all agree    → 98%                      │
│    Strong semantic only       → 67%                      │
│    Moderate, ML confirms      → 76%                      │
│    Vague question             → 51%                      │
│    Weak/ambiguous             → 32%                      │
└─────────────────────┬───────────────────────────────────┘
                      ▼
          ┌───────────┴───────────┐
          │ Categories detected?  │
          └───┬───────────┬───────┘
           No │           │ Yes (up to 3)
              ▼           ▼
┌────────────────┐  ┌──────────────────────────────────────┐
│ "Did you mean?"│  │  STEP 6: Answer Retrieval             │
│ Show top-3     │  │  Find best answer from EACH category  │
│ suggestions    │  │  using semantic similarity between     │
│ OR             │  │  question and stored answers           │
│ Show learn form│  │                                       │
└────────────────┘  │  Multi-category? Merge answers:        │
                    │  Primary → full answer                 │
                    │  Secondary → first sentence + connector│
                    └──────────────┬────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 7: Hybrid RAG Generation (if TinyLlama available)  │
│                                                          │
│  1. Retrieve top-5 Q&A pairs from FAISS (RAG context)    │
│  2. Fetch last 5 conversation turns (sliding window)     │
│  3. Build prompt:                                        │
│     <|system|> Answer using ONLY retrieved knowledge      │
│     <|user|> [5 Q&A pairs] + [conversation history]      │
│              Question: {user's question}                  │
│     <|assistant|>                                        │
│  4. TinyLlama generates 2-3 sentence answer              │
│  5. Clean artifacts ("based on the text" etc.)            │
│                                                          │
│  Skip for: greeting, farewell, thanks (dataset is better)│
│  Fallback: if LLM fails → use dataset answer + templates │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 8: Response Refinement                             │
│                                                          │
│  If dataset answer (not LLM):                            │
│    → AnswerTemplates: add prefix/suffix by sentiment     │
│    → 30% chance append secondary answer snippet          │
│    → SentimentDetector: adjust tone for angry/sad users  │
│                                                          │
│  Store feeling to category file                          │
│  Save turn to SQLite (session, message, reply, intent)   │
└─────────────────────┬───────────────────────────────────┘
                      ▼
              Return JSON response:
              {
                "reply": "Python is a high-level...",
                "confidence": 0.97,
                "categories": ["python"],
                "generated": false
              }
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Chat Web UI |
| `GET` | `/api/health` | Status: question count, categories, feedback stats, pending learns |
| `POST` | `/api/session` | Create new conversation session |
| `POST` | `/api/chat` | Send message → get answer (supports `chosen_category` for suggestions) |
| `POST` | `/api/feedback` | Submit like/dislike + optional correction |
| `POST` | `/api/learn` | Teach bot new Q&A pair |
| `POST` | `/api/process_learns` | Manually flush the pending learning queue |
| `GET` | `/api/pending_learns` | Check learning queue status |

---

## Classes in chatbot.py (1,865 lines)

| Class | Lines | Purpose |
|---|---|---|
| `SemanticEncoder` | ~45 | ONNX MiniLM-L6-v2 — encode text to 384-dim vectors, cosine similarity |
| `TinyLlamaGenerator` | ~115 | Hybrid RAG — build prompt from retrieved Q&A + conversation history, generate via ctransformers |
| `Database` | ~110 | SQLite persistence — sessions, turns, feedback, learn log, pending queue. Indexed for performance. |
| `AnswerTemplates` | ~50 | Response variation — sentiment prefixes, suffixes, 30% secondary answer snippets |
| `SpellCorrector` | ~70 | Length-guarded correction — adaptive cutoff (0.78-0.88), max 2-char length diff, 80+ skip words |
| `SentimentDetector` | ~50 | Keyword-based — angry/sad/happy/curious/confused/neutral. Adjusts response tone. |
| `CategoryStore` | ~180 | Loads category JSON files, manages tags index, questions/answers/feelings CRUD |
| `IntentSignalDetector` | ~90 | 22 regex patterns — catches user intent that semantic search misses (Bangla included) |
| `ConfidenceScorer` | ~90 | Multi-factor sigmoid scorer — 5 signals → weighted sum → sigmoid → calibrated 5-98% |
| `ChatBot` | ~700 | Main engine — init, FAISS index, detection pipeline, RAG, learning queue, answer merging |

---

## Key Design Decisions

1. **Fully Offline** — No API keys, no internet needed at runtime. Both models (MiniLM + TinyLlama) run locally on CPU.

2. **Hybrid RAG, Not Raw LLM** — TinyLlama doesn't hallucinate freely. It receives the top-5 semantically matched Q&A pairs from the dataset as context, so its answers are grounded in actual data.

3. **Multi-Signal Detection** — No single method is trusted alone. Semantic search + ML classifier + regex intent patterns + tag index all vote, then sigmoid normalization produces a calibrated confidence.

4. **Dual-Path Spell Correction** — Tries both corrected AND original text for category detection. Prevents the spell corrector from destroying meaning ("dont know" → "donate now" was a real bug).

5. **Background Learning Queue** — `learn()` saves data immediately but defers the expensive retrain (FAISS rebuild + ML refit) until 5 items queue up. Prevents system lockups.

6. **Sliding Context Window** — Last 5 conversation turns are injected into the LLM prompt, enabling pronoun resolution ("what about it?", "explain that").

7. **Category-as-File Architecture** — Each category is a standalone JSON file. Companies can add/remove categories by dropping files into a folder. No central database to corrupt.

8. **Many-to-Many Tags** — Categories can declare tags. A question about "python flask deployment" matches categories tagged with any of those words, beyond just the category name.

---

## Run Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Download models (first time only)
python download_model.py

# Run web UI (port 5000)
python app.py

# Run CLI mode
python chatbot.py

# Run tests
python test_deep_analysis.py --fast    # 60 questions, TinyLlama disabled
python test_100q.py                     # 100 questions
```

---

## Performance Snapshot

| Metric | Value |
|---|---|
| Dataset | 1,094 categories, 6,479 questions |
| Boot time | ~60-70s (FAISS index build + model load) |
| Query latency (dataset) | ~50-200ms |
| Query latency (LLM) | ~3-8s per question |
| FAISS index | 6,479 vectors, 384 dimensions |
| Memory footprint | ~800 MB (MiniLM + TinyLlama + FAISS) |
| Accuracy (deep test) | ~68-75% on 60 diverse questions |
