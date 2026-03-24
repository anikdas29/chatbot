"""
Mini Chatbot - NLP + ML + Semantic Search + Self Learning + TinyLlama Generation
ONNX MiniLM Embedding + ML Classifier + Spell Correction + Answer Templates
TinyLlama 1.1B Local LLM for answer generation
SQLite Persistence + "Did you mean?" Suggestions + Conversation Memory
Category-wise dataset with answer refinement and feeling tracking.
No third-party AI API needed. Fully offline.
"""

import json
import logging
import os
import random
import re
import sqlite3
import time
import uuid
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from difflib import get_close_matches, SequenceMatcher

# Logging setup
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ============================================================
# Feature 1: ONNX Semantic Embedding (replaces TF-IDF search)
# ============================================================

class SemanticEncoder:
    """ONNX embedding model for semantic search.
    Supports multilingual-e5-small (50+ languages including Bangla) with MiniLM fallback.
    Converts text to 384-dim semantic vectors.
    Understands meaning: 'feeling sad' ≈ 'i am depressed' ≈ 'mon kharap'
    """

    def __init__(self, model_dir="models/minilm"):
        self.available = False
        self.is_e5 = False
        self.model_name = None
        self.tokenizer = None
        self.session = None

        # Try multilingual model first (better Bangla support)
        multilingual_dir = "models/multilingual-e5"
        multilingual_onnx = os.path.join(multilingual_dir, "onnx", "model.onnx")
        multilingual_tok = os.path.join(multilingual_dir, "tokenizer.json")

        if os.path.exists(multilingual_onnx) and os.path.exists(multilingual_tok):
            self.model_dir = multilingual_dir
            self.model_name = "multilingual-e5-small"
            self.is_e5 = True
        elif os.path.exists(os.path.join(model_dir, "onnx", "model.onnx")):
            self.model_dir = model_dir
            self.model_name = "MiniLM-L6-v2"
            self.is_e5 = False
        else:
            logging.warning("No embedding model found — semantic search disabled (Tier 4: fixed dataset mode)")
            return

        try:
            self.tokenizer = Tokenizer.from_file(os.path.join(self.model_dir, "tokenizer.json"))
            self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            self.tokenizer.enable_truncation(max_length=128)
            self.session = ort.InferenceSession(
                os.path.join(self.model_dir, "onnx", "model.onnx"),
                providers=["CPUExecutionProvider"]
            )
            self.available = True
            logging.info(f"SemanticEncoder loaded (ONNX {self.model_name})")
        except Exception as e:
            logging.warning(f"SemanticEncoder failed to load: {e} — semantic search disabled")

    def encode(self, texts):
        """Encode list of texts into normalized embeddings. Returns (N, 384) numpy array."""
        if not self.available:
            return np.zeros((1, 384), dtype=np.float32) if isinstance(texts, str) else np.zeros((len(texts) if not isinstance(texts, str) else 1, 384), dtype=np.float32)
        if isinstance(texts, str):
            texts = [texts]
        # E5 models need "query: " prefix for optimal performance
        if self.is_e5:
            texts = [f"query: {t}" if not t.startswith("query: ") else t for t in texts]
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })

        # Mean pooling
        embeddings = outputs[0]
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (embeddings * mask).sum(axis=1) / mask.sum(axis=1)

        # L2 normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return pooled / norms

    def similarity(self, query_embedding, corpus_embeddings):
        """Cosine similarity between query and corpus. Returns (N,) scores."""
        return np.dot(corpus_embeddings, query_embedding.flatten())


# ============================================================
# Feature 5: TinyLlama Local LLM Text Generation
# ============================================================

class TinyLlamaGenerator:
    """Hybrid RAG (Retrieval-Augmented Generation) with local LLM.

    Supports multiple LLMs with auto-detection (best first):
    1. Phi-3 Mini 3.8B (Q3_K_M) — better quality, 4096 context
    2. TinyLlama 1.1B (Q4_K_M) — smaller, faster, 1024 context

    Pipeline:
    1. Takes top-5 semantic search results (question + answer pairs)
    2. Feeds them as retrieval context into LLM's prompt
    3. LLM reasons over the retrieved data to generate a new answer
    """

    # LLM configs: (path, model_type, context_length, max_tokens, name)
    _LLM_OPTIONS = [
        ("models/phi3/Phi-3-mini-4k-instruct-Q3_K_M.gguf", "llama", 4096, 200, "Phi-3 Mini 3.8B"),
        ("models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "llama", 1024, 150, "TinyLlama 1.1B"),
    ]

    def __init__(self, model_path=None):
        self.model = None
        self.available = False
        self.model_name = None
        self.is_phi3 = False
        self._max_tokens = 150
        self._backend = None  # "llama_cpp" or "ctransformers"

        # Auto-detect best available LLM
        if model_path and os.path.exists(model_path):
            candidates = [(model_path, "llama", 1024, 150, "custom")]
        else:
            candidates = self._LLM_OPTIONS

        for path, mtype, ctx_len, max_tok, name in candidates:
            if not os.path.exists(path):
                continue

            is_phi = "phi" in name.lower()

            # Phi-3 needs llama-cpp-python (ctransformers doesn't support it)
            if is_phi:
                try:
                    from llama_cpp import Llama
                    self.model = Llama(
                        model_path=path,
                        n_ctx=ctx_len,
                        n_threads=4,
                        verbose=False,
                    )
                    self.available = True
                    self.model_name = name
                    self.is_phi3 = True
                    self._max_tokens = max_tok
                    self._backend = "llama_cpp"
                    logging.info(f"{name} RAG generator loaded via llama-cpp (context: {ctx_len} tokens)")
                    break
                except Exception as e:
                    logging.warning(f"{name} load failed (llama-cpp): {e} — trying next...")
                    continue

            # TinyLlama uses ctransformers
            try:
                from ctransformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    os.path.dirname(path),
                    model_file=os.path.basename(path),
                    model_type=mtype,
                    max_new_tokens=max_tok,
                    temperature=0.4,
                    top_p=0.85,
                    repetition_penalty=1.15,
                    context_length=ctx_len,
                )
                self.available = True
                self.model_name = name
                self.is_phi3 = False
                self._max_tokens = max_tok
                self._backend = "ctransformers"
                logging.info(f"{name} RAG generator loaded via ctransformers (context: {ctx_len} tokens)")
                break
            except Exception as e:
                logging.warning(f"{name} load failed (ctransformers): {e} — trying next...")

        if not self.available:
            logging.warning("No LLM model found — generation disabled")

    def generate_rag(self, question, retrieved_context, categories, conversation_history=None):
        """Hybrid RAG: Generate answer from retrieved semantic search results.

        Args:
            question: User's original question
            retrieved_context: List of {"question": str, "answer": str, "category": str, "score": float}
            categories: List of detected category names
            conversation_history: Optional list of {"user": str, "bot": str} recent turns for context

        Returns: Generated text, or None if unavailable/failed.
        """
        if not self.available or not self.model:
            return None

        # Build retrieval context — top 5 Q&A pairs from semantic search
        # Don't include [category] or Q:/A: labels — TinyLlama copies them into output
        rag_lines = []
        for i, ctx in enumerate(retrieved_context[:5], 1):
            rag_lines.append(
                f"{i}. {ctx['answer']}"
            )
        rag_context = "\n".join(rag_lines)

        topic = ", ".join(c.replace("_", " ") for c in categories[:3])

        # Build sliding context window from recent conversation
        conv_block = ""
        if conversation_history:
            conv_lines = []
            for turn in conversation_history[-5:]:  # last 5 exchanges max
                if turn.get("user"):
                    conv_lines.append(f"User: {turn['user']}")
                if turn.get("bot"):
                    # Truncate long bot replies to save context space
                    bot_reply = turn["bot"][:150]
                    conv_lines.append(f"Assistant: {bot_reply}")
            if conv_lines:
                conv_block = "\n\nRecent conversation:\n" + "\n".join(conv_lines)

        if self.is_phi3:
            # Phi-3 uses <|system|>...<|end|> format with better instruction following
            prompt = f"""<|system|>
You are Mini Bot, a helpful assistant. Answer using ONLY the knowledge below. Be concise (2-3 sentences). Never start with "Yes", "Answer:", or "The question is". Just answer naturally. Do not mention "the text" or "the context".
<|end|>
<|user|>
Knowledge about {topic}:
{rag_context}{conv_block}

{question}
<|end|>
<|assistant|>
"""
        else:
            # TinyLlama uses <|system|>...</s> format
            prompt = f"""<|system|>
You are a helpful assistant. Answer the user's question using ONLY the retrieved knowledge below. Be concise (2-3 sentences). Do not mention "the text" or "the context" — just answer naturally. If the knowledge doesn't fully answer the question, use what's available. Use the recent conversation to understand pronouns like "it", "they", "that", "this".
</s>
<|user|>
Retrieved knowledge about {topic}:
{rag_context}{conv_block}

Question: {question}
</s>
<|assistant|>
"""

        try:
            if self._backend == "llama_cpp":
                output = self.model(prompt, max_tokens=self._max_tokens,
                                    temperature=0.4, top_p=0.85,
                                    repeat_penalty=1.15, stop=["<|end|>", "<|user|>", "</s>"])
                response = output["choices"][0]["text"].strip()
            else:
                response = self.model(prompt)
            response = response.strip()
            # Clean prompt artifacts
            if "<|" in response:
                response = response.split("<|")[0].strip()
            # Remove self-references to context
            for phrase in ["based on the", "according to the", "the text says",
                          "mentioned in the", "from the context", "the retrieved"]:
                response = response.replace(phrase, "").replace(phrase.capitalize(), "")
            response = response.strip()
            if len(response) < 10 or len(response) > 600:
                return None
            return response
        except Exception as e:
            logging.warning(f"{self.model_name or 'LLM'} RAG generation failed: {e}")
            return None

    # Backward compat
    def generate(self, question, category, dataset_answers, max_answers=3):
        """Legacy method — converts to RAG format."""
        context = [{"question": "", "answer": a, "category": category, "score": 1.0}
                   for a in dataset_answers[:max_answers]]
        return self.generate_rag(question, context, [category])


# ============================================================
# Cross-Encoder Reranker (optional precision boost for FAISS)
# ============================================================

class CrossEncoderReranker:
    """Optional cross-encoder for re-ranking FAISS results.
    Cross-encoders see question+candidate together (more accurate than bi-encoder).
    Falls back gracefully if model not available.
    """

    def __init__(self, model_dir="models/reranker"):
        self.available = False
        self.session = None
        self.tokenizer = None

        model_path = os.path.join(model_dir, "model.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            logging.info("Cross-encoder reranker not found — skipping (optional)")
            return

        try:
            self.session = ort.InferenceSession(model_path)
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            self.tokenizer.enable_truncation(max_length=256)
            self.available = True
            logging.info("Cross-encoder reranker loaded")
        except Exception as e:
            logging.warning(f"Cross-encoder load failed: {e}")

    def rerank(self, question, candidates, top_n=5):
        """Re-rank candidates using cross-encoder scores.

        Args:
            question: User question string
            candidates: List of dicts with 'question' and 'answer' keys
            top_n: Number of results to return

        Returns: candidates reordered by cross-encoder score, or original if unavailable
        """
        if not self.available or not candidates:
            return candidates[:top_n]

        try:
            pairs = []
            for c in candidates:
                text = f"{question} [SEP] {c.get('answer', c.get('question', ''))}"
                pairs.append(text)

            encoded = self.tokenizer.encode_batch(pairs)
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

            # Some models need token_type_ids
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            input_names = [inp.name for inp in self.session.get_inputs()]
            feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = token_type_ids

            outputs = self.session.run(None, feeds)
            scores = outputs[0].flatten()

            # Sort by cross-encoder score
            scored = list(zip(scores, candidates))
            scored.sort(key=lambda x: x[0], reverse=True)

            return [c for _, c in scored[:top_n]]
        except Exception as e:
            logging.warning(f"Reranker failed: {e}")
            return candidates[:top_n]


# ============================================================
# Feature 4: SQLite Persistence
# ============================================================

class Database:
    """SQLite database for persistent sessions, feedback, and learning logs."""

    def __init__(self, db_path="chatbot.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logging.info(f"Database connected: {db_path}")

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT,
                bot_reply TEXT,
                intent TEXT,
                confidence REAL DEFAULT 0,
                sentiment TEXT DEFAULT 'neutral',
                created_at REAL DEFAULT (strftime('%s','now')),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                bot_answer TEXT,
                intent TEXT,
                feedback_type TEXT,
                correct_answer TEXT,
                correct_category TEXT,
                created_at REAL DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS learn_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                category TEXT,
                answer TEXT,
                source TEXT DEFAULT 'user',
                created_at REAL DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS pending_learns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                category TEXT NOT NULL,
                answer TEXT,
                source TEXT DEFAULT 'user',
                created_at REAL DEFAULT (strftime('%s','now')),
                processed INTEGER DEFAULT 0
            );

            -- Indexes for fast lookups as data grows
            CREATE INDEX IF NOT EXISTS idx_turns_session
                ON conversation_turns(session_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_turns_created
                ON conversation_turns(created_at);
            CREATE INDEX IF NOT EXISTS idx_sessions_created
                ON sessions(created_at);
            CREATE INDEX IF NOT EXISTS idx_feedback_created
                ON feedback(created_at);
            CREATE INDEX IF NOT EXISTS idx_pending_processed
                ON pending_learns(processed, id);
        """)
        self.conn.commit()

    # --- Session ---
    def create_session(self):
        session_id = str(uuid.uuid4())
        self.conn.execute("INSERT INTO sessions (session_id) VALUES (?)", (session_id,))
        self.conn.commit()
        return session_id

    def add_turn(self, session_id, user_msg, bot_reply, intent, confidence=0, sentiment="neutral"):
        self.conn.execute(
            "INSERT INTO conversation_turns (session_id, user_message, bot_reply, intent, confidence, sentiment) VALUES (?,?,?,?,?,?)",
            (session_id, user_msg, bot_reply, intent, confidence, sentiment)
        )
        self.conn.commit()

    def get_history(self, session_id, limit=10):
        rows = self.conn.execute(
            "SELECT user_message, bot_reply, intent FROM conversation_turns WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
        return [{"user": r["user_message"], "bot": r["bot_reply"], "intent": r["intent"]} for r in reversed(rows)]

    def get_last_intent(self, session_id):
        row = self.conn.execute(
            "SELECT intent FROM conversation_turns WHERE session_id=? AND intent IS NOT NULL ORDER BY id DESC LIMIT 1",
            (session_id,)
        ).fetchone()
        return row["intent"] if row else None

    # --- Feedback ---
    def add_feedback(self, question, bot_answer, intent, feedback_type, correct_answer=None, correct_category=None):
        self.conn.execute(
            "INSERT INTO feedback (question, bot_answer, intent, feedback_type, correct_answer, correct_category) VALUES (?,?,?,?,?,?)",
            (question, bot_answer, intent, feedback_type, correct_answer, correct_category)
        )
        self.conn.commit()

    def get_feedback_stats(self):
        row = self.conn.execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN feedback_type='like' THEN 1 ELSE 0 END) as likes, SUM(CASE WHEN feedback_type='dislike' THEN 1 ELSE 0 END) as dislikes FROM feedback"
        ).fetchone()
        return {"total": row["total"], "likes": row["likes"] or 0, "dislikes": row["dislikes"] or 0}

    # --- Learn Log ---
    def log_learn(self, question, category, answer, source="user"):
        self.conn.execute(
            "INSERT INTO learn_log (question, category, answer, source) VALUES (?,?,?,?)",
            (question, category, answer, source)
        )
        self.conn.commit()

    # --- Pending Learns ---
    def add_pending_learn(self, question, category, answer, source="user"):
        self.conn.execute(
            "INSERT INTO pending_learns (question, category, answer, source) VALUES (?,?,?,?)",
            (question, category, answer, source)
        )
        self.conn.commit()

    def get_pending_learns(self):
        rows = self.conn.execute(
            "SELECT id, question, category, answer, source FROM pending_learns WHERE processed=0 ORDER BY id"
        ).fetchall()
        return [{"id": r["id"], "question": r["question"], "category": r["category"],
                 "answer": r["answer"], "source": r["source"]} for r in rows]

    def get_pending_count(self):
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM pending_learns WHERE processed=0").fetchone()
        return row["cnt"]

    def mark_pending_processed(self, ids):
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(f"UPDATE pending_learns SET processed=1 WHERE id IN ({placeholders})", ids)
        self.conn.commit()

    # --- Cleanup ---
    def cleanup_old_sessions(self, max_age_hours=48):
        cutoff = time.time() - (max_age_hours * 3600)
        self.conn.execute("DELETE FROM conversation_turns WHERE created_at < ?", (cutoff,))
        self.conn.execute("DELETE FROM sessions WHERE created_at < ?", (cutoff,))
        self.conn.commit()


# ============================================================
# Feature 3: Answer Templates
# ============================================================

class AnswerTemplates:
    """Generates diverse responses from stored answers using templates.
    Instead of repeating same 4 answers, creates variations.
    """

    PREFIXES = {
        "curious": [
            "Great question! ", "Good one! ", "Interesting! ",
            "Valo question! ", "Nice question! ",
        ],
        "happy": [
            "Glad you asked! ", "Awesome! ", "Nice! ",
        ],
        "sad": [
            "I understand. ", "It's okay. ", "Don't worry. ",
        ],
        "angry": [
            "I hear you. ", "Let me help. ",
        ],
        "confused": [
            "Let me explain. ", "Simply put: ", "Easy version: ",
        ],
        "neutral": [
            "", "", "",  # no prefix most of the time
            "Here's what I know: ", "So, ",
        ],
    }

    SUFFIXES = [
        "",
        "\n\nAr kichu jante chaile bolo!",
        "\n\nHope this helps!",
        "\n\nAro kichu lagbe bolo!",
        "",
        "",  # more weight to no suffix
    ]

    CONNECTORS = [
        "Also, ", "Additionally, ", "Plus, ", "And ", "Moreover, ",
    ]

    @staticmethod
    def generate(base_answer, sentiment="neutral", answers_pool=None, question=""):
        """Generate a varied response from base answer + optional secondary answer."""
        # Pick prefix based on sentiment
        prefixes = AnswerTemplates.PREFIXES.get(sentiment, AnswerTemplates.PREFIXES["neutral"])
        prefix = random.choice(prefixes)

        result = prefix + base_answer

        # 30% chance to append a secondary answer snippet from pool
        if answers_pool and len(answers_pool) > 1 and random.random() < 0.3:
            other_answers = [a for a in answers_pool if a != base_answer]
            if other_answers:
                secondary = random.choice(other_answers)
                # Take first sentence only
                first_sentence = secondary.split(".")[0].strip()
                if first_sentence and len(first_sentence) > 15:
                    connector = random.choice(AnswerTemplates.CONNECTORS)
                    result += " " + connector + first_sentence + "."

        # Add suffix
        suffix = random.choice(AnswerTemplates.SUFFIXES)
        result += suffix

        return result


# ============================================================
# Existing classes (improved)
# ============================================================

class SpellCorrector:
    """Known words er sathe match kore spelling fix kore.
    Length-guarded: short words (3-4 chars) need a higher match cutoff,
    and corrections must be within a length ratio to prevent
    'sql' → 'ssl', 'api' → 'ami', 'npm' → 'nlp' type overwrites.
    """

    # Common words that should NEVER be spell-corrected
    SKIP_WORDS = {
        # Common English
        "i", "me", "my", "we", "you", "your", "he", "she", "it", "they", "them",
        "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
        "do", "does", "did", "dont", "doesnt", "didnt", "wont", "cant", "shouldnt",
        "have", "has", "had", "will", "would", "could", "should", "can", "may",
        "not", "no", "yes", "ok", "and", "or", "but", "if", "so", "to", "for",
        "in", "on", "at", "by", "of", "with", "from", "up", "out", "off",
        "what", "how", "why", "when", "where", "who", "which", "that", "this",
        "im", "ive", "its", "thats", "whats", "hows", "ill", "id", "theyre",
        "there", "their", "here", "very", "too", "also", "just", "now", "then",
        "all", "every", "some", "any", "many", "much", "more", "most",
        "about", "know", "dont", "need", "want", "like", "feel", "think",
        "make", "get", "go", "come", "keep", "let", "say", "tell", "give",
        "work", "try", "use", "find", "take", "run", "see", "look", "help",
        "new", "old", "good", "bad", "best", "late", "still", "even",
        # Bangla common
        "ki", "ke", "ta", "er", "e", "te", "na", "ar", "ba", "o", "r",
        "ami", "tumi", "apni", "amake", "tomar", "keno", "koto", "kobe",
        "kivabe", "kotha", "kothay", "kon", "ki", "niye", "theke", "diye",
        "kore", "kori", "korte", "korbo", "hoy", "hobe", "ache", "chilo",
        "chai", "chao", "lagbe", "bolo", "bolte", "shikhao", "shikhte",
        "banabo", "banate", "dekhao", "jante", "jani", "pari", "parbo",
    }

    # Max allowed length difference between original and correction
    MAX_LENGTH_DIFF = 2

    def __init__(self, known_words):
        self.known_words = known_words

    def _get_cutoff(self, word_len):
        """Adaptive cutoff: shorter words need stricter matching.
        3-4 chars: 0.88 (protects abbreviations: sql, api, npm, git, css)
        5-6 chars: 0.82 (catches transpositions: pytohn→python)
        7+ chars:  0.78 (longer words tolerate more edits: javscript→javascript)
        """
        if word_len <= 4:
            return 0.88
        elif word_len <= 6:
            return 0.82
        return 0.78

    def correct(self, text):
        words = text.lower().split()
        corrected = []
        for word in words:
            # Skip: already known, too short, or in skip list
            if word in self.known_words or len(word) <= 2 or word in self.SKIP_WORDS:
                corrected.append(word)
                continue

            # Adaptive cutoff based on word length
            cutoff = self._get_cutoff(len(word))
            matches = get_close_matches(word, self.known_words, n=3, cutoff=cutoff)

            if not matches:
                corrected.append(word)
                continue

            # Length guard: reject candidates that differ too much in length
            best = None
            for match in matches:
                length_diff = abs(len(match) - len(word))
                if length_diff <= self.MAX_LENGTH_DIFF:
                    best = match
                    break

            corrected.append(best if best else word)
        return " ".join(corrected)


class SentimentDetector:
    """Simple sentiment detection"""

    ANGRY_WORDS = {
        "kharap", "baje", "worst", "bad", "angry", "problem", "fix",
        "error", "kaj kore na", "broken", "slow", "hate", "bekar",
        "ghatia", "scam", "fraud", "thug", "terrible", "horrible",
        "annoyed", "frustrated", "useless"
    }
    SAD_WORDS = {
        "sad", "depressed", "lonely", "stressed", "upset", "crying",
        "hopeless", "tired", "exhausted", "anxious", "worried",
        "dukhito", "kosto", "ekla", "thaka"
    }
    HAPPY_WORDS = {
        "great", "awesome", "valo", "bhalo", "excellent", "best",
        "good", "love", "happy", "amazing", "wonderful", "darun",
        "oshadharon", "khushi", "thanks", "thank", "perfect"
    }
    CURIOUS_WORDS = {
        "what", "how", "why", "when", "where", "who", "which",
        "explain", "tell", "ki", "keno", "kivabe", "kokhon"
    }
    CONFUSED_WORDS = {
        "confused", "dont understand", "bujhi na", "unclear",
        "what do you mean", "help", "lost"
    }

    @staticmethod
    def detect(text):
        words = set(text.lower().split())
        angry = len(words & SentimentDetector.ANGRY_WORDS)
        sad = len(words & SentimentDetector.SAD_WORDS)
        happy = len(words & SentimentDetector.HAPPY_WORDS)
        curious = len(words & SentimentDetector.CURIOUS_WORDS)
        confused = len(words & SentimentDetector.CONFUSED_WORDS)
        scores = {"angry": angry, "sad": sad, "happy": happy, "curious": curious, "confused": confused}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "neutral"
        return best

    @staticmethod
    def adjust_response(answer, sentiment):
        if sentiment == "angry":
            return "Ami bujhte parchi apni frustrated. " + answer + "\n\nApnar somossa solve korte amra committed. Ar kichu lagleo bolun."
        if sentiment == "sad":
            return "Ami bujhte parchi tumi ektu kharap feel korcho. " + answer + "\n\nTumi ekla na, amra achi."
        if sentiment == "happy":
            return answer + "\n\nApnar bhalo lagche jene amrao khushi!"
        if sentiment == "confused":
            return "Chinta koro na, ami bujhiye bolchi. " + answer
        return answer


class CategoryStore:
    """Category-wise dataset folder manage kore.
    Supports optional 'tags' field for many-to-many mapping:
    a category can have tags like ["python", "web", "backend"]
    so questions can match through tag overlap, not just category name.
    """

    def __init__(self, folder="category_wise_dataset"):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.categories = {}
        self.tag_index = {}  # tag → set of category names
        self._load_all()

    def _cat_path(self, category):
        return os.path.join(self.folder, f"{category}.json")

    def _load_all(self):
        self.categories = {}
        self.tag_index = {}
        for fname in os.listdir(self.folder):
            if fname.endswith(".json"):
                fpath = os.path.join(self.folder, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cat_name = data.get("category", fname.replace(".json", ""))
                    self.categories[cat_name] = data
                    # Build tag index
                    for tag in data.get("tags", []):
                        tag_lower = tag.lower().strip()
                        if tag_lower:
                            if tag_lower not in self.tag_index:
                                self.tag_index[tag_lower] = set()
                            self.tag_index[tag_lower].add(cat_name)
                except (json.JSONDecodeError, KeyError):
                    logging.warning(f"Skipped invalid category file: {fname}")
        logging.info(
            f"CategoryStore loaded: {len(self.categories)} categories, "
            f"{len(self.tag_index)} unique tags from {self.folder}/"
        )

    def get(self, category):
        return self.categories.get(category)

    def get_all_categories(self):
        return list(self.categories.keys())

    def get_questions_and_labels(self):
        questions = []
        labels = []
        for cat_name, data in self.categories.items():
            for q in data.get("questions", []):
                questions.append(q.lower())
                labels.append(cat_name)
        return questions, labels

    def get_answers(self, category):
        data = self.categories.get(category)
        if not data:
            return []
        return data.get("answers", [])

    def get_type(self, category):
        data = self.categories.get(category)
        if not data:
            return "general"
        return data.get("type", "general")

    def get_tags(self, category):
        data = self.categories.get(category)
        if not data:
            return []
        return [t.lower().strip() for t in data.get("tags", [])]

    def find_categories_by_tag(self, tag):
        """Find all categories that have this tag. Returns set of category names."""
        return self.tag_index.get(tag.lower().strip(), set())

    def find_categories_by_tags(self, tags):
        """Find categories matching ANY of the given tags.
        Returns dict: category → number of matching tags (for ranking).
        """
        cat_hits = {}
        for tag in tags:
            for cat in self.find_categories_by_tag(tag):
                cat_hits[cat] = cat_hits.get(cat, 0) + 1
        return cat_hits

    def save_category(self, category):
        data = self.categories.get(category)
        if data:
            fpath = self._cat_path(category)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    def add_question(self, category, question):
        if category in self.categories:
            qs = self.categories[category].get("questions", [])
            if question not in qs:
                qs.append(question)
                self.categories[category]["questions"] = qs
                self.save_category(category)

    def add_answer(self, category, answer):
        if category in self.categories:
            ans = self.categories[category].get("answers", [])
            if answer not in ans:
                ans.append(answer)
                self.categories[category]["answers"] = ans
                self.save_category(category)

    def add_feeling(self, category, question, sentiment, answer_given):
        if category not in self.categories:
            return
        feelings = self.categories[category].get("feelings", [])
        feelings.append({
            "question": question,
            "sentiment": sentiment,
            "answer": answer_given
        })
        self.categories[category]["feelings"] = feelings[-50:]
        self.save_category(category)

    def create_category(self, category, cat_type="general", questions=None, answers=None):
        data = {
            "category": category,
            "type": cat_type,
            "questions": questions or [],
            "answers": answers or [],
            "feelings": []
        }
        self.categories[category] = data
        self.save_category(category)
        logging.info(f"New category created: {category} [{cat_type}]")

    def category_exists(self, category):
        return category in self.categories


# ============================================================
# Intent Signal Detection (catches meaning ONNX misses)
# ============================================================

class IntentSignalDetector:
    """Pre-detection layer: catches user intent from patterns/signals
    BEFORE semantic search. Solves the "understanding" problem where
    ONNX matches surface words but misses what user actually needs.
    """

    # Pattern → list of boosted categories
    # Each pattern: (regex_or_keywords, target_categories, boost_score)
    SIGNALS = [
        # === Confusion / Life guidance ===
        (r"(dont know what to do|no idea what|confused about|lost in life|what should i do)",
         ["career", "motivation", "career_coaching"], 0.75),
        (r"(is it too late|am i too old|should i change|switch career|career change)",
         ["career", "career_coaching", "motivation"], 0.70),
        (r"(what.?s the point|why should i|why bother|is it worth)",
         ["motivation", "career", "philosophy"], 0.65),

        # === Frustration / Things not working ===
        (r"(not working|not loading|doesnt work|wont open|keeps crashing|broken|stopped working)",
         ["coding_errors", "debugging", "troubleshooting"], 0.70),
        (r"(my code.*(wrong|error|bug|fail|break|crash))",
         ["coding_errors", "debugging"], 0.75),
        (r"(i found a bug|report.*(bug|issue|problem)|there.?s a (bug|issue|problem))",
         ["coding_errors", "bug_bounty"], 0.70),

        # === Emotional / Mental health ===
        (r"(everything.*(wrong|bad|failing)|nothing.*(works|right)|i.?m (stuck|lost|hopeless))",
         ["motivation", "mental_health", "stress_management"], 0.75),
        (r"(overwhelmed|too much|cant handle|burning out|burnt out|burnout)",
         ["stress_management", "mental_health", "motivation"], 0.75),
        (r"(feeling (sad|down|depressed|lonely|anxious|stressed|low))",
         ["mental_health", "emotions", "loneliness"], 0.80),
        (r"(help me|i need help|please help|someone help)",
         ["mental_health", "motivation"], 0.50),
        (r"(feel better|cheer.* up|make me happy|motivate me)",
         ["motivation", "mental_health", "entertainment"], 0.70),

        # === Nobody uses / Marketing ===
        (r"(nobody uses|no users|no downloads|no traffic|how to (get|attract) (users|customers|traffic))",
         ["marketing", "seo", "social_media"], 0.70),
        (r"(how to (promote|market|grow|scale)|get more (users|views|followers))",
         ["marketing", "social_media", "startup"], 0.70),

        # === Learning / How to start ===
        (r"(where.* start|how.* begin|how.* learn|teach me|shikhao|shikhte chai)",
         ["programming", "coding", "career"], 0.55),
        (r"(best way to learn|roadmap|learning path|guide for beginner)",
         ["programming", "career", "study"], 0.60),

        # === Running / Fitness (disambiguate from "run a program") ===
        (r"(run faster|running (tips|speed|form)|jogging|marathon|sprint)",
         ["running", "fitness", "gym_workout"], 0.80),
        (r"(run.*(program|script|code|app|server|command))",
         ["programming", "linux", "coding"], 0.70),

        # === Interview / Job ===
        (r"(fail.*(interview|exam|test)|rejected|didnt get.*(job|offer))",
         ["job_interview", "career", "motivation"], 0.75),
        (r"(prepare for interview|interview tips|crack.*(interview|job))",
         ["job_interview", "career"], 0.75),

        # === Bangla intent patterns ===
        (r"(ki korbo|bujhi na|bujhtesi na|help koro|sahajjo koro)",
         ["motivation", "career", "mental_health"], 0.60),
        (r"(keno hocche na|kaj kortese na|problem hocche|error ashche)",
         ["coding_errors", "debugging"], 0.70),
        (r"(kivabe (shikhi|shuru kori|kori)|shikhte chai|shikhao)",
         ["programming", "coding", "career"], 0.65),

        # === Bot meta-questions ===
        (r"(tumi ki (chatbot|bot|ai|real|human|robot)|ke tomake (banai|create|made))",
         ["about_bot", "bot_capability", "bot_name"], 0.80),
        (r"(tumi ki (korte paro|paro|help korte paro)|what can you do|your (capability|limitation|feature))",
         ["bot_capability", "about_bot"], 0.80),
        (r"(tumi ki offline|are you offline|do you (need|use) internet)",
         ["bot_capability", "about_bot"], 0.75),

        # === Web development / Backend / Frontend ===
        (r"(web (dev|development)|frontend|backend|fullstack|full.?stack)",
         ["web_dev", "html_css", "javascript"], 0.70),
        (r"(backend (ki|keno|diye|with)|server.?side|api (build|develop|create))",
         ["web_dev", "api", "flask_framework"], 0.70),

        # === Database ===
        (r"(database|sql|mysql|postgres|mongodb|nosql).*(ki|kothay|keno|use|learn|basics)",
         ["database", "sql", "mongodb"], 0.70),

        # === Boredom / Entertainment ===
        (r"(bored|bore) (lagche|feel|hocche)|nothing to do|ki korbo time pass",
         ["entertainment", "puzzle_games", "motivation"], 0.65),

        # === Friendship / Loneliness ===
        (r"(tumi ki (amar |amr )?(friend|bondhu)|can (you|we) be friend|ami (ekla|lonely))",
         ["loneliness", "emotions", "mental_health"], 0.70),

        # === Confused about choosing ===
        (r"(confused|confuse).*(frontend|backend|choose|select|pick|decide)",
         ["web_dev", "career", "career_coaching"], 0.70),
        (r"(ami |i )?(confused|confuse).*(naki|or|vs|between)",
         ["career", "career_coaching"], 0.65),

        # === Shopping / E-commerce ===
        (r"(buy|purchase|order|shop|kinte chai|kinbo)",
         ["ecommerce", "shopping", "online_shopping"], 0.65),
        (r"(price|cost|dam|koto taka|budget|expensive|cheap|sasta)",
         ["pricing_strategy", "budgeting", "ecommerce"], 0.60),
        (r"(delivery|shipping|courier|parcel)",
         ["logistics", "ecommerce", "supply_chain"], 0.65),
        (r"(refund|return|exchange|money back)",
         ["consumer_rights", "ecommerce"], 0.70),

        # === Health symptoms ===
        (r"(headache|matha byatha|matha ghurche|dizzy)",
         ["headache", "migraine", "health"], 0.75),
        (r"(fever|jor|temperature|sick|osthir)",
         ["health", "first_aid", "dehydration"], 0.70),
        (r"(stomach|pet byatha|pet kharap|vomit|diarrhea)",
         ["food_poisoning", "health", "dehydration"], 0.75),
        (r"(back pain|waist pain|komorey byatha|joint pain)",
         ["joint_pain", "health", "fitness"], 0.70),
        (r"(can.?t sleep|ghum hoy na|insomnia|sleeping problem)",
         ["insomnia", "sleep_hygiene", "health"], 0.75),

        # === Education / Study ===
        (r"(exam|porikkha|test preparation|study tips)",
         ["exam_preparation", "study", "education"], 0.70),
        (r"(scholarship|brishtti|free education|tuition)",
         ["scholarship", "education", "study_abroad"], 0.70),
        (r"(university|college|admission|varsity|school)",
         ["college", "education", "study_abroad"], 0.65),
        (r"(gpa|cgpa|grade|marks|result)",
         ["exam_preparation", "education", "college"], 0.65),

        # === Finance / Money ===
        (r"(invest|investment|taka invest|portfolio)",
         ["stock_market", "mutual_funds", "finance"], 0.70),
        (r"(loan|rin|mortgage|emi|interest rate)",
         ["finance", "insurance", "credit_score"], 0.70),
        (r"(tax|kar|income tax|vat)",
         ["finance", "freelance_tax", "accounting"], 0.70),
        (r"(savings?|bachat|save money|emergency fund)",
         ["budgeting", "finance", "savings"], 0.65),

        # === Travel / Transport ===
        (r"(visa|passport|immigration|embassy)",
         ["visa_process", "passport", "travel"], 0.75),
        (r"(hotel|booking|reservation|airbnb)",
         ["travel", "airbnb", "honeymoon"], 0.65),
        (r"(flight|plane|airport|airline)",
         ["travel", "adventure_travel"], 0.65),
        (r"(train|bus|transport|ride)",
         ["travel", "ride_sharing", "carpooling"], 0.60),

        # === Food / Cooking ===
        (r"(recipe|ranna|cook|randhbo kivabe)",
         ["cooking", "bangladeshi_food", "biryani"], 0.70),
        (r"(healthy food|diet|khabar|nutrition|protein)",
         ["diet_plans", "health", "vitamins"], 0.65),
        (r"(restaurant|food delivery|khabar order)",
         ["cooking", "food_truck", "ecommerce"], 0.60),

        # === Relationship / Social ===
        (r"(relationship|breakup|love|crush|ভালোবাসা)",
         ["emotions", "mental_health", "counseling"], 0.70),
        (r"(marriage|biye|wedding|divorce)",
         ["marriage", "divorce", "counseling"], 0.70),
        (r"(family|parents|children|son|daughter|baba|ma)",
         ["parenting", "family", "elder_care"], 0.60),
        (r"(friend|bondhu|friendship|social)",
         ["loneliness", "emotions", "mental_health"], 0.60),

        # === Legal ===
        (r"(lawyer|ukil|court|case|sue)",
         ["consumer_rights", "human_rights", "patient_rights"], 0.65),
        (r"(police|crime|thana|fir|report)",
         ["human_rights", "domestic_violence", "cyber_bullying"], 0.65),
    ]

    @staticmethod
    def detect(text):
        """Detect intent signals from text.
        Returns list of (category, boost_score) or empty list.
        """
        text_lower = text.lower().strip()
        boosts = {}

        for pattern, categories, score in IntentSignalDetector.SIGNALS:
            if re.search(pattern, text_lower):
                for cat in categories:
                    if cat not in boosts or boosts[cat] < score:
                        boosts[cat] = score

        # Sort by score descending
        return sorted(boosts.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# Multi-Factor Confidence Scorer
# ============================================================

class ConfidenceScorer:
    """Combines multiple signals into a calibrated 0-100% "sureness" score.

    Problem with raw scores:
    - Cosine similarity clusters in 0.3-0.8, never hits 0 or 1
    - ML probabilities spread across 1000+ classes, so even correct = low prob
    - Linear blending (0.6*sem + 0.4*ml) produces unintuitive numbers

    Solution: weighted combination → sigmoid normalization → calibrated percentage.
    The sigmoid stretches the useful range (~0.3-0.8) into a full 0-100% scale
    while compressing the tails, matching human intuition about "sureness".

    Factors:
    1. Semantic similarity (FAISS cosine) — primary signal
    2. ML classification probability — secondary signal
    3. Intent signal match — pattern-based boost
    4. Signal agreement — how many methods agree on the category
    5. Score gap — distance between #1 and #2 category (larger gap = more sure)
    """

    # Weights for each factor (tuned for MiniLM + LogReg + IntentDetector)
    W_SEMANTIC = 0.45      # Semantic similarity is the strongest signal
    W_ML = 0.25            # ML classifier probability
    W_INTENT = 0.15        # Intent pattern match
    W_AGREEMENT = 0.10     # How many methods agree
    W_GAP = 0.05           # Score gap to runner-up

    # Sigmoid parameters: sigmoid(k * (x - midpoint))
    # k controls steepness, midpoint controls the 50% crossing point
    SIGMOID_K = 12.0       # Steepness — higher = sharper transition
    SIGMOID_MID = 0.38     # Raw score that maps to ~50% confidence

    # Floor and ceiling to avoid 0% and 100% (never fully sure or unsure)
    CONF_FLOOR = 0.05      # Minimum confidence shown (5%)
    CONF_CEILING = 0.98    # Maximum confidence shown (98%)

    @staticmethod
    def _sigmoid(x, k=None, mid=None):
        """Standard sigmoid: 1 / (1 + exp(-k*(x - mid)))"""
        if k is None:
            k = ConfidenceScorer.SIGMOID_K
        if mid is None:
            mid = ConfidenceScorer.SIGMOID_MID
        z = k * (x - mid)
        # Clip to prevent overflow
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def score(semantic_score, ml_score=0.0, intent_score=0.0,
              n_agreeing_signals=1, score_gap=0.0, is_bangla=False):
        """Calculate calibrated confidence from multiple factors.
        Fix #2: Dynamic weights — Bangla queries boost intent/tag, reduce semantic.

        Args:
            semantic_score: Best cosine similarity from FAISS (0.0-1.0)
            ml_score: ML classifier probability for this category (0.0-1.0)
            intent_score: Intent signal match score (0.0-1.0), 0 if no match
            n_agreeing_signals: Number of methods that agree (1=semantic only, 2=sem+ml, 3=sem+ml+intent)
            score_gap: Difference between this category's score and the runner-up (0.0-1.0)
            is_bangla: If True, use Bangla-optimized weights (boost intent, reduce semantic)

        Returns: Calibrated confidence (0.05-0.98)
        """
        # Fix #2: Dynamic weights for Bangla vs English
        if is_bangla:
            w_sem, w_ml, w_int, w_agr, w_gap = 0.25, 0.20, 0.35, 0.15, 0.05
        else:
            w_sem = ConfidenceScorer.W_SEMANTIC
            w_ml = ConfidenceScorer.W_ML
            w_int = ConfidenceScorer.W_INTENT
            w_agr = ConfidenceScorer.W_AGREEMENT
            w_gap = ConfidenceScorer.W_GAP

        # Normalize agreement to 0-1 range (1 signal=0.33, 2=0.67, 3=1.0)
        agreement = min(n_agreeing_signals, 3) / 3.0

        # Normalize gap (typical gap 0-0.3, stretch to 0-1)
        gap_norm = min(score_gap / 0.30, 1.0)

        # Weighted combination of all factors
        raw = (
            w_sem * semantic_score +
            w_ml * ml_score +
            w_int * intent_score +
            w_agr * agreement +
            w_gap * gap_norm
        )

        # Sigmoid normalization — maps raw to a calibrated 0-1 scale
        calibrated = ConfidenceScorer._sigmoid(raw)

        # Apply floor and ceiling
        calibrated = max(ConfidenceScorer.CONF_FLOOR,
                         min(ConfidenceScorer.CONF_CEILING, calibrated))

        return round(calibrated, 3)

    @staticmethod
    def score_multi(candidates_with_signals, is_bangla=False):
        """Score multiple candidates at once.

        Args:
            candidates_with_signals: list of dicts with keys:
                category, semantic_score, ml_score, intent_score,
                n_agreeing, gap, method
            is_bangla: Use Bangla-optimized weights

        Returns: list of (category, calibrated_confidence, method) sorted desc
        """
        results = []
        for c in candidates_with_signals:
            conf = ConfidenceScorer.score(
                semantic_score=c.get("semantic_score", 0),
                ml_score=c.get("ml_score", 0),
                intent_score=c.get("intent_score", 0),
                n_agreeing_signals=c.get("n_agreeing", 1),
                score_gap=c.get("gap", 0),
                is_bangla=is_bangla,
            )
            results.append((c["category"], conf, c.get("method", "semantic")))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ============================================================
# Follow-up detection
# ============================================================

FOLLOW_UP_WORDS = {
    "tell me more", "more", "aro bolo", "ar bolo", "elaborate",
    "explain more", "details", "aro details", "bistorito",
    "what else", "ar ki", "continue", "go on", "then",
    "ar kono", "aro kichu", "bolo aro", "keep going"
}

# Meta-commands: user wants a different STYLE of the same answer, not a new topic
# These should route to last_intent without topic similarity check
META_COMMAND_PATTERNS = re.compile(
    r"^(eta (explain|describe) koro|"
    r"eta keno lage|"
    r"eta kivabe kaj kore|"
    r"aro easy kore bolo|"
    r"example (dao|de|dew)|"
    r"abar bolo|"
    r"short(er)? answer( dao)?|"
    r"bujhte par(si|chi|lam) na|"
    r"eta solve koro|"
    r"eta thik naki (vul|bhul)|"
    r"(simple|easy|short) (kore|e) (bolo|bolte paro)|"
    r"(give|show) (me )?(an? )?example|"
    r"explain (again|simply|easily)|"
    r"say (that |it )?again|"
    r"repeat (that|it|please))$",
    re.IGNORECASE
)

# Pronoun patterns that reference previous topic (need conversation context)
PRONOUN_FOLLOW_UP = re.compile(
    r"^(what about (it|that|this|them)|"
    r"tell me about (it|that|this|them)|"
    r"explain (it|that|this)|"
    r"how about (that|this)|"
    r"(eta|ota|seita) (ki|niye|bolo|explain koro)|"
    r"and (that|this|it)\??|"
    r"why (is that|is it)|"
    r"(it|that|this) (ki|keno|kivabe))$",
    re.IGNORECASE
)

# Emotional fallback patterns — route to mental_health/motivation when nothing else matches
EMOTIONAL_FALLBACK = re.compile(
    r"(i (feel|am) (like )?(giving up|hopeless|lost|empty|alone|stuck|scared|worthless|useless)|"
    r"(life|everything) (is |e )?(too |onek )?(hard|difficult|meaningless|pointless)|"
    r"(amar|amr) (life|jibon) e (pressure|problem|kosto)|"
    r"(ami|i) (khub |very )?(sad|depressed|lonely|disappointed|frustrated|anxious|stressed)|"
    r"(amar|amr) kono (goal|hope|asha) nai|"
    r"(dont|don.?t) know what to do|"
    r"i (need|want) (help|someone)|"
    r"(nobody|no one) (cares|understands|loves)|"
    r"(help me|please help|sahajjo koro) (relax|calm|sleep|feel better)|"
    r"(inspire|motivate) me|"
    r"(ami|i) (bored|bore) (lagche|feel))",
    re.IGNORECASE
)


def is_follow_up(text):
    cleaned = text.lower().strip()
    for phrase in FOLLOW_UP_WORDS:
        if phrase in cleaned:
            return True
    # Check pronoun-based follow-ups
    if PRONOUN_FOLLOW_UP.search(cleaned):
        return True
    return False


def is_meta_command(text):
    """Check if user wants a different STYLE of the previous answer (not a new topic)."""
    cleaned = text.lower().strip()
    return bool(META_COMMAND_PATTERNS.search(cleaned))


# ============================================================
# Main ChatBot
# ============================================================

class ChatBot:
    def __init__(self, general_folder=None, specialized_folders=None,
                 model_dir="models/minilm", db_path="chatbot.db"):
        """
        Full-featured chatbot with:
        1. ONNX MiniLM semantic embedding (replaces TF-IDF for search)
        2. "Did you mean?" suggestions
        3. Answer templates for diverse responses
        4. SQLite persistence for sessions, feedback, learning
        5. TinyLlama 1.1B local LLM for answer generation
        """
        # Database (Feature 4: SQLite)
        self.db = Database(db_path)

        # Auto-detect mode: ISP (general/ + isp_business/) or General Purpose
        if general_folder is None:
            if os.path.isdir("general") and any(
                os.path.isdir(d) and d.endswith("_business") for d in os.listdir(".")
            ):
                general_folder = "general"
                logging.info("ISP mode detected: using general/ + *_business/ folders")
            else:
                general_folder = "category_wise_dataset"
                logging.info("General purpose mode: using category_wise_dataset/")

        # General store
        self.general_store = CategoryStore(general_folder)

        # Specialized stores
        self.specialized_stores = []
        if specialized_folders is None:
            specialized_folders = self._auto_detect_specialized_folders(general_folder)
        for folder in specialized_folders:
            if os.path.isdir(folder):
                store = CategoryStore(folder)
                self.specialized_stores.append(store)
                logging.info(f"Specialized store loaded: {folder} ({len(store.get_all_categories())} categories)")

        self.all_stores = self.specialized_stores + [self.general_store]

        # Category -> store mapping + global tag index
        self.category_store_map = {}
        self.global_tag_index = {}  # tag → set of categories across all stores
        self._build_category_map()

        # Load all questions + labels
        self.questions = []
        self.labels = []
        self._prepare_data()

        # Spell corrector
        self._build_spell_corrector()

        # Feature 1: Semantic encoder (ONNX MiniLM)
        self.encoder = SemanticEncoder(model_dir)
        self._build_semantic_index()

        # ML classifier (kept as secondary signal)
        self._train_intent_classifier()

        # Sentiment + templates
        self.sentiment = SentimentDetector()
        self.templates = AnswerTemplates()

        # Feature 5: TinyLlama local LLM generation
        self.generator = TinyLlamaGenerator()

        # Optional: Cross-encoder reranker for better FAISS precision
        self.reranker = CrossEncoderReranker()

        self.learn_count = 0
        logging.info(
            f"ChatBot initialized | Semantic + Templates + SQLite"
            f" | LLM: {self.generator.model_name if self.generator.available else 'OFF'}"
        )

    @staticmethod
    def _auto_detect_specialized_folders(general_folder):
        folders = []
        for item in os.listdir("."):
            if not os.path.isdir(item) or item == general_folder:
                continue
            # Match: *_dataset (coding_dataset) or *_business (isp_business)
            if item.endswith("_dataset") or item.endswith("_business"):
                folders.append(item)
        folders.sort()
        return folders

    def _build_category_map(self):
        self.category_store_map = {}
        self.global_tag_index = {}
        for cat in self.general_store.get_all_categories():
            self.category_store_map[cat] = self.general_store
        for store in self.specialized_stores:
            for cat in store.get_all_categories():
                self.category_store_map[cat] = store
        # Build global tag index across all stores
        for store in self.all_stores:
            for tag, cats in store.tag_index.items():
                if tag not in self.global_tag_index:
                    self.global_tag_index[tag] = set()
                self.global_tag_index[tag].update(cats)
        if self.global_tag_index:
            logging.info(f"Global tag index: {len(self.global_tag_index)} unique tags")

    def _get_store_for(self, category):
        return self.category_store_map.get(category, self.general_store)

    @property
    def store(self):
        return self.general_store

    def _prepare_data(self):
        self.questions = []
        self.labels = []
        for store in self.all_stores:
            qs, ls = store.get_questions_and_labels()
            self.questions.extend(qs)
            self.labels.extend(ls)
        total_cats = len(self.category_store_map)
        logging.info(
            f"Total questions: {len(self.questions)}, "
            f"Categories: {total_cats}"
        )

    def _build_spell_corrector(self):
        all_words = set()
        for q in self.questions:
            all_words.update(q.split())
        self.spell = SpellCorrector(all_words)

    # ========== Feature 1: FAISS Semantic Index ==========

    def _build_semantic_index(self):
        """Pre-compute embeddings and build FAISS index for fast vector search.
        FAISS uses Inner Product (= cosine similarity for L2-normalized vectors).
        Scales to 100K+ questions with sub-millisecond search.
        Tier 3/4: skips gracefully if encoder not available.
        """
        if not self.encoder.available:
            self.question_embeddings = None
            self.faiss_index = None
            logging.info("FAISS index skipped — no embedding model (Tier 4: fixed dataset mode)")
            return

        import faiss

        if self.questions:
            logging.info(f"Building FAISS index for {len(self.questions)} questions...")
            # Batch encode in chunks
            batch_size = 64
            all_embeddings = []
            for i in range(0, len(self.questions), batch_size):
                batch = self.questions[i:i + batch_size]
                emb = self.encoder.encode(batch)
                all_embeddings.append(emb)
            self.question_embeddings = np.vstack(all_embeddings).astype(np.float32)

            # Build FAISS index — Inner Product (cosine sim for normalized vectors)
            dim = self.question_embeddings.shape[1]  # 384
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.question_embeddings)

            logging.info(
                f"FAISS index built: {self.faiss_index.ntotal} vectors, "
                f"{dim}d, IndexFlatIP (cosine)"
            )
        else:
            self.question_embeddings = None
            self.faiss_index = None

    def _faiss_search(self, query_embedding, top_k=15):
        """Search FAISS index. Returns (scores, indices) arrays of top-K matches.
        Much faster than numpy dot product for large datasets.
        """
        if self.faiss_index is None:
            return np.array([]), np.array([])
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_index.search(query, top_k)
        return scores[0], indices[0]

    def _train_intent_classifier(self):
        """ML classifier as secondary signal (backup for semantic search)."""
        unique_labels = len(set(self.labels))
        if unique_labels < len(self.questions) * 0.6 and len(self.questions) > 0:
            self.intent_clf = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=1000, C=10))
            ])
            self.intent_clf.fit(self.questions, self.labels)
            self.use_ml = True
        else:
            self.use_ml = False

    def retrain(self):
        """Full model retrain — all stores reload + rebuild semantic index."""
        for store in self.all_stores:
            store._load_all()
        self._build_category_map()
        self._prepare_data()
        self._build_spell_corrector()
        self._build_semantic_index()
        self._train_intent_classifier()
        logging.info("Model retrained with new data")

    # ========== Category Detection (Semantic + ML) ==========

    def _detect_category(self, cleaned_question):
        """Single-category detection (backward compat). Returns: (category, confidence, method) or (None, 0, None)"""
        results = self._detect_categories(cleaned_question, max_cats=1)
        if results:
            return results[0]
        return None, 0, None

    # Bangla filler words for language detection
    _BANGLA_FILLER = {"ki", "keno", "kivabe", "kothay", "kokhon", "ke", "ta", "er", "e", "te",
                       "na", "ar", "ba", "o", "ami", "tumi", "apni", "ache", "hole", "kore",
                       "korte", "korbo", "hoy", "hobe", "paro", "diye", "niye", "theke",
                       "bolo", "lagbe", "chai", "shikhte", "kon", "onek", "amar", "tomar"}

    def _is_bangla_query(self, text):
        """Detect if query is Bangla/Banglish-heavy."""
        words = text.lower().split()
        if not words:
            return False
        bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        if bangla_chars > 3:
            return True
        banglish_count = sum(1 for w in words if w in self._BANGLA_FILLER)
        return banglish_count / len(words) > 0.3

    def _detect_categories(self, cleaned_question, max_cats=3, min_conf=0.25, gap_threshold=0.20):
        """Multi-category detection: Semantic + ML + Intent + ConfidenceScorer.
        Fix #2: Dynamic weights for Bangla. Fix #3: Two-threshold confidence.
        Returns list of (category, confidence, method) tuples, sorted by confidence desc.
        """
        if self.faiss_index is None:
            return []

        # Fix #2: Detect if Bangla-heavy query
        is_bangla = self._is_bangla_query(cleaned_question)

        # Fix #3: Two-threshold — lower bar if question contains a known technical term
        has_technical_term = any(w in self.category_store_map or w in self.global_tag_index
                                 for w in cleaned_question.split())
        effective_min_conf = 0.20 if has_technical_term else min_conf

        # ── Step 1: FAISS semantic search (top-15 matches) ──
        query_emb = self.encoder.encode(cleaned_question)
        scores, indices = self._faiss_search(query_emb, top_k=15)
        top_scores = [(self.labels[i], float(scores[j]))
                      for j, i in enumerate(indices) if i >= 0]

        # Aggregate by category — keep max score per category
        cat_sem_scores = {}
        for label, score in top_scores:
            if label not in cat_sem_scores:
                cat_sem_scores[label] = score
            else:
                cat_sem_scores[label] = max(cat_sem_scores[label], score)

        if not cat_sem_scores:
            return []

        sorted_cats = sorted(cat_sem_scores.items(), key=lambda x: x[1], reverse=True)

        # ── Step 2: ML classifier probabilities ──
        ml_cat_scores = {}  # category → ML probability
        if self.use_ml:
            ml_proba = self.intent_clf.predict_proba([cleaned_question])[0]
            ml_top3_idx = ml_proba.argsort()[-3:][::-1]
            for i in ml_top3_idx:
                ml_cat_scores[self.intent_clf.classes_[i]] = float(ml_proba[i])

        # ── Step 3: Intent signal detection ──
        intent_signals = IntentSignalDetector.detect(cleaned_question)
        intent_cat_scores = {}  # category → intent score
        for cat, score in intent_signals:
            if cat in self.category_store_map:
                intent_cat_scores[cat] = score

        # ── Step 3b: Tag-based category discovery ──
        # Extract words from question, look up in global tag index
        # This finds categories that tagged themselves with relevant keywords
        tag_cat_hits = {}  # category → number of matching tags
        if self.global_tag_index:
            q_words = set(cleaned_question.lower().split())
            for word in q_words:
                if word in self.global_tag_index:
                    for cat in self.global_tag_index[word]:
                        tag_cat_hits[cat] = tag_cat_hits.get(cat, 0) + 1

        # ── Step 4: Build candidate list with per-signal scores ──
        candidates = {}  # category → {semantic_score, ml_score, intent_score, method_parts}

        # From semantic top-8
        for cat, sem_score in sorted_cats[:8]:
            candidates[cat] = {
                "category": cat,
                "semantic_score": sem_score,
                "ml_score": ml_cat_scores.get(cat, 0),
                "intent_score": intent_cat_scores.get(cat, 0),
                "method_parts": ["semantic"],
            }
            if cat in ml_cat_scores and ml_cat_scores[cat] >= 0.15:
                candidates[cat]["method_parts"].append("ml")
            if cat in intent_cat_scores:
                candidates[cat]["method_parts"].append("intent")
            if cat in tag_cat_hits:
                candidates[cat]["method_parts"].append("tag")

        # Inject ML predictions missing from semantic top-8
        for ml_cat, ml_c in ml_cat_scores.items():
            if ml_cat not in candidates and ml_c >= 0.25:
                sem = cat_sem_scores.get(ml_cat, 0)
                if sem >= 0.20:
                    candidates[ml_cat] = {
                        "category": ml_cat,
                        "semantic_score": sem,
                        "ml_score": ml_c,
                        "intent_score": intent_cat_scores.get(ml_cat, 0),
                        "method_parts": ["ml", "semantic"],
                    }
                    if ml_cat in intent_cat_scores:
                        candidates[ml_cat]["method_parts"].append("intent")
                    if ml_cat in tag_cat_hits:
                        candidates[ml_cat]["method_parts"].append("tag")

        # Inject intent-detected categories missing entirely
        for intent_cat, intent_sc in intent_cat_scores.items():
            if intent_cat not in candidates and intent_sc >= 0.50:
                candidates[intent_cat] = {
                    "category": intent_cat,
                    "semantic_score": cat_sem_scores.get(intent_cat, 0),
                    "ml_score": ml_cat_scores.get(intent_cat, 0),
                    "intent_score": intent_sc,
                    "method_parts": ["intent"],
                }
                if intent_cat in tag_cat_hits:
                    candidates[intent_cat]["method_parts"].append("tag")

        # Inject tag-discovered categories missing from all other methods
        # Only if 2+ tags match (single tag match is too weak on its own)
        for tag_cat, hits in tag_cat_hits.items():
            if tag_cat not in candidates and hits >= 2:
                sem = cat_sem_scores.get(tag_cat, 0)
                candidates[tag_cat] = {
                    "category": tag_cat,
                    "semantic_score": sem,
                    "ml_score": ml_cat_scores.get(tag_cat, 0),
                    "intent_score": intent_cat_scores.get(tag_cat, 0),
                    "method_parts": ["tag"],
                }

        if not candidates:
            return []

        # ── Step 5: Calculate score gap for each candidate ──
        # Gap = how far ahead this candidate is vs the runner-up (for #1) or vs #1 (for others)
        all_sem = sorted(cat_sem_scores.values(), reverse=True)
        best_sem = all_sem[0] if all_sem else 0
        runner_sem = all_sem[1] if len(all_sem) > 1 else 0
        top_gap = best_sem - runner_sem

        # ── Step 6: Score each candidate through ConfidenceScorer ──
        scorer_input = []
        for cat, info in candidates.items():
            # Gap: #1 gets the actual gap, others get 0
            is_top = (info["semantic_score"] == best_sem)
            gap = top_gap if is_top else 0

            scorer_input.append({
                "category": cat,
                "semantic_score": info["semantic_score"],
                "ml_score": info["ml_score"],
                "intent_score": info["intent_score"],
                "n_agreeing": len(info["method_parts"]),
                "gap": gap,
                "method": "+".join(info["method_parts"]),
            })

        scored = ConfidenceScorer.score_multi(scorer_input, is_bangla=is_bangla)

        # ── Step 7: Filter by effective_min_conf and gap_threshold ──
        if not scored or scored[0][1] < effective_min_conf:
            return []

        best_conf = scored[0][1]
        results = []
        for cat, conf, method in scored:
            if conf < effective_min_conf:
                continue
            if best_conf - conf > gap_threshold:
                break
            results.append((cat, conf, method))
            if len(results) >= max_cats:
                break

        return results

    # ========== Feature 2: "Did you mean?" Suggestions ==========

    def get_suggestions(self, cleaned_question, top_n=3):
        """Return top-N category suggestions when confidence is low.
        Returns: [{"category": str, "score": float, "sample_question": str}, ...]
        """
        if self.faiss_index is None:
            return []

        query_emb = self.encoder.encode(cleaned_question)
        # Search more than needed to allow deduplication by category
        scores, indices = self._faiss_search(query_emb, top_k=30)

        seen_cats = set()
        suggestions = []

        for j, idx in enumerate(indices):
            if idx < 0:
                continue
            cat = self.labels[idx]
            score = float(scores[j])
            if score < 0.15:
                break
            if cat not in seen_cats:
                seen_cats.add(cat)
                suggestions.append({
                    "category": cat,
                    "score": round(score, 3),
                    "sample_question": self.questions[idx]
                })
                if len(suggestions) >= top_n:
                    break

        return suggestions

    # ========== Math Calculator ==========

    _MATH_PATTERN = re.compile(
        r"^[\d\s\+\-\*\/\%\.\(\)]+$"
    )
    # Bangla math: "5+7 koto?", "30% of 150 koto?", "50-20 koto?"
    _BANGLA_MATH = re.compile(
        r"^([\d\s\+\-\*\/\%\.\(\)]+)\s*(koto|er|=)\s*\??$",
        re.IGNORECASE
    )
    # English: "what is 5+7", "calculate 30% of 150", "25% of 200"
    _ENGLISH_MATH = re.compile(
        r"^(?:what is |calculate |whats )?([\d\s\+\-\*\/\%\.\(\)]+)$",
        re.IGNORECASE
    )
    # Percentage: "30% of 150", "25% of 200"
    _PERCENT_OF = re.compile(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:of|er)\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE
    )
    # Combined math: "40+60 and 25% of 200"
    _MULTI_MATH = re.compile(
        r"^[\d\s\+\-\*\/\%\.\(\)]+(?:\s+and\s+[\d\s\+\-\*\/\%\.\(\)of]+)+$",
        re.IGNORECASE
    )

    def _try_math(self, text):
        """Try to evaluate math expressions. Returns result string or None."""
        cleaned = text.strip().rstrip("?").strip()

        # Handle "X% of Y" patterns first
        percent_matches = list(self._PERCENT_OF.finditer(cleaned))
        if percent_matches:
            results = []
            remaining = cleaned
            for m in percent_matches:
                pct = float(m.group(1))
                base = float(m.group(2))
                result = (pct / 100) * base
                results.append(f"{pct}% of {base} = {result:g}")
                remaining = remaining.replace(m.group(0), "").strip()

            # Check if there's also a simple expression (e.g. "40+60 and 25% of 200")
            remaining = re.sub(r"^and\s+|and\s*$", "", remaining).strip()
            if remaining and re.match(r"^[\d\s\+\-\*\/\.\(\)]+$", remaining):
                try:
                    # Safe eval: only allow digits and math operators
                    if not re.search(r"[a-zA-Z_]", remaining):
                        val = eval(remaining)
                        results.insert(0, f"{remaining.strip()} = {val:g}")
                except:
                    pass

            if results:
                return ", ".join(results)

        # Handle Bangla: "5+7 koto?"
        m = self._BANGLA_MATH.match(cleaned)
        if m:
            expr = m.group(1).strip()
        else:
            # Handle English: "what is 5+7", "calculate 50-20"
            m = self._ENGLISH_MATH.match(cleaned)
            if m:
                expr = m.group(1).strip()
            elif self._MATH_PATTERN.match(cleaned):
                expr = cleaned
            else:
                return None

        # Safe eval: reject anything with letters (prevents code injection)
        if re.search(r"[a-zA-Z_]", expr):
            return None

        try:
            result = eval(expr)
            return f"{expr.strip()} = {result:g}"
        except:
            return None

    # ── Fix #7: Banglish lexicon — common Bangla phrases → category ──
    _BANGLISH_LEXICON = {
        # Greetings
        "kemon acho": "greeting", "kemon achen": "greeting",
        "ki holo": "greeting", "ki khobor": "greeting",
        "assalamu alaikum": "greeting", "nomoskar": "greeting",
        # Thanks / Farewell
        "dhonnobad": "thanks", "shukriya": "thanks", "onek thanks": "thanks",
        "bidai": "farewell", "abar dekha hobe": "farewell",
        # Emotions
        "mon kharap": "emotions", "bhalo lagche na": "mental_health",
        "onek kosto": "mental_health", "ekla lagche": "loneliness",
        "khub sad": "emotions", "onek tension": "stress_management",
        "ami stressed": "stress_management", "onek pressure": "stress_management",
        # Health
        "matha byatha": "headache", "matha ghurche": "headache",
        "pet byatha": "food_poisoning", "pet kharap": "food_poisoning",
        "ghum hoy na": "insomnia", "ghum ashche na": "insomnia",
        "jor hoyeche": "health", "gorrom lagche": "health",
        "komor byatha": "joint_pain", "hatu byatha": "joint_pain",
        # Motivation
        "ki korbo": "motivation", "bujhi na": "motivation",
        "kono asha nai": "motivation", "ar pari na": "motivation",
        "himmat hariye felchi": "motivation",
        # Bot
        "tomar nam ki": "bot_name", "tumi ke": "about_bot",
        "tumi ki paro": "bot_capability",
        # Misc
        "bored lagche": "entertainment", "time pass": "entertainment",
        "ghumote chai": "sleep_hygiene", "khida lagche": "cooking",
    }

    def _banglish_lookup(self, text):
        """Fix #7: Direct Banglish phrase → category lookup."""
        text_lower = text.lower().strip()
        for phrase, cat in self._BANGLISH_LEXICON.items():
            if phrase in text_lower:
                return cat
        return None

    # ── Fix #5: LLM answer topic coherence validation ──
    def _validate_llm_answer(self, answer, category, question):
        """Check if LLM answer is actually about the right topic."""
        if not self.encoder.available:
            return True
        try:
            ans_emb = self.encoder.encode(answer)
            cat_answers = self._get_store_for(category).get_answers(category)
            if not cat_answers or len(cat_answers) < 2:
                return True
            cat_embs = self.encoder.encode(cat_answers[:5])
            similarity = float(np.dot(cat_embs, ans_emb.flatten()).max())
            if similarity < 0.25:
                logging.info(f"LLM answer rejected: off-topic (sim={similarity:.2f}) for {category}")
                return False
            return True
        except Exception:
            return True  # Don't block on validation errors

    def _is_absurd_question(self, text):
        """Check if a question is about fictional/impossible things that can't be answered."""
        absurd_patterns = [
            r"president of mars",
            r"king of (mars|earth|moon|sun)",
            r"moon er currency",
            r"invented oxygen",
            r"discovered dark fire",
            r"xyzland",
            r"ice diye toyri",
        ]
        text_lower = text.lower().strip()
        for pattern in absurd_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    # ========== Hybrid RAG: Retrieve context for LLM ==========

    def _retrieve_rag_context(self, question, top_n=5):
        """Retrieve top-N semantic search results with their answers for RAG.
        Uses FAISS for fast retrieval.
        Returns list of {"question": str, "answer": str, "category": str, "score": float}
        """
        if self.faiss_index is None:
            return []

        query_emb = self.encoder.encode(question)
        # Fetch more than needed to allow deduplication
        scores, indices = self._faiss_search(query_emb, top_k=top_n * 3)

        results = []
        seen = set()  # avoid duplicate answers
        for j, idx in enumerate(indices):
            if idx < 0 or len(results) >= top_n:
                break
            score = float(scores[j])
            if score < 0.20:
                break

            cat = self.labels[idx]
            matched_q = self.questions[idx]

            # Get the best answer for this category
            store = self._get_store_for(cat)
            answers = store.get_answers(cat)
            if not answers:
                continue

            # Pick best answer by semantic match to user question
            a_embs = self.encoder.encode(answers)
            a_scores = self.encoder.similarity(query_emb, a_embs)
            best_a_idx = a_scores.argmax()
            best_answer = answers[best_a_idx]

            # Deduplicate
            key = f"{cat}:{best_answer[:50]}"
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "question": matched_q,
                "answer": best_answer,
                "category": cat,
                "score": round(score, 3)
            })

        # Optional: re-rank with cross-encoder for better precision
        if self.reranker.available and len(results) > 2:
            results = self.reranker.rerank(question, results, top_n=top_n)

        return results

    # Patterns for cleaning LLM output artifacts
    _LLM_CLEAN_PATTERNS = [
        (re.compile(r"\[[\w_]+\]\s*Q:", re.IGNORECASE), ""),              # [category] Q:
        (re.compile(r"Retrieved knowledge about[^:]*:", re.IGNORECASE), ""),  # Retrieved knowledge about...
        (re.compile(r"^Question:.*$", re.MULTILINE | re.IGNORECASE), ""),  # Question: ... lines
        (re.compile(r"</s>"), ""),                                         # </s> tokens
        (re.compile(r"<\|[^|]+\|>"), ""),                                  # <|system|>, <|user|> etc
        (re.compile(r"^\s*\d+\.\s*\[[\w_]+\]", re.MULTILINE), ""),       # 1. [category_name]
        (re.compile(r"^\s*A:\s*", re.MULTILINE), ""),                      # A: prefix
        (re.compile(r"^(Answer|Response|Adapted Knowledge Base Response)\s*:\s*", re.IGNORECASE), ""),  # Answer:/Response: prefix
        (re.compile(r"^The question is [\"']?[^\"']*[\"']?\s*", re.IGNORECASE), ""),  # "The question is X" meta-talk
        (re.compile(r"\n{3,}"), "\n\n"),                                   # Triple+ newlines → double
    ]

    def _clean_llm_output(self, text):
        """Strip leaked internal format and detect garbage LLM output.
        Returns None if output is garbage → caller falls back to dataset answer.
        """
        if not text:
            return text

        # ── Garbage detection (before cleaning) ──

        # Check 1: Hindi/Devanagari hallucination — bot should output English/Bangla only
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        if devanagari_chars > 5:
            logging.info(f"LLM garbage rejected: Hindi hallucination ({devanagari_chars} devanagari chars)")
            return None

        # Check 2: Too little real content (latin + bangla)
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        if latin_chars + bangla_chars < 15:
            logging.info(f"LLM garbage rejected: too few real chars (latin={latin_chars}, bangla={bangla_chars})")
            return None

        # Check 3: Repetitive output — single word repeated 4+ times
        words = text.lower().split()
        if words:
            from collections import Counter
            word_counts = Counter(words)
            for word, count in word_counts.items():
                if count > 4 and len(word) > 2:
                    logging.info(f"LLM garbage rejected: '{word}' repeated {count} times")
                    return None

        # ── Format cleaning ──
        for pattern, replacement in self._LLM_CLEAN_PATTERNS:
            text = pattern.sub(replacement, text)

        # Strip "Yes, " / "Yes. " prefix when it's not answering a yes/no question
        # TinyLlama adds "Yes, " to almost everything
        if re.match(r"^Yes[,.]?\s+", text, re.IGNORECASE):
            text = re.sub(r"^Yes[,.]?\s+", "", text, count=1, flags=re.IGNORECASE)
            # Capitalize first letter after stripping
            if text:
                text = text[0].upper() + text[1:]

        # Clean up leftover whitespace
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]  # remove empty lines
        text = "\n".join(lines)
        # Final trim
        text = text.strip()
        # If cleaning removed everything, return None so fallback kicks in
        return text if len(text) >= 10 else None

    def _retrieve_rag_context_single(self, question, target_category, top_n=5):
        """Retrieve top-N semantic results ONLY from the target category.
        Prevents topic bleeding by restricting RAG context to one category.
        Falls back to _retrieve_rag_context() if target has too few matches.
        """
        if self.faiss_index is None:
            return []

        query_emb = self.encoder.encode(question)
        # Fetch more to filter by category
        scores, indices = self._faiss_search(query_emb, top_k=top_n * 6)

        results = []
        seen = set()
        for j, idx in enumerate(indices):
            if idx < 0 or len(results) >= top_n:
                break
            score = float(scores[j])
            if score < 0.15:
                break

            cat = self.labels[idx]
            # Only keep results from the target category
            if cat != target_category:
                continue

            matched_q = self.questions[idx]
            store = self._get_store_for(cat)
            answers = store.get_answers(cat)
            if not answers:
                continue

            a_embs = self.encoder.encode(answers)
            a_scores = self.encoder.similarity(query_emb, a_embs)
            best_a_idx = a_scores.argmax()
            best_answer = answers[best_a_idx]

            key = f"{cat}:{best_answer[:50]}"
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "question": matched_q,
                "answer": best_answer,
                "category": cat,
                "score": round(score, 3)
            })

        # Fallback: if single category has <2 results, supplement with category's full answer pool
        if len(results) < 2:
            store = self._get_store_for(target_category)
            answers = store.get_answers(target_category)
            if answers:
                for ans in answers[:top_n - len(results)]:
                    key = f"{target_category}:{ans[:50]}"
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            "question": question,
                            "answer": ans,
                            "category": target_category,
                            "score": 0.5
                        })

        # Optional: re-rank with cross-encoder for better precision
        if self.reranker.available and len(results) > 2:
            results = self.reranker.rerank(question, results, top_n=top_n)

        return results

    # ========== Answer Finding & Refinement ==========

    def _find_best_answer(self, category, question, session_id=None):
        """Find best answer using semantic similarity between question and answers.
        Falls back to random selection if encoder unavailable (Tier 4).
        """
        store = self._get_store_for(category)
        answers = store.get_answers(category)
        if not answers:
            return None

        if len(answers) == 1:
            return answers[0]

        # Tier 4 fallback: random answer if encoder not available
        if not self.encoder.available:
            return random.choice(answers)

        # Semantic scoring: compare question embedding with answer embeddings
        q_emb = self.encoder.encode(question)
        a_embs = self.encoder.encode(answers)
        scores = self.encoder.similarity(q_emb, a_embs)

        # Sort by score
        scored = sorted(zip(scores, answers), key=lambda x: x[0], reverse=True)

        # Avoid recent answers in session
        if session_id:
            history = self.db.get_history(session_id, limit=3)
            recent_replies = set()
            for turn in history:
                if turn["bot"]:
                    # Strip template variations to compare base content
                    recent_replies.add(turn["bot"][:80])

            for score, ans in scored:
                if ans[:80] not in recent_replies:
                    return ans

        return scored[0][1]

    def _find_multi_answers(self, categories, question, session_id=None):
        """Get best answer from each detected category.
        Returns list of (category, answer, confidence) tuples.
        """
        results = []
        for cat, conf, method in categories:
            ans = self._find_best_answer(cat, question, session_id)
            if ans:
                results.append((cat, ans, conf))
        return results

    MERGE_CONNECTORS = [
        "Also regarding {cat}: ",
        "On the {cat} side: ",
        "About {cat}: ",
        "For {cat}: ",
        "Regarding {cat}: ",
    ]

    def _merge_answers(self, cat_answers, sentiment="neutral"):
        """Merge answers from multiple categories into one coherent response.
        cat_answers: list of (category, answer, confidence)
        Returns: (merged_answer, primary_category, all_categories, avg_confidence)
        """
        if not cat_answers:
            return None, None, [], 0

        # Single category — normal flow, no merging needed
        if len(cat_answers) == 1:
            cat, ans, conf = cat_answers[0]
            return ans, cat, [cat], conf

        # Multi-category: primary gets full answer, others get first sentence
        primary_cat, primary_ans, primary_conf = cat_answers[0]

        # Build merged response
        parts = [primary_ans]

        for cat, ans, conf in cat_answers[1:]:
            # Take first sentence (or first 150 chars) from secondary answers
            sentences = re.split(r'(?<=[.!?])\s+', ans)
            snippet = sentences[0] if sentences else ans
            if len(snippet) > 150:
                snippet = snippet[:147] + "..."

            # Check snippet is meaningfully different from primary
            # (avoid repeating nearly identical info)
            if self._text_overlap(primary_ans, snippet) < 0.5:
                connector = random.choice(self.MERGE_CONNECTORS).format(cat=cat.replace("_", " "))
                parts.append(connector + snippet)

        merged = "\n\n".join(parts)
        all_cats = [c[0] for c in cat_answers]
        avg_conf = sum(c[2] for c in cat_answers) / len(cat_answers)

        return merged, primary_cat, all_cats, round(avg_conf, 3)

    @staticmethod
    def _text_overlap(text_a, text_b):
        """Simple word overlap ratio between two texts. Returns 0.0-1.0."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        smaller = min(len(words_a), len(words_b))
        return len(intersection) / smaller if smaller > 0 else 0.0

    def _refine_answer(self, answer, question, sentiment, category, session_id=None):
        """Refine answer with templates + sentiment adjustment."""
        store = self._get_store_for(category)
        answers_pool = store.get_answers(category)

        # Feature 3: Answer templates — generate diverse response
        answer = self.templates.generate(
            base_answer=answer,
            sentiment=sentiment,
            answers_pool=answers_pool,
            question=question
        )

        # Sentiment adjustment (for strong emotions)
        if sentiment in ("angry", "sad"):
            answer = self.sentiment.adjust_response(answer, sentiment)

        return answer

    def _store_feeling(self, category, question, sentiment, answer):
        if sentiment != "neutral":
            store = self._get_store_for(category)
            store.add_feeling(category, question, sentiment, answer)

    # ========== Learning (Background Queue) ==========

    # Batch threshold: retrain only after N pending learns accumulate
    LEARN_BATCH_THRESHOLD = 5

    def learn(self, question, category, answer):
        """Queue new Q&A for batch learning.
        Saves to category file immediately (so answers are available right away)
        but defers the expensive retrain (FAISS index + ML classifier) to batch processing.
        This prevents system lockups during learn() calls.
        """
        question = question.lower().strip()
        category = category.lower().strip()

        # Save to category file immediately — data is available for direct lookups
        store = self._get_store_for(category)
        if store.category_exists(category):
            store.add_question(category, question)
            if answer:
                store.add_answer(category, answer)
        else:
            self.general_store.create_category(category, "general", [question], [answer])
            self.category_store_map[category] = self.general_store

        # Queue for batch processing (deferred retrain)
        self.db.add_pending_learn(question, category, answer)
        # Also log to learn_log for history
        self.db.log_learn(question, category, answer)

        self.learn_count += 1
        logging.info(f"QUEUED_LEARN | Q: {question} | Category: {category} | Pending: {self.db.get_pending_count()}")

        # Auto-trigger batch if threshold reached
        pending_count = self.db.get_pending_count()
        if pending_count >= self.LEARN_BATCH_THRESHOLD:
            logging.info(f"Auto-batch triggered: {pending_count} pending learns")
            self.process_pending_learns()

        return True

    def process_pending_learns(self):
        """Batch process all pending learns: retrain FAISS + ML once for all queued items.
        Called automatically when threshold is reached, or manually via API.
        Much more efficient than retraining after every single learn().
        """
        pending = self.db.get_pending_learns()
        if not pending:
            logging.info("No pending learns to process")
            return {"processed": 0}

        count = len(pending)
        ids = [p["id"] for p in pending]

        # Single retrain covers all pending items
        self.retrain()

        # Mark all as processed
        self.db.mark_pending_processed(ids)

        logging.info(f"BATCH_LEARN | Processed {count} pending learns | Retrained model")
        return {"processed": count}

    def handle_feedback(self, question, bot_answer, intent, feedback_type, correct_answer=None, correct_category=None):
        """Handle user feedback. Save to SQLite. Re-learn on correction."""
        self.db.add_feedback(question, bot_answer, intent, feedback_type, correct_answer, correct_category)
        logging.info(
            f"FEEDBACK | {feedback_type.upper()} | Q: {question} | Intent: {intent}"
            + (f" | Correct: {correct_answer}" if correct_answer else "")
        )

        if feedback_type == "dislike" and correct_answer:
            category = correct_category or intent or "general"
            self.learn(question, category, correct_answer)
            return {"status": "corrected", "message": "Dhonnobad! Bhul theke shikhlam!"}

        if feedback_type == "like":
            return {"status": "liked", "message": "Thanks for the feedback!"}

        return {"status": "noted", "message": "Feedback noted."}

    # ── Fix #6: Weak categories detection from dislike feedback ──
    def get_weak_categories(self, min_dislikes=3):
        """Find categories with high dislike rates for targeted improvement."""
        rows = self.db.conn.execute("""
            SELECT intent, COUNT(*) as dislikes,
                   (SELECT COUNT(*) FROM feedback f2
                    WHERE f2.intent = f1.intent AND f2.feedback_type='like') as likes
            FROM feedback f1
            WHERE feedback_type='dislike' AND intent IS NOT NULL AND intent != ''
            GROUP BY intent
            HAVING dislikes >= ?
            ORDER BY dislikes DESC
        """, (min_dislikes,)).fetchall()

        weak = []
        for r in rows:
            total = r["dislikes"] + (r["likes"] or 0)
            dislike_rate = r["dislikes"] / total if total > 0 else 1.0
            if dislike_rate > 0.4:  # 40%+ dislike rate = weak category
                weak.append({
                    "category": r["intent"],
                    "dislikes": r["dislikes"],
                    "likes": r["likes"] or 0,
                    "dislike_rate": round(dislike_rate * 100),
                })
        return weak

    # ========== Main Answer Pipeline ==========

    def _tier4_keyword_match(self, question):
        """Tier 4 fallback: simple keyword matching against category names and questions.
        Used when no embedding model / FAISS is available.
        Returns (category, answer) or (None, None).
        """
        q_words = set(question.lower().split())
        best_cat = None
        best_overlap = 0

        for cat_name, store in self.category_store_map.items():
            # Match against category name words
            cat_words = set(cat_name.replace("_", " ").split())
            overlap = len(q_words & cat_words)

            # Also check question words in the category's questions
            cat_data = store.get(cat_name)
            if cat_data:
                for stored_q in cat_data.get("questions", [])[:10]:
                    stored_words = set(stored_q.lower().split())
                    q_overlap = len(q_words & stored_words)
                    overlap = max(overlap, q_overlap)

            if overlap > best_overlap:
                best_overlap = overlap
                best_cat = cat_name

        if best_cat and best_overlap >= 1:
            answers = self._get_store_for(best_cat).get_answers(best_cat)
            if answers:
                answer = random.choice(answers)
                return best_cat, answer

        return None, None

    def get_answer(self, user_question, session_id=None, skip_llm=False):
        """Main answer pipeline with 4-tier fallback:

        Tier 1: Phi-3 LLM + FAISS + ML (best quality)
        Tier 2: TinyLlama LLM + FAISS + ML (good quality)
        Tier 3: FAISS + ML only, no LLM (fast, dataset answers)
        Tier 4: Keyword matching only (no models at all, fixed dataset)

        Returns dict: {"reply": str, "intent": str|list, "confidence": float, "categories": list}
                   or {"suggestions": [...], "needs_learning": True}
                   or None
        """
        original = user_question.strip()
        cleaned = re.sub(r"[^\w\s]", "", original.lower()).strip()

        # ── Math calculator — catch "5+7 koto?", "30% of 150", "80+20" ──
        math_result = self._try_math(original)
        if math_result is not None:
            reply = f"Answer: {math_result}"
            if session_id:
                self.db.add_turn(session_id, original, reply, "calculator", 1.0, "neutral")
            logging.info(f"Q: {original} | CALCULATOR | Result: {math_result}")
            return {"reply": reply, "intent": "calculator", "confidence": 1.0, "categories": ["calculator"]}

        # Meta-command check — "eta explain koro", "example dao", "abar bolo"
        # These should reuse last_intent WITHOUT topic similarity check
        if is_meta_command(cleaned):
            last_intent = self.db.get_last_intent(session_id) if session_id else None
            if last_intent and last_intent in self.category_store_map:
                answer = self._find_best_answer(last_intent, cleaned, session_id)
                if answer:
                    sentiment = self.sentiment.detect(cleaned)
                    answer = self._refine_answer(answer, cleaned, sentiment, last_intent, session_id)
                    self._store_feeling(last_intent, original, sentiment, answer)
                    if session_id:
                        self.db.add_turn(session_id, original, answer, last_intent, 1.0, sentiment)
                    logging.info(f"Q: {original} | META-COMMAND | Intent: {last_intent}")
                    return {"reply": answer, "intent": last_intent, "confidence": 1.0, "categories": [last_intent]}

        # Follow-up check — with topic similarity verification
        if is_follow_up(cleaned):
            last_intent = self.db.get_last_intent(session_id) if session_id else None
            if last_intent and last_intent in self.category_store_map:
                # Verify the current question is actually related to last_intent
                # "messi vs ronaldo" after "quantum computing" should NOT be a follow-up
                topic_sim = 1.0  # Default: trust follow-up if no encoder
                if self.encoder.available:
                    topic_name = last_intent.replace("_", " ")
                    q_emb = self.encoder.encode(cleaned)
                    topic_emb = self.encoder.encode(topic_name)
                    topic_sim = float(np.dot(q_emb.flatten(), topic_emb.flatten()))

                if topic_sim >= 0.45:
                    answer = self._find_best_answer(last_intent, cleaned, session_id)
                    if answer:
                        sentiment = self.sentiment.detect(cleaned)
                        answer = self._refine_answer(answer, cleaned, sentiment, last_intent, session_id)
                        self._store_feeling(last_intent, original, sentiment, answer)
                        if session_id:
                            self.db.add_turn(session_id, original, answer, last_intent, 1.0, sentiment)
                        logging.info(f"Q: {original} | FOLLOW-UP | Intent: {last_intent} | TopicSim: {topic_sim:.2f}")
                        return {"reply": answer, "intent": last_intent, "confidence": 1.0, "categories": [last_intent]}
                else:
                    logging.info(f"Q: {original} | FOLLOW-UP rejected | Intent: {last_intent} | TopicSim: {topic_sim:.2f} < 0.45")

        # ── Fix #7: Banglish lexicon — direct phrase matching before any detection ──
        banglish_cat = self._banglish_lookup(cleaned)
        if banglish_cat and banglish_cat in self.category_store_map:
            answer = self._find_best_answer(banglish_cat, cleaned, session_id)
            if answer:
                sentiment = self.sentiment.detect(cleaned)
                answer = self._refine_answer(answer, cleaned, sentiment, banglish_cat, session_id)
                self._store_feeling(banglish_cat, original, sentiment, answer)
                if session_id:
                    self.db.add_turn(session_id, original, answer, banglish_cat, 0.85, sentiment)
                logging.info(f"Q: {original} | BANGLISH_LEXICON | Intent: {banglish_cat}")
                return {"reply": answer, "intent": banglish_cat, "confidence": 0.85, "categories": [banglish_cat]}

        # ── Fix #1: Detect on RAW text first, spell-correct only if low confidence ──
        detected = self._detect_categories(cleaned, max_cats=3)
        raw_conf = detected[0][1] if detected else 0

        # Only spell-correct if raw detection is weak (< 0.40)
        if raw_conf < 0.40:
            corrected = self.spell.correct(cleaned)
            if corrected != cleaned:
                detected_cor = self._detect_categories(corrected, max_cats=3)
                if detected_cor:
                    cor_best = detected_cor[0][1]
                    if cor_best > raw_conf:
                        detected = detected_cor
                        logging.info(f"Spell correction helped: '{cleaned}' → '{corrected}' | {raw_conf:.0%} → {cor_best:.0%}")
        else:
            corrected = cleaned  # No correction needed

        # Step 1b: Context-aware query enrichment (Fix #4: topic-shift guard)
        # ONLY triggers when confidence is very low AND question shares words with previous
        CONTEXT_ENRICH_THRESHOLD = 0.28
        best_conf = detected[0][1] if detected else 0
        if session_id and best_conf < CONTEXT_ENRICH_THRESHOLD:
            history = self.db.get_history(session_id, limit=2)
            if history:
                last_user_msg = None
                for turn in reversed(history):
                    if turn.get("user"):
                        last_user_msg = turn["user"]
                        break

                if last_user_msg:
                    last_cleaned = re.sub(r"[^\w\s]", "", last_user_msg.lower()).strip()
                    context_words = [w for w in last_cleaned.split()
                                     if len(w) > 2 and w not in {"what", "how", "why", "the", "and", "for", "tell", "about",
                                                                   "ki", "keno", "kivabe", "kothay", "ami", "tumi"}]
                    # Fix #4: Topic-shift detection — only enrich if questions share meaningful words
                    curr_words = set(cleaned.split()) - {"ki", "keno", "how", "what", "why", "the"}
                    last_words = set(last_cleaned.split()) - {"ki", "keno", "how", "what", "why", "the"}
                    shared_words = curr_words & last_words

                    if context_words and len(shared_words) >= 1:
                        enriched = " ".join(context_words[:4]) + " " + cleaned
                        detected_enriched = self._detect_categories(enriched, max_cats=3)
                        if detected_enriched:
                            enriched_best = detected_enriched[0][1]
                            if enriched_best > best_conf:
                                detected = detected_enriched
                                logging.info(
                                    f"Context enrichment: '{cleaned}' → '{enriched}' | "
                                    f"Confidence: {best_conf:.0%} → {enriched_best:.0%}"
                                )
                    elif context_words and not shared_words:
                        logging.info(f"Context enrichment SKIPPED (topic shift): '{cleaned}' vs '{last_cleaned}'")

        # No categories found — try emotional fallback, then suggestions, then unknown
        if not detected:
            # Absurd question check: catch fictional/impossible questions early
            if not EMOTIONAL_FALLBACK.search(cleaned) and self._is_absurd_question(cleaned):
                logging.info(f"ABSURD_QUESTION | Q: {original}")
                if session_id:
                    self.db.add_turn(session_id, original, None, None, 0, "neutral")
                return {
                    "reply": "Hmm, eta amar knowledge er baire! Ami factual topics niye help korte pari — coding, AI, health, finance, career etc. Onno kichu jante chaile bolo!",
                    "intent": "unknown",
                    "confidence": 0.0,
                    "categories": []
                }

            # Emotional fallback: route distress/motivation to mental health categories
            if EMOTIONAL_FALLBACK.search(cleaned):
                for fallback_cat in ["motivation", "mental_health", "emotions", "stress_management", "loneliness"]:
                    if fallback_cat in self.category_store_map:
                        answer = self._find_best_answer(fallback_cat, cleaned, session_id)
                        if answer:
                            sentiment = self.sentiment.detect(cleaned)
                            answer = self._refine_answer(answer, cleaned, sentiment, fallback_cat, session_id)
                            self._store_feeling(fallback_cat, original, sentiment, answer)
                            if session_id:
                                self.db.add_turn(session_id, original, answer, fallback_cat, 0.5, sentiment)
                            logging.info(f"Q: {original} | EMOTIONAL_FALLBACK | Intent: {fallback_cat}")
                            return {"reply": answer, "intent": fallback_cat, "confidence": 0.5, "categories": [fallback_cat]}

            # Bangla keyword extraction: strip Bangla filler words, re-detect with just the technical terms
            bangla_filler = {"ki", "keno", "kivabe", "kothay", "kokhon", "ke", "ta", "er", "e", "te", "na",
                             "ar", "ba", "o", "r", "ami", "tumi", "apni", "ache", "hole", "kore", "korte",
                             "korbo", "hoy", "hobe", "paro", "parbo", "diye", "niye", "theke", "jante",
                             "bolo", "bolte", "lagbe", "chai", "chao", "shikhte", "shikhao", "start",
                             "kon", "kono", "onek", "khub", "ektu", "amr", "amar", "tomar"}
            tech_words = [w for w in cleaned.split() if w not in bangla_filler and len(w) > 2]
            if tech_words and len(tech_words) < len(cleaned.split()):
                tech_query = " ".join(tech_words)
                detected_tech = self._detect_categories(tech_query, max_cats=3)
                if detected_tech and detected_tech[0][1] > 0.35:
                    detected = detected_tech
                    logging.info(f"Bangla keyword extraction: '{cleaned}' → '{tech_query}' | Conf: {detected_tech[0][1]:.0%}")

        if not detected:
            suggestions = self.get_suggestions(cleaned)
            if suggestions and suggestions[0]["score"] >= 0.20:
                logging.info(f"SUGGESTIONS | Q: {original} | Top: {suggestions[0]}")
                if session_id:
                    self.db.add_turn(session_id, original, None, None, 0, "neutral")
                return {
                    "reply": None,
                    "suggestions": suggestions,
                    "needs_learning": True,
                    "question": original
                }
            # Tier 4 fallback: keyword matching (no AI models needed)
            t4_cat, t4_answer = self._tier4_keyword_match(cleaned)
            if t4_cat and t4_answer:
                sentiment = self.sentiment.detect(cleaned)
                t4_answer = self._refine_answer(t4_answer, cleaned, sentiment, t4_cat, session_id)
                if session_id:
                    self.db.add_turn(session_id, original, t4_answer, t4_cat, 0.3, sentiment)
                logging.info(f"Q: {original} | TIER4_KEYWORD | Intent: {t4_cat}")
                return {"reply": t4_answer, "intent": t4_cat, "confidence": 0.3, "categories": [t4_cat]}

            # Truly unknown
            logging.info(f"UNANSWERED | Q: {original}")
            if session_id:
                self.db.add_turn(session_id, original, None, None, 0, "neutral")
            return None

        sentiment = self.sentiment.detect(cleaned)

        # Step 2: Find best answer from each category
        cat_answers = self._find_multi_answers(detected, corrected, session_id)

        if not cat_answers:
            primary_cat = detected[0][0]
            logging.info(f"NO_ANSWER | Q: {original} | Categories: {[d[0] for d in detected]}")
            if session_id:
                self.db.add_turn(session_id, original, None, primary_cat, detected[0][1], sentiment)
            return None

        # Step 3: Merge answers from multiple categories
        merged_answer, primary_cat, all_cats, avg_conf = self._merge_answers(cat_answers, sentiment)

        # Step 4: Hybrid RAG — TinyLlama generates from retrieved context
        # Skip LLM for simple conversational categories — dataset answers are better
        SKIP_LLM_CATS = {
            "greeting", "farewell", "thanks", "about_bot", "bot_name",
            "who", "emotions", "gratitude"
        }
        generated = False
        if self.generator.available and not skip_llm and not (set(all_cats) & SKIP_LLM_CATS):
            # Single-category RAG: only feed the PRIMARY category's answers to LLM
            # This prevents topic bleeding (e.g. carpooling answer talking about juggling)
            rag_context = self._retrieve_rag_context_single(original, primary_cat, top_n=5)

            # Sliding Context Window: fetch last 3-5 conversation turns for pronoun resolution
            conv_history = None
            if session_id:
                conv_history = self.db.get_history(session_id, limit=5)

            if rag_context:
                llm_answer = self.generator.generate_rag(
                    question=original,
                    retrieved_context=rag_context,
                    categories=[primary_cat],
                    conversation_history=conv_history
                )
                if llm_answer:
                    cleaned_llm = self._clean_llm_output(llm_answer)
                    # Fix #5: Validate LLM answer is about the right topic
                    if cleaned_llm and self._validate_llm_answer(cleaned_llm, primary_cat, original):
                        merged_answer = cleaned_llm
                        generated = True
                        logging.info(
                            f"RAG generated | Q: {original} | "
                            f"Context: {[c['category'] for c in rag_context]}"
                        )
                    else:
                        # LLM output was garbage — keep dataset answer (merged_answer unchanged)
                        logging.info(f"LLM output rejected, using dataset answer | Q: {original}")

        # Step 5: Fallback — refine dataset answer with templates (if TinyLlama didn't generate)
        if not generated:
            if len(all_cats) == 1:
                merged_answer = self._refine_answer(merged_answer, corrected, sentiment, primary_cat, session_id)
            else:
                if sentiment in ("angry", "sad"):
                    merged_answer = self.sentiment.adjust_response(merged_answer, sentiment)

        # Step 6: Store feeling for primary category
        self._store_feeling(primary_cat, original, sentiment, merged_answer)

        # Step 7: Save to SQLite
        intent_str = ",".join(all_cats)
        if session_id:
            self.db.add_turn(session_id, original, merged_answer, intent_str, avg_conf, sentiment)

        source = "llm" if generated else "dataset"
        logging.info(
            f"Q: {original} | Intent: {all_cats} | "
            f"Confidence: {avg_conf:.3f} | Sentiment: {sentiment} | "
            f"Source: {source} | Multi: {len(all_cats) > 1}"
        )

        return {
            "reply": merged_answer,
            "intent": all_cats[0] if len(all_cats) == 1 else all_cats,
            "confidence": avg_conf,
            "categories": all_cats,
            "generated": generated
        }

    # ========== Session (via SQLite) ==========

    @property
    def memory(self):
        """Backward compatibility — returns db as memory interface."""
        return self

    def create_session(self):
        return self.db.create_session()


# ============================================================
# CLI
# ============================================================

def run_chatbot():
    print("=" * 55)
    print("  Mini Chatbot (Semantic AI + ML + Self Learning)")
    print("  'exit' likho session sesh korte")
    print("=" * 55)

    bot = ChatBot()
    session_id = bot.db.create_session()
    total_cats = len(bot.category_store_map)
    general_count = len(bot.general_store.get_all_categories())
    special_count = sum(len(s.get_all_categories()) for s in bot.specialized_stores)
    print(f"  {len(bot.questions)} questions | {total_cats} categories")
    print(f"  General: {general_count} | Specialized: {special_count}")
    print(f"  Semantic: ONNX MiniLM-L6-v2 | SQLite: chatbot.db")
    print("=" * 55)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            if bot.learn_count > 0:
                print(f"\n[Info] Aj {bot.learn_count} ta notun jinish shikhechi!")
            stats = bot.db.get_feedback_stats()
            if stats["total"] > 0:
                print(f"[Info] Feedback: {stats['likes']} likes, {stats['dislikes']} dislikes")
            print("\nBot: Bye bye! Abar esho!")
            break

        result = bot.get_answer(user_input, session_id)

        if result is None:
            print("\nBot: Ami ei question er answer jani na. Amake shikhao!")
            print("     (skip korte khali Enter dao)\n")

            category = input("  Category ki hobe? (e.g. greeting, product, info): ").strip()
            if not category:
                print("\nBot: Okay, skip korlam. Onno kichu jiggesh koro!")
                continue

            answer = input("  Answer ki hobe?: ").strip()
            if not answer:
                print("\nBot: Okay, skip korlam. Onno kichu jiggesh koro!")
                continue

            bot.learn(user_input, category, answer)
            print(f"\nBot: Dhonnobad! Ami shikhe nilam!")

        elif result.get("suggestions") and result.get("reply") is None:
            # Feature 2: "Did you mean?"
            print("\nBot: Ami exactly bujhte parlam na. Apni ki eta jante chaichen?")
            for i, s in enumerate(result["suggestions"], 1):
                print(f"     {i}. {s['category']} (e.g. \"{s['sample_question']}\")")

            choice = input("\n  Number likho (1/2/3) or skip (Enter): ").strip()
            if choice in ("1", "2", "3"):
                idx = int(choice) - 1
                if idx < len(result["suggestions"]):
                    chosen_cat = result["suggestions"][idx]["category"]
                    # Re-answer with chosen category
                    store = bot._get_store_for(chosen_cat)
                    answers = store.get_answers(chosen_cat)
                    if answers:
                        answer = random.choice(answers)
                        sentiment = bot.sentiment.detect(user_input)
                        answer = bot._refine_answer(answer, user_input, sentiment, chosen_cat, session_id)
                        print(f"\nBot: {answer}")
                    else:
                        print("\nBot: Ei category te kono answer nai.")
            else:
                print("\nBot: Okay! Amake shikhao tahole!")
                category = input("  Category ki hobe?: ").strip()
                if category:
                    answer = input("  Answer ki hobe?: ").strip()
                    if answer:
                        bot.learn(user_input, category, answer)
                        print(f"\nBot: Dhonnobad! Shikhe nilam!")
        else:
            confidence = result.get("confidence", 0)
            categories = result.get("categories", [])
            conf_bar = "●" * int(confidence * 10) + "○" * (10 - int(confidence * 10))
            print(f"\nBot: {result['reply']}")
            if len(categories) > 1:
                cat_str = " + ".join(categories)
                print(f"     [{cat_str}] [{conf_bar}] {confidence:.0%} (multi-category)")
            else:
                intent = categories[0] if categories else result.get("intent", "")
                print(f"     [{intent}] [{conf_bar}] {confidence:.0%}")


if __name__ == "__main__":
    run_chatbot()
