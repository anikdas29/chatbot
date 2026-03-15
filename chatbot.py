"""
Mini Chatbot - NLP + ML + Semantic Search + Self Learning
ONNX MiniLM Embedding + ML Classifier + Spell Correction + Answer Templates
SQLite Persistence + "Did you mean?" Suggestions + Conversation Memory
Category-wise dataset with answer refinement and feeling tracking.
No third-party AI API needed.
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
    """ONNX MiniLM-L6-v2 embedding model.
    Converts text to 384-dim semantic vectors.
    Understands meaning: 'feeling sad' ≈ 'i am depressed' ≈ 'mon kharap'
    """

    def __init__(self, model_dir="models/minilm"):
        self.model_dir = model_dir
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=128)
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "onnx", "model.onnx"),
            providers=["CPUExecutionProvider"]
        )
        logging.info("SemanticEncoder loaded (ONNX MiniLM-L6-v2)")

    def encode(self, texts):
        """Encode list of texts into normalized embeddings. Returns (N, 384) numpy array."""
        if isinstance(texts, str):
            texts = [texts]
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
    """Known words er sathe match kore spelling fix kore"""

    def __init__(self, known_words):
        self.known_words = known_words

    def correct(self, text):
        words = text.lower().split()
        corrected = []
        for word in words:
            if word in self.known_words:
                corrected.append(word)
            else:
                matches = get_close_matches(word, self.known_words, n=1, cutoff=0.75)
                corrected.append(matches[0] if matches else word)
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
    """Category-wise dataset folder manage kore."""

    def __init__(self, folder="category_wise_dataset"):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.categories = {}
        self._load_all()

    def _cat_path(self, category):
        return os.path.join(self.folder, f"{category}.json")

    def _load_all(self):
        self.categories = {}
        for fname in os.listdir(self.folder):
            if fname.endswith(".json"):
                fpath = os.path.join(self.folder, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cat_name = data.get("category", fname.replace(".json", ""))
                    self.categories[cat_name] = data
                except (json.JSONDecodeError, KeyError):
                    logging.warning(f"Skipped invalid category file: {fname}")
        logging.info(f"CategoryStore loaded: {len(self.categories)} categories from {self.folder}/")

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
# Follow-up detection
# ============================================================

FOLLOW_UP_WORDS = {
    "tell me more", "more", "aro bolo", "ar bolo", "elaborate",
    "explain more", "details", "aro details", "bistorito",
    "what else", "ar ki", "continue", "go on", "then",
    "ar kono", "aro kichu", "bolo aro", "keep going"
}


def is_follow_up(text):
    cleaned = text.lower().strip()
    for phrase in FOLLOW_UP_WORDS:
        if phrase in cleaned:
            return True
    return False


# ============================================================
# Main ChatBot
# ============================================================

class ChatBot:
    def __init__(self, general_folder="category_wise_dataset", specialized_folders=None,
                 model_dir="models/minilm", db_path="chatbot.db"):
        """
        Full-featured chatbot with:
        1. ONNX MiniLM semantic embedding (replaces TF-IDF for search)
        2. "Did you mean?" suggestions
        3. Answer templates for diverse responses
        4. SQLite persistence for sessions, feedback, learning
        """
        # Database (Feature 4: SQLite)
        self.db = Database(db_path)

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

        # Category -> store mapping
        self.category_store_map = {}
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

        self.learn_count = 0
        logging.info("ChatBot initialized with semantic search + templates + SQLite")

    @staticmethod
    def _auto_detect_specialized_folders(general_folder):
        folders = []
        for item in os.listdir("."):
            if os.path.isdir(item) and item.endswith("_dataset") and item != general_folder:
                folders.append(item)
        folders.sort()
        return folders

    def _build_category_map(self):
        self.category_store_map = {}
        for cat in self.general_store.get_all_categories():
            self.category_store_map[cat] = self.general_store
        for store in self.specialized_stores:
            for cat in store.get_all_categories():
                self.category_store_map[cat] = store

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

    # ========== Feature 1: Semantic Index ==========

    def _build_semantic_index(self):
        """Pre-compute embeddings for all questions in dataset."""
        if self.questions:
            logging.info(f"Building semantic index for {len(self.questions)} questions...")
            # Batch encode in chunks to avoid memory issues
            batch_size = 64
            all_embeddings = []
            for i in range(0, len(self.questions), batch_size):
                batch = self.questions[i:i + batch_size]
                emb = self.encoder.encode(batch)
                all_embeddings.append(emb)
            self.question_embeddings = np.vstack(all_embeddings)
            logging.info(f"Semantic index built: {self.question_embeddings.shape}")
        else:
            self.question_embeddings = None

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
        """Semantic embedding search + ML classifier combined.
        Returns: (category, confidence, method) or (None, 0, None)
        """
        if self.question_embeddings is None:
            return None, 0, None

        # Semantic search
        query_emb = self.encoder.encode(cleaned_question)
        similarities = self.encoder.similarity(query_emb, self.question_embeddings)

        # Get top-5 matches
        top_indices = similarities.argsort()[-5:][::-1]
        top_scores = [(self.labels[i], float(similarities[i])) for i in top_indices]

        # Aggregate by category (multiple questions per category)
        cat_scores = {}
        for label, score in top_scores:
            if label not in cat_scores:
                cat_scores[label] = []
            cat_scores[label].append(score)

        # Best category by max score
        best_cat = max(cat_scores, key=lambda c: max(cat_scores[c]))
        best_score = max(cat_scores[best_cat])

        # ML classifier as secondary signal
        ml_intent = None
        ml_conf = 0
        if self.use_ml:
            ml_proba = self.intent_clf.predict_proba([cleaned_question])[0]
            ml_idx = ml_proba.argmax()
            ml_intent = self.intent_clf.classes_[ml_idx]
            ml_conf = ml_proba[ml_idx]

        # Decision logic
        # High semantic score — trust it
        if best_score >= 0.65:
            return best_cat, best_score, "semantic"

        # Semantic + ML agree — strong signal
        if best_cat == ml_intent and best_score >= 0.35:
            combined = best_score * 0.6 + ml_conf * 0.4
            return best_cat, combined, "semantic+ml"

        # ML top result is in semantic top-5 — boost
        if ml_intent and ml_intent in cat_scores and ml_conf >= 0.3:
            combined = max(cat_scores[ml_intent]) * 0.5 + ml_conf * 0.5
            if combined >= 0.35:
                return ml_intent, combined, "ml+semantic_top5"

        # Moderate semantic score
        if best_score >= 0.45:
            return best_cat, best_score, "semantic_moderate"

        # ML moderate confidence
        if self.use_ml and ml_conf >= 0.35:
            return ml_intent, ml_conf, "ml_fallback"

        return None, best_score, None

    # ========== Feature 2: "Did you mean?" Suggestions ==========

    def get_suggestions(self, cleaned_question, top_n=3):
        """Return top-N category suggestions when confidence is low.
        Returns: [{"category": str, "score": float, "sample_question": str}, ...]
        """
        if self.question_embeddings is None:
            return []

        query_emb = self.encoder.encode(cleaned_question)
        similarities = self.encoder.similarity(query_emb, self.question_embeddings)

        # Get top matches, deduplicate by category
        top_indices = similarities.argsort()[::-1]
        seen_cats = set()
        suggestions = []

        for idx in top_indices:
            cat = self.labels[idx]
            score = float(similarities[idx])
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

    # ========== Answer Finding & Refinement ==========

    def _find_best_answer(self, category, question, session_id=None):
        """Find best answer using semantic similarity between question and answers."""
        store = self._get_store_for(category)
        answers = store.get_answers(category)
        if not answers:
            return None

        if len(answers) == 1:
            return answers[0]

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

    # ========== Learning ==========

    def learn(self, question, category, answer):
        """Learn new Q&A. Save to category file + SQLite log + retrain."""
        question = question.lower().strip()
        category = category.lower().strip()

        store = self._get_store_for(category)
        if store.category_exists(category):
            store.add_question(category, question)
            if answer:
                store.add_answer(category, answer)
        else:
            self.general_store.create_category(category, "general", [question], [answer])
            self.category_store_map[category] = self.general_store

        # Log to SQLite
        self.db.log_learn(question, category, answer)

        self.retrain()
        self.learn_count += 1
        logging.info(f"LEARNED | Q: {question} | Category: {category} | Total learns: {self.learn_count}")
        return True

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

    # ========== Main Answer Pipeline ==========

    def get_answer(self, user_question, session_id=None):
        """Main answer pipeline:
        1. Follow-up check
        2. Spell correction
        3. Semantic category detection (ONNX embedding)
        4. If low confidence → return suggestions ("Did you mean?")
        5. Find best answer from category
        6. Refine with templates + sentiment
        7. Store feeling + save to SQLite
        Returns dict: {"reply": str, "intent": str, "confidence": float}
                   or {"suggestions": [...], "needs_learning": True}
                   or None
        """
        original = user_question.strip()
        cleaned = re.sub(r"[^\w\s]", "", original.lower()).strip()

        # Follow-up check
        if is_follow_up(cleaned):
            last_intent = self.db.get_last_intent(session_id) if session_id else None
            if last_intent and last_intent in self.category_store_map:
                answer = self._find_best_answer(last_intent, cleaned, session_id)
                if answer:
                    sentiment = self.sentiment.detect(cleaned)
                    answer = self._refine_answer(answer, cleaned, sentiment, last_intent, session_id)
                    self._store_feeling(last_intent, original, sentiment, answer)
                    if session_id:
                        self.db.add_turn(session_id, original, answer, last_intent, 1.0, sentiment)
                    logging.info(f"Q: {original} | FOLLOW-UP | Intent: {last_intent}")
                    return {"reply": answer, "intent": last_intent, "confidence": 1.0}

        # Spell correction
        corrected = self.spell.correct(cleaned)

        # Step 1: Detect category (semantic + ML)
        category, confidence, method = self._detect_category(corrected)

        # Feature 2: "Did you mean?" — low confidence
        if category is None:
            suggestions = self.get_suggestions(corrected)
            if suggestions and suggestions[0]["score"] >= 0.20:
                logging.info(f"SUGGESTIONS | Q: {original} | Top: {suggestions[0]}")
                if session_id:
                    self.db.add_turn(session_id, original, None, None, confidence, "neutral")
                return {
                    "reply": None,
                    "suggestions": suggestions,
                    "needs_learning": True,
                    "question": original
                }
            # Truly unknown
            logging.info(f"UNANSWERED | Q: {original} | Confidence: {confidence:.3f}")
            if session_id:
                self.db.add_turn(session_id, original, None, None, confidence, "neutral")
            return None

        # Step 2: Find best answer
        sentiment = self.sentiment.detect(cleaned)
        answer = self._find_best_answer(category, corrected, session_id)

        if answer is None:
            logging.info(f"NO_ANSWER | Q: {original} | Category: {category}")
            if session_id:
                self.db.add_turn(session_id, original, None, category, confidence, sentiment)
            return None

        # Step 3: Refine with templates
        answer = self._refine_answer(answer, corrected, sentiment, category, session_id)

        # Step 4: Store feeling
        self._store_feeling(category, original, sentiment, answer)

        # Step 5: Save to SQLite
        if session_id:
            self.db.add_turn(session_id, original, answer, category, confidence, sentiment)

        cat_type = self._get_store_for(category).get_type(category)
        logging.info(
            f"Q: {original} | Intent: {category} | Type: {cat_type} | "
            f"Method: {method} ({confidence:.3f}) | Sentiment: {sentiment}"
        )
        return {"reply": answer, "intent": category, "confidence": round(confidence, 3)}

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
            intent = result.get("intent", "")
            conf_bar = "●" * int(confidence * 10) + "○" * (10 - int(confidence * 10))
            print(f"\nBot: {result['reply']}")
            print(f"     [{intent}] [{conf_bar}] {confidence:.0%}")


if __name__ == "__main__":
    run_chatbot()
