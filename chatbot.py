"""
Mini Chatbot - NLP + ML + Self Learning
Intent Classification + TF-IDF + Spell Correction + Auto Learn + Conversation Memory
Category-wise dataset with answer refinement and feeling tracking.
No third-party AI API needed.
"""

import json
import logging
import os
import random
import re
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches, SequenceMatcher

# Logging setup
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


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


class ConversationMemory:
    """Session-based conversation history track kore"""

    FOLLOW_UP_WORDS = {
        "tell me more", "more", "aro bolo", "ar bolo", "elaborate",
        "explain more", "details", "aro details", "bistorito",
        "what else", "ar ki", "continue", "go on", "then",
        "ar kono", "aro kichu", "bolo aro", "keep going"
    }

    def __init__(self, max_turns=10, max_sessions=200):
        self.sessions = {}
        self.max_turns = max_turns
        self.max_sessions = max_sessions

    def create_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        self._cleanup_old_sessions()
        return session_id

    def add_turn(self, session_id, user_msg, bot_reply, intent):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({
            "user": user_msg,
            "bot": bot_reply,
            "intent": intent
        })
        if len(self.sessions[session_id]) > self.max_turns:
            self.sessions[session_id] = self.sessions[session_id][-self.max_turns:]

    def get_last_intent(self, session_id):
        if session_id and session_id in self.sessions and self.sessions[session_id]:
            return self.sessions[session_id][-1]["intent"]
        return None

    def get_history(self, session_id):
        return self.sessions.get(session_id, [])

    def is_follow_up(self, text):
        cleaned = text.lower().strip()
        for phrase in self.FOLLOW_UP_WORDS:
            if phrase in cleaned:
                return True
        return False

    def _cleanup_old_sessions(self):
        if len(self.sessions) > self.max_sessions:
            oldest_keys = list(self.sessions.keys())[:-self.max_sessions]
            for key in oldest_keys:
                del self.sessions[key]


class FeedbackStore:
    """Like/Dislike feedback track kore + feedback log maintain kore"""

    def __init__(self, log_path="feedback.json"):
        self.log_path = log_path
        self.feedbacks = self._load()

    def _load(self):
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.feedbacks, f, ensure_ascii=False, indent=2)

    def add(self, question, bot_answer, intent, feedback_type, correct_answer=None):
        entry = {
            "question": question,
            "bot_answer": bot_answer,
            "intent": intent,
            "feedback": feedback_type,
            "correct_answer": correct_answer
        }
        self.feedbacks.append(entry)
        self._save()
        logging.info(
            f"FEEDBACK | {feedback_type.upper()} | Q: {question} | Intent: {intent}"
            + (f" | Correct: {correct_answer}" if correct_answer else "")
        )
        return entry

    def get_stats(self):
        likes = sum(1 for f in self.feedbacks if f["feedback"] == "like")
        dislikes = sum(1 for f in self.feedbacks if f["feedback"] == "dislike")
        return {"total": len(self.feedbacks), "likes": likes, "dislikes": dislikes}


class CategoryStore:
    """Category-wise dataset folder manage kore.
    Each category = one JSON file in category_wise_dataset/ folder.
    """

    def __init__(self, folder="category_wise_dataset"):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.categories = {}  # category_name -> file data dict
        self._load_all()

    def _cat_path(self, category):
        return os.path.join(self.folder, f"{category}.json")

    def _load_all(self):
        """Folder er sob JSON file load kore"""
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
        """Single category er data return kore"""
        return self.categories.get(category)

    def get_all_categories(self):
        return list(self.categories.keys())

    def get_questions_and_labels(self):
        """All questions + labels for training ML model"""
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
        """Single category file save kore"""
        data = self.categories.get(category)
        if data:
            fpath = self._cat_path(category)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logging.info(f"Category saved: {category}")

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
        """Feeling/sentiment entry save kore category file e"""
        if category not in self.categories:
            return
        feelings = self.categories[category].get("feelings", [])
        feelings.append({
            "question": question,
            "sentiment": sentiment,
            "answer": answer_given
        })
        # Last 50 feelings rakho per category
        self.categories[category]["feelings"] = feelings[-50:]
        self.save_category(category)

    def create_category(self, category, cat_type="general", questions=None, answers=None):
        """Notun category create kore"""
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


class ChatBot:
    def __init__(self, general_folder="category_wise_dataset", specialized_folders=None):
        """Multiple dataset folder support.
        general_folder: basic/general categories (default)
        specialized_folders: list of specialized dataset folders (checked first)
            e.g. ["coding_dataset"] — coding questions go here first
        """
        # General store (fallback)
        self.general_store = CategoryStore(general_folder)

        # Specialized stores (checked first, in order)
        self.specialized_stores = []
        if specialized_folders is None:
            specialized_folders = self._auto_detect_specialized_folders(general_folder)
        for folder in specialized_folders:
            if os.path.isdir(folder):
                store = CategoryStore(folder)
                self.specialized_stores.append(store)
                logging.info(f"Specialized store loaded: {folder} ({len(store.get_all_categories())} categories)")

        # All stores combined for convenience
        self.all_stores = self.specialized_stores + [self.general_store]

        # Category -> store mapping for quick lookup
        self.category_store_map = {}
        self._build_category_map()

        self.questions = []
        self.labels = []
        self.learn_count = 0
        self.memory = ConversationMemory()
        self.feedback = FeedbackStore()

        self._prepare_data()
        self._build_spell_corrector()
        self._train_intent_classifier()
        self._train_tfidf()
        self.sentiment = SentimentDetector()

    @staticmethod
    def _auto_detect_specialized_folders(general_folder):
        """Auto-detect specialized dataset folders (any folder ending with _dataset except general)"""
        folders = []
        for item in os.listdir("."):
            if os.path.isdir(item) and item.endswith("_dataset") and item != general_folder:
                folders.append(item)
        folders.sort()
        return folders

    def _build_category_map(self):
        """Category name -> store mapping. Specialized stores get priority."""
        self.category_store_map = {}
        # General store first (will be overwritten by specialized if overlap)
        for cat in self.general_store.get_all_categories():
            self.category_store_map[cat] = self.general_store
        # Specialized stores override (priority)
        for store in self.specialized_stores:
            for cat in store.get_all_categories():
                self.category_store_map[cat] = store

    def _get_store_for(self, category):
        """Category er jonno correct store return kore"""
        return self.category_store_map.get(category, self.general_store)

    @property
    def store(self):
        """Backward compatibility — returns general store"""
        return self.general_store

    def _prepare_data(self):
        """All stores theke combined questions + labels load kore"""
        self.questions = []
        self.labels = []
        for store in self.all_stores:
            qs, ls = store.get_questions_and_labels()
            self.questions.extend(qs)
            self.labels.extend(ls)
        total_cats = len(self.category_store_map)
        logging.info(
            f"Total questions: {len(self.questions)}, "
            f"Categories: {total_cats} (General: {len(self.general_store.get_all_categories())}, "
            f"Specialized: {sum(len(s.get_all_categories()) for s in self.specialized_stores)})"
        )

    def _build_spell_corrector(self):
        all_words = set()
        for q in self.questions:
            all_words.update(q.split())
        self.spell = SpellCorrector(all_words)

    def _train_intent_classifier(self):
        unique_labels = len(set(self.labels))
        if unique_labels < len(self.questions) * 0.6 and len(self.questions) > 0:
            self.intent_clf = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=1000, C=10))
            ])
            self.intent_clf.fit(self.questions, self.labels)
            self.use_ml = True
            logging.info(f"Intent classifier trained: {unique_labels} categories")
        else:
            self.use_ml = False
            logging.info(f"Skipping ML classifier: {unique_labels} categories / {len(self.questions)} questions")

    def _train_tfidf(self):
        if self.questions:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None

    def retrain(self):
        """Full model retrain — all stores reload"""
        for store in self.all_stores:
            store._load_all()
        self._build_category_map()
        self._prepare_data()
        self._build_spell_corrector()
        self._train_intent_classifier()
        self._train_tfidf()
        logging.info("Model retrained with new data")

    # ========== Category Detection ==========

    def _detect_category(self, cleaned_question):
        """Step 1: Fast category detection.
        ML classifier + TF-IDF combine kore best category khouje ber kore.
        Uses voting: if both ML and TF-IDF agree, confidence boosts.
        Returns: (category, confidence, method) or (None, 0, None)
        """
        # ML classifier
        ml_intent = None
        ml_conf = 0
        ml_top3 = []
        if self.use_ml:
            intent_proba = self.intent_clf.predict_proba([cleaned_question])[0]
            classes = self.intent_clf.classes_
            top_indices = intent_proba.argsort()[-3:][::-1]
            ml_top3 = [(classes[i], intent_proba[i]) for i in top_indices]
            ml_intent = ml_top3[0][0]
            ml_conf = ml_top3[0][1]

        # TF-IDF similarity
        tfidf_score, tfidf_intent = self._tfidf_match(cleaned_question)

        # High confidence ML — trust it
        if self.use_ml and ml_conf >= 0.45:
            return ml_intent, ml_conf, "ml"

        # Both agree — strong signal even at lower confidence
        if self.use_ml and ml_intent == tfidf_intent and ml_conf >= 0.15:
            combined = ml_conf * 0.6 + tfidf_score * 0.4
            return ml_intent, combined, "ml+tfidf"

        # TF-IDF is in ML top-3 — prefer TF-IDF's pick
        if self.use_ml and tfidf_score >= 0.30:
            ml_top3_cats = [c for c, s in ml_top3]
            if tfidf_intent in ml_top3_cats:
                return tfidf_intent, tfidf_score, "tfidf+ml_top3"

        # ML moderate confidence
        if self.use_ml and ml_conf >= 0.25:
            return ml_intent, ml_conf, "ml_moderate"

        # TF-IDF fallback (stricter threshold)
        if tfidf_score >= 0.35:
            return tfidf_intent, tfidf_score, "tfidf"

        return None, max(ml_conf, tfidf_score), None

    def _tfidf_match(self, question):
        """TF-IDF based similarity match"""
        if self.vectorizer is None:
            return 0.0, None
        user_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        top_indices = similarities.argsort()[-5:][::-1]
        for idx in top_indices:
            score = similarities[idx]
            label = self.labels[idx]
            if score < 0.15:
                break
            if not label.startswith("conv_"):
                return score, label

        best_idx = similarities.argmax()
        return similarities[best_idx], self.labels[best_idx]

    # ========== Answer Finding & Refinement ==========

    def _find_best_answer(self, category, question, session_id=None):
        """Step 2: Category file theke best answer khouje ber kore.
        Answers compare kore question er sathe, best match pick kore.
        """
        store = self._get_store_for(category)
        answers = store.get_answers(category)
        if not answers:
            return None

        if len(answers) == 1:
            return answers[0]

        # Score each answer by relevance to question
        scored = []
        for ans in answers:
            score = self._answer_relevance(question, ans)
            scored.append((score, ans))

        # Sort by relevance score (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Avoid repeating recent answers in session
        if session_id:
            history = self.memory.get_history(session_id)
            recent_replies = {turn["bot"] for turn in history[-3:]}
            for score, ans in scored:
                if ans not in recent_replies:
                    return ans

        # Return best scored answer
        return scored[0][1]

    def _answer_relevance(self, question, answer):
        """Question er sathe answer kotatuku relevant seta score kore.
        Keywords overlap + sequence similarity use kore.
        """
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())

        # Remove common stop words
        stop_words = {"is", "a", "the", "to", "of", "and", "in", "for", "what", "how", "i", "me", "my", "it", "do", "can", "this", "that"}
        q_words -= stop_words
        a_words -= stop_words

        if not q_words:
            return 0.0

        # Keyword overlap score
        overlap = len(q_words & a_words) / len(q_words) if q_words else 0

        # Sequence similarity
        seq_score = SequenceMatcher(None, question.lower(), answer.lower()[:100]).ratio()

        # Combined score (keyword overlap matters more)
        return overlap * 0.6 + seq_score * 0.4

    def _refine_answer(self, answer, question, sentiment):
        """Step 3: Answer ke question er context e refine kore.
        Sentiment adjust + question-specific modification.
        """
        # Sentiment based adjustment
        answer = self.sentiment.adjust_response(answer, sentiment)

        return answer

    # ========== Feeling Storage ==========

    def _store_feeling(self, category, question, sentiment, answer):
        """Step 4: Question er feeling/sentiment category file e save kore"""
        if sentiment != "neutral":
            store = self._get_store_for(category)
            store.add_feeling(category, question, sentiment, answer)
            logging.info(f"FEELING | Category: {category} | Sentiment: {sentiment} | Q: {question}")

    # ========== Main Flow ==========

    def _pick_answer(self, category, session_id=None):
        """Simple random answer pick (fallback)"""
        store = self._get_store_for(category)
        answers = store.get_answers(category)
        if not answers:
            return None
        if len(answers) == 1:
            return answers[0]
        if session_id:
            history = self.memory.get_history(session_id)
            recent_replies = {turn["bot"] for turn in history[-3:]}
            fresh = [a for a in answers if a not in recent_replies]
            if fresh:
                return random.choice(fresh)
        return random.choice(answers)

    def learn(self, question, category, answer):
        """Notun question-answer shikhbe. Correct store er category file e save korbe."""
        question = question.lower().strip()
        category = category.lower().strip()

        # Check if category already exists in any store
        store = self._get_store_for(category)
        if store.category_exists(category):
            store.add_question(category, question)
            if answer:
                store.add_answer(category, answer)
        else:
            # New category goes to general store by default
            self.general_store.create_category(category, "general", [question], [answer])
            self.category_store_map[category] = self.general_store

        self.retrain()
        self.learn_count += 1
        logging.info(f"LEARNED | Q: {question} | Category: {category} | Total learns: {self.learn_count}")
        return True

    def handle_feedback(self, question, bot_answer, intent, feedback_type, correct_answer=None, correct_category=None):
        """User feedback handle kore. Dislike hole correct answer diye re-learn kore."""
        self.feedback.add(question, bot_answer, intent, feedback_type, correct_answer)

        if feedback_type == "dislike" and correct_answer:
            category = correct_category or intent or "general"
            self.learn(question, category, correct_answer)
            logging.info(f"CORRECTED | Q: {question} | Old intent: {intent} | New category: {category}")
            return {"status": "corrected", "message": "Dhonnobad! Bhul theke shikhlam!"}

        if feedback_type == "like":
            return {"status": "liked", "message": "Thanks for the feedback!"}

        return {"status": "noted", "message": "Feedback noted."}

    def get_answer(self, user_question, session_id=None):
        """Main answer pipeline:
        1. Category detect koro (fast ML + TF-IDF)
        2. Category file theke best answer khouje bero
        3. Answer refine koro (question er sathe compare + sentiment)
        4. Feeling store koro category file e
        Returns dict: {"reply": str, "intent": str} or None
        """
        original = user_question.strip()
        cleaned = re.sub(r"[^\w\s]", "", original.lower()).strip()

        # Follow-up check
        if self.memory.is_follow_up(cleaned):
            last_intent = self.memory.get_last_intent(session_id)
            if last_intent and last_intent in self.category_store_map:
                answer = self._find_best_answer(last_intent, cleaned, session_id)
                if answer:
                    sentiment = self.sentiment.detect(cleaned)
                    answer = self._refine_answer(answer, cleaned, sentiment)
                    self._store_feeling(last_intent, original, sentiment, answer)
                    self.memory.add_turn(session_id, original, answer, last_intent)
                    logging.info(f"Q: {original} | FOLLOW-UP | Intent: {last_intent} | Session: {session_id}")
                    return {"reply": answer, "intent": last_intent}

        # Spell correction
        corrected = self.spell.correct(cleaned)

        # Step 1: Detect category
        category, confidence, method = self._detect_category(corrected)

        if category is None:
            logging.info(f"UNANSWERED | Q: {original} | Confidence: {confidence:.2f}")
            if session_id:
                self.memory.add_turn(session_id, original, None, None)
            return None

        # Step 2: Find best answer from category file
        sentiment = self.sentiment.detect(cleaned)
        answer = self._find_best_answer(category, corrected, session_id)

        if answer is None:
            logging.info(f"NO_ANSWER | Q: {original} | Category: {category} (empty answers)")
            if session_id:
                self.memory.add_turn(session_id, original, None, None)
            return None

        # Step 3: Refine answer
        answer = self._refine_answer(answer, corrected, sentiment)

        # Step 4: Store feeling
        self._store_feeling(category, original, sentiment, answer)

        # Save to conversation memory
        if session_id:
            self.memory.add_turn(session_id, original, answer, category)

        cat_type = self._get_store_for(category).get_type(category)
        logging.info(
            f"Q: {original} | Corrected: {corrected} | Intent: {category} | "
            f"Type: {cat_type} | Method: {method} ({confidence:.2f}) | Sentiment: {sentiment} | Session: {session_id}"
        )
        return {"reply": answer, "intent": category}


def run_chatbot():
    print("=" * 50)
    print("  Mini Chatbot (NLP + ML + Self Learning)")
    print("  'exit' likho session sesh korte")
    print("=" * 50)

    bot = ChatBot()
    session_id = bot.memory.create_session()
    total_cats = len(bot.category_store_map)
    general_count = len(bot.general_store.get_all_categories())
    special_count = sum(len(s.get_all_categories()) for s in bot.specialized_stores)
    print(f"  {len(bot.questions)} questions | {total_cats} categories")
    print(f"  General: {general_count} | Specialized: {special_count}")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            if bot.learn_count > 0:
                print(f"\n[Info] Aj {bot.learn_count} ta notun jinish shikhechi!")
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
            print(f"\nBot: Dhonnobad! Ami shikhe nilam! Ekhon theke '{user_input}' jiggesh korle ami answer dite parbo.")
        else:
            print(f"\nBot: {result['reply']}")


if __name__ == "__main__":
    run_chatbot()
