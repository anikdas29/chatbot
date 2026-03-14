"""
Mini Chatbot - NLP + ML + Self Learning
Intent Classification + TF-IDF + Spell Correction + Auto Learn + Conversation Memory
No third-party AI API needed.
"""

import json
import logging
import random
import re
import uuid
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

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
        "ghatia", "scam", "fraud", "thug"
    }
    HAPPY_WORDS = {
        "great", "awesome", "valo", "bhalo", "excellent", "best",
        "good", "love", "happy", "amazing", "wonderful", "darun",
        "oshadharon", "khushi"
    }

    @staticmethod
    def detect(text):
        words = set(text.lower().split())
        angry = len(words & SentimentDetector.ANGRY_WORDS)
        happy = len(words & SentimentDetector.HAPPY_WORDS)
        if angry > happy:
            return "angry"
        if happy > angry:
            return "happy"
        return "neutral"

    @staticmethod
    def adjust_response(answer, sentiment):
        if sentiment == "angry":
            return "Ami bujhte parchi apni frustrated. " + answer + "\n\nApnar somossa solve korte amra committed. Ar kichu lagleo bolun."
        if sentiment == "happy":
            return answer + "\n\nApnar bhalo lagche jene amrao khushi!"
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
        # Keep only last max_turns
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


class ChatBot:
    def __init__(self, dataset_path="dataset.json"):
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset(dataset_path)
        self.questions = []
        self.labels = []
        self.category_answers = {}
        self.learn_count = 0
        self.memory = ConversationMemory()
        self.feedback = FeedbackStore()

        self._prepare_data()
        self._build_spell_corrector()
        self._train_intent_classifier()
        self._train_tfidf()
        self.sentiment = SentimentDetector()

    def _load_dataset(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"Dataset loaded: {len(data)} categories")
        return data

    def _save_dataset(self):
        """Dataset ke JSON file e save kore"""
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=4)
        logging.info("Dataset saved to file")

    def _prepare_data(self):
        """Dataset theke question-label-answer prepare kore"""
        self.questions = []
        self.labels = []
        self.category_answers = {}

        for item in self.dataset:
            category = item["category"]
            # Support both "answers" (list) and legacy "answer" (string)
            if "answers" in item:
                self.category_answers[category] = item["answers"]
            else:
                self.category_answers[category] = [item["answer"]]
            for q in item["questions"]:
                self.questions.append(q.lower())
                self.labels.append(category)

        logging.info(f"Total questions: {len(self.questions)}, Categories: {len(self.category_answers)}")

    def _build_spell_corrector(self):
        all_words = set()
        for q in self.questions:
            all_words.update(q.split())
        self.spell = SpellCorrector(all_words)

    def _train_intent_classifier(self):
        # Only train ML classifier if categories < 60% of questions
        # Otherwise too many categories for ML to work well
        unique_labels = len(set(self.labels))
        if unique_labels < len(self.questions) * 0.6:
            self.intent_clf = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=1000, C=10))
            ])
            self.intent_clf.fit(self.questions, self.labels)
            self.use_ml = True
            logging.info(f"Intent classifier trained: {unique_labels} categories")
        else:
            self.use_ml = False
            logging.info(f"Skipping ML classifier: too many categories ({unique_labels}/{len(self.questions)})")

    def _train_tfidf(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def retrain(self):
        """Full model retrain - notun data shikhle call hoy"""
        self._prepare_data()
        self._build_spell_corrector()
        self._train_intent_classifier()
        self._train_tfidf()
        logging.info("Model retrained with new data")

    def _tfidf_fallback(self, question):
        user_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        # Get top 5 matches
        top_indices = similarities.argsort()[-5:][::-1]

        # Prefer non-conv_ categories (intent-based) over conv_ (casual conversation)
        for idx in top_indices:
            score = similarities[idx]
            label = self.labels[idx]
            if score < 0.15:
                break
            if not label.startswith("conv_"):
                return score, label

        # Fallback to best match
        best_idx = similarities.argmax()
        return similarities[best_idx], self.labels[best_idx]

    def _pick_answer(self, category, session_id=None):
        """Category er answers list theke random answer pick kore.
        Same session e same answer repeat avoid kore."""
        answers = self.category_answers[category]
        if len(answers) == 1:
            return answers[0]

        # Try to avoid repeating the last answer in this session
        if session_id:
            history = self.memory.get_history(session_id)
            recent_replies = {turn["bot"] for turn in history[-3:]}
            fresh = [a for a in answers if a not in recent_replies]
            if fresh:
                return random.choice(fresh)

        return random.choice(answers)

    def learn(self, question, category, answer):
        """Notun question-answer shikhbe ar dataset e save korbe"""
        question = question.lower().strip()
        category = category.lower().strip()

        # Check: ei category already ache ki na
        found = False
        for item in self.dataset:
            if item["category"] == category:
                # Existing category te question add koro
                if question not in item["questions"]:
                    item["questions"].append(question)
                # Answer add koro answers list e (duplicate avoid)
                if answer:
                    if "answers" in item:
                        if answer not in item["answers"]:
                            item["answers"].append(answer)
                    else:
                        item["answers"] = [answer]
                found = True
                break

        if not found:
            # Notun category create koro
            self.dataset.append({
                "category": category,
                "questions": [question],
                "answers": [answer]
            })

        # Save and retrain
        self._save_dataset()
        self.retrain()
        self.learn_count += 1
        logging.info(f"LEARNED | Q: {question} | Category: {category} | Total learns: {self.learn_count}")
        return True

    def handle_feedback(self, question, bot_answer, intent, feedback_type, correct_answer=None, correct_category=None):
        """User feedback handle kore. Dislike hole correct answer diye re-learn kore."""
        self.feedback.add(question, bot_answer, intent, feedback_type, correct_answer)

        if feedback_type == "dislike" and correct_answer:
            # User er correct answer diye dataset update koro
            category = correct_category or intent or "general"
            self.learn(question, category, correct_answer)
            logging.info(f"CORRECTED | Q: {question} | Old intent: {intent} | New category: {category}")
            return {"status": "corrected", "message": "Dhonnobad! Bhul theke shikhlam!"}

        if feedback_type == "like":
            return {"status": "liked", "message": "Thanks for the feedback!"}

        return {"status": "noted", "message": "Feedback noted."}

    def get_answer(self, user_question, session_id=None):
        """ML + NLP combine kore best answer return kore. Session memory support.
        Returns dict: {"reply": str, "intent": str} or None
        """
        original = user_question.strip()
        cleaned = re.sub(r"[^\w\s]", "", original.lower()).strip()

        # Check: follow-up question ki na? ("tell me more", "aro bolo" etc.)
        if self.memory.is_follow_up(cleaned):
            last_intent = self.memory.get_last_intent(session_id)
            if last_intent and last_intent in self.category_answers:
                answer = self._pick_answer(last_intent, session_id)
                sentiment = self.sentiment.detect(cleaned)
                answer = self.sentiment.adjust_response(answer, sentiment)
                self.memory.add_turn(session_id, original, answer, last_intent)
                logging.info(f"Q: {original} | FOLLOW-UP | Intent: {last_intent} | Session: {session_id}")
                return {"reply": answer, "intent": last_intent}

        # Spell correction
        corrected = self.spell.correct(cleaned)

        # Sentiment detect
        sentiment = self.sentiment.detect(cleaned)

        # TF-IDF similarity (primary)
        tfidf_score, tfidf_intent = self._tfidf_fallback(corrected)

        # Intent classification (ML) if available
        max_confidence = 0
        predicted_intent = None
        if self.use_ml:
            intent_proba = self.intent_clf.predict_proba([corrected])
            max_confidence = intent_proba.max()
            predicted_intent = self.intent_clf.predict([corrected])[0]

        # Decision: ML first (if confident), then TF-IDF
        if self.use_ml and max_confidence >= 0.55:
            predicted_intent = predicted_intent
            method = f"intent_clf ({max_confidence:.2f})"
        elif tfidf_score >= 0.20:
            predicted_intent = tfidf_intent
            method = f"tfidf ({tfidf_score:.2f})"
        else:
            logging.info(f"UNANSWERED | Q: {original} | ML: {max_confidence:.2f} | TF-IDF: {tfidf_score:.2f}")
            # Still save unanswered to memory so context is tracked
            if session_id:
                self.memory.add_turn(session_id, original, None, None)
            return None

        answer = self._pick_answer(predicted_intent, session_id)
        answer = self.sentiment.adjust_response(answer, sentiment)

        # Save to conversation memory
        if session_id:
            self.memory.add_turn(session_id, original, answer, predicted_intent)

        logging.info(
            f"Q: {original} | Corrected: {corrected} | Intent: {predicted_intent} | "
            f"Method: {method} | Sentiment: {sentiment} | Session: {session_id}"
        )
        return {"reply": answer, "intent": predicted_intent}


def run_chatbot():
    print("=" * 50)
    print("  Mini Chatbot (NLP + ML + Self Learning)")
    print("  'exit' likho session sesh korte")
    print("=" * 50)

    bot = ChatBot()
    session_id = bot.memory.create_session()
    print(f"  {len(bot.questions)} questions | {len(bot.category_answers)} categories")
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
            # Answer janina - user ke jiggesh kori
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
