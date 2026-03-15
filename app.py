"""
Mini Chatbot - Flask API + Web UI + Self Learning + Semantic Search
Run with: python app.py
"""

from flask import Flask, request, jsonify, render_template
from chatbot import ChatBot

app = Flask(__name__)
bot = ChatBot()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    stats = bot.db.get_feedback_stats()
    return jsonify({
        "status": "ok",
        "total_questions": len(bot.questions),
        "total_categories": len(bot.category_store_map),
        "feedback": stats
    })


@app.route("/api/session", methods=["POST"])
def create_session():
    session_id = bot.db.create_session()
    return jsonify({"session_id": session_id})


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field."}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    session_id = data.get("session_id")

    # If user selected a suggestion category, answer from that category directly
    chosen_category = data.get("chosen_category")
    if chosen_category:
        store = bot._get_store_for(chosen_category)
        answers = store.get_answers(chosen_category)
        if answers:
            import random
            answer = random.choice(answers)
            sentiment = bot.sentiment.detect(user_message)
            answer = bot._refine_answer(answer, user_message, sentiment, chosen_category, session_id)
            if session_id:
                bot.db.add_turn(session_id, user_message, answer, chosen_category, 1.0, sentiment)
            return jsonify({
                "reply": answer,
                "intent": chosen_category,
                "confidence": 1.0,
                "needs_learning": False
            })

    result = bot.get_answer(user_message, session_id)

    # Truly unknown — no result at all
    if result is None:
        return jsonify({
            "reply": None,
            "needs_learning": True,
            "question": user_message
        })

    # Has suggestions but no direct answer — "Did you mean?"
    if result.get("suggestions") and result.get("reply") is None:
        return jsonify({
            "reply": None,
            "suggestions": result["suggestions"],
            "needs_learning": True,
            "question": result.get("question", user_message)
        })

    # Normal answer with confidence
    return jsonify({
        "reply": result["reply"],
        "intent": result["intent"],
        "confidence": result.get("confidence", 0),
        "needs_learning": False
    })


@app.route("/api/feedback", methods=["POST"])
def feedback_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing data."}), 400

    question = data.get("question", "").strip()
    bot_answer = data.get("bot_answer", "").strip()
    intent = data.get("intent", "").strip()
    feedback_type = data.get("feedback", "").strip()
    correct_answer = data.get("correct_answer", "").strip() or None
    correct_category = data.get("correct_category", "").strip() or None

    if not question or feedback_type not in ("like", "dislike"):
        return jsonify({"error": "question and feedback (like/dislike) required."}), 400

    result = bot.handle_feedback(
        question, bot_answer, intent, feedback_type,
        correct_answer, correct_category
    )
    return jsonify(result)


@app.route("/api/learn", methods=["POST"])
def learn_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing data."}), 400

    question = data.get("question", "").strip()
    category = data.get("category", "").strip()
    answer = data.get("answer", "").strip()

    if not all([question, category, answer]):
        return jsonify({"error": "question, category, answer sobai required."}), 400

    bot.learn(question, category, answer)
    return jsonify({
        "message": f"Shikhe nilam! '{question}' er answer ekhon jani.",
        "total_questions": len(bot.questions),
        "total_categories": len(bot.category_store_map)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
