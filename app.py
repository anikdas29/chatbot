"""
Mini Chatbot - Flask API + Web UI + Self Learning + Semantic Search
Run with: python app.py
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chatbot import ChatBot

app = Flask(__name__)
CORS(app)
bot = ChatBot()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    stats = bot.db.get_feedback_stats()

    # Determine active tier
    has_llm = bot.generator.available
    has_encoder = bot.encoder.available
    if has_llm and has_encoder:
        tier = f"Tier {'1' if bot.generator.is_phi3 else '2'}"
    elif has_encoder:
        tier = "Tier 3 (no LLM)"
    else:
        tier = "Tier 4 (keyword only)"

    return jsonify({
        "status": "ok",
        "tier": tier,
        "llm_model": bot.generator.model_name if has_llm else None,
        "encoder": bot.encoder.model_name if has_encoder else None,
        "total_questions": len(bot.questions),
        "total_categories": len(bot.category_store_map),
        "feedback": stats,
        "pending_learns": bot.db.get_pending_count()
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
                "categories": [chosen_category],
                "needs_learning": False
            })

    # API default: FAISS + ML only (fast, ~0.5s) — send "fast": false for LLM
    fast_mode = data.get("fast", True)
    result = bot.get_answer(user_message, session_id, skip_llm=fast_mode)

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

    # Normal answer with confidence (supports multi-category)
    return jsonify({
        "reply": result["reply"],
        "intent": result["intent"],
        "confidence": result.get("confidence", 0),
        "categories": result.get("categories", []),
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


@app.route("/api/process_learns", methods=["POST"])
def process_learns_endpoint():
    """Manually trigger batch processing of queued learns.
    Useful for admin/cron to flush the pending queue on demand.
    """
    result = bot.process_pending_learns()
    return jsonify({
        "message": f"Processed {result['processed']} pending learns.",
        "processed": result["processed"],
        "total_questions": len(bot.questions),
        "total_categories": len(bot.category_store_map)
    })


@app.route("/api/pending_learns", methods=["GET"])
def pending_learns_endpoint():
    """Check how many learns are waiting in the queue."""
    count = bot.db.get_pending_count()
    return jsonify({
        "pending_count": count,
        "batch_threshold": bot.LEARN_BATCH_THRESHOLD
    })


@app.route("/api/feedback_report", methods=["GET"])
def feedback_report():
    """Review all disliked answers for manual improvement.
    Shows questions where bot gave wrong answers — useful for dataset improvement.
    """
    rows = bot.db.conn.execute(
        "SELECT question, bot_answer, intent, correct_answer, correct_category, created_at "
        "FROM feedback WHERE feedback_type='dislike' ORDER BY created_at DESC LIMIT 100"
    ).fetchall()
    dislikes = []
    for r in rows:
        dislikes.append({
            "question": r["question"],
            "bot_answer": r["bot_answer"],
            "intent": r["intent"],
            "correct_answer": r["correct_answer"],
            "correct_category": r["correct_category"],
            "timestamp": r["created_at"]
        })
    stats = bot.db.get_feedback_stats()
    return jsonify({
        "total_feedback": stats["total"],
        "likes": stats["likes"],
        "dislikes": stats["dislikes"],
        "accuracy_rate": round(stats["likes"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0,
        "recent_dislikes": dislikes
    })


@app.route("/api/weak_categories", methods=["GET"])
def weak_categories():
    """Find categories with high dislike rates that need better answers."""
    min_dislikes = request.args.get("min", 3, type=int)
    weak = bot.get_weak_categories(min_dislikes)
    return jsonify({
        "weak_categories": weak,
        "total": len(weak)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
