"""
REST API for the Customer Support Chatbot
Run with: python app.py
"""

from flask import Flask, request, jsonify, session
from chatbot import chat, create_client
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

client = create_client()
sessions = {}  # In-memory store: session_id -> conversation history

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Customer Support Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Send a message",
            "POST /reset": "Reset conversation",
            "GET /health": "Health check"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field."}), 400

    session_id = data.get("session_id", "default")
    user_message = data["message"].strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    if session_id not in sessions:
        sessions[session_id] = []

    try:
        reply = chat(client, sessions[session_id], user_message)
        return jsonify({
            "session_id": session_id,
            "reply": reply,
            "turn": len(sessions[session_id]) // 2
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    sessions.pop(session_id, None)
    return jsonify({"message": f"Session '{session_id}' reset successfully."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
