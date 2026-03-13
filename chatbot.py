"""
AI-Based Customer Support Chatbot
----------------------------------
Answers customer questions accurately and politely.
"""

import os
import anthropic

SYSTEM_PROMPT = """You are an AI-based customer support chatbot.
Your job is to answer customer questions accurately and politely.

Rules:
1. Always understand the customer's question first.
2. Respond in clear, simple language.
3. Provide helpful and relevant information.
4. If you do not know the answer, politely say that you do not have enough information.
5. Keep responses short, friendly, and professional.
"""

def create_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
    return anthropic.Anthropic(api_key=api_key)

def chat(client, conversation_history: list, user_message: str) -> str:
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=conversation_history
    )

    assistant_reply = response.content[0].text

    conversation_history.append({
        "role": "assistant",
        "content": assistant_reply
    })

    return assistant_reply

def run_chatbot():
    print("=" * 50)
    print("  Customer Support Chatbot")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=" * 50)

    client = create_client()
    conversation_history = []

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("\nBot: Thank you for reaching out! Have a great day. 👋")
            break

        try:
            reply = chat(client, conversation_history, user_input)
            print(f"\nBot: {reply}")
        except Exception as e:
            print(f"\nBot: Sorry, I encountered an error: {e}")

if __name__ == "__main__":
    run_chatbot()
