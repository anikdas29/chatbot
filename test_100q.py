"""
100 random out-of-context questions to test chatbot answer quality.
Mix of: general knowledge, math, opinion, Bangla, tricky, multi-topic, nonsense.
Compares bot answer vs what Claude would ideally answer.
"""
import json
import logging
logging.disable(logging.CRITICAL)

from chatbot import ChatBot

bot = ChatBot()

# 100 test questions — intentionally diverse, many outside dataset categories
QUESTIONS = [
    # General knowledge (should answer)
    "what is python?",
    "how does machine learning work?",
    "what is javascript used for?",
    "tell me about cricket",
    "what is photosynthesis?",
    "how to learn programming?",
    "what is global warming?",
    "tell me about solar system",
    "what is artificial intelligence?",
    "how does the internet work?",

    # Math / Logic (bot probably can't)
    "2+2 koto?",
    "what is 10 times 5?",
    "square root of 144?",
    "what is 15% of 200?",
    "solve x+5=10",

    # Bangla / Banglish
    "tumi ki?",
    "tomar nam ki?",
    "ami kemon achi?",
    "bangladesh er capital ki?",
    "bangla sahityo niye bolo",
    "ki korcho?",
    "kemon acho?",
    "amake help koro",
    "dhonnobad",
    "tumi ki manush?",

    # Multi-topic (should trigger multi-category)
    "how to deploy python flask app on aws?",
    "best machine learning algorithm for image classification",
    "how to cook healthy food for weight loss?",
    "learn react and javascript together",
    "every football player likes cricket?",
    "python vs javascript which is better?",
    "how to start a youtube cooking channel?",
    "fitness tips for software developers",
    "how to use git with python projects?",
    "photography tips for travel bloggers",

    # Opinion / Subjective
    "what is the meaning of life?",
    "is AI dangerous?",
    "which programming language is best?",
    "should i learn python or javascript?",
    "is social media bad for mental health?",
    "what makes a good leader?",
    "is remote work better than office?",
    "what is happiness?",
    "do aliens exist?",
    "is college worth it?",

    # Tricky / Edge cases
    "hello",
    "bye",
    "thanks",
    "ok",
    "hmm",
    "lol",
    "???",
    "",
    "a",
    "tell me everything about everything",

    # Out of dataset (should say unknown)
    "what time is it?",
    "what is the weather today?",
    "who is the president of USA?",
    "when was world war 2?",
    "how old is the earth?",
    "what is quantum computing?",
    "explain blockchain simply",
    "who invented telephone?",
    "what is DNA?",
    "how do airplanes fly?",

    # Misspelled / Informal
    "hwo to lern pytohn?",
    "waht is machne lerning?",
    "javscript tutoral",
    "recact vs anglar",
    "progrming for bignners",
    "hw to mke website?",
    "data scince career",
    "artifical inteligence kya hai?",
    "cod revew tips",
    "bes laptop for coding",

    # Specific technical
    "what is REST API?",
    "explain docker containers",
    "what is kubernetes?",
    "how does SQL join work?",
    "what is git rebase?",
    "explain microservices architecture",
    "what is TCP/IP?",
    "how does OAuth work?",
    "what is CI/CD pipeline?",
    "explain design patterns",

    # Emotional / Conversational
    "i am feeling sad",
    "i am so happy today",
    "i am confused about my career",
    "i am angry",
    "i feel lonely",
    "i am stressed about exams",
    "help me feel better",
    "i am bored",
    "motivate me",
    "tell me a joke",
]

# Expected ideal category (what Claude would classify as)
EXPECTED = [
    # General knowledge
    "python", "ml", "javascript", "cricket", "biology/science",
    "programming", "climate/environment", "astronomy", "ai", "networking/internet",
    # Math
    "math(4)", "math(50)", "math(12)", "math(30)", "math(x=5)",
    # Bangla
    "bot_identity", "bot_identity", "greeting/feelings", "bangladesh/geography", "bangla_literature",
    "greeting", "greeting", "help", "gratitude", "bot_identity",
    # Multi-topic
    "flask+aws", "ml+classification", "cooking+diet+weight_loss", "react+javascript", "football+cricket",
    "python+javascript", "youtube+cooking", "fitness+programming", "git+python", "photography+travel",
    # Opinion
    "philosophy", "ai_ethics", "programming", "python+javascript", "social_media+mental_health",
    "leadership", "remote_work", "philosophy", "science/unknown", "education",
    # Tricky
    "greeting", "farewell", "gratitude", "unknown", "unknown",
    "unknown", "unknown", "unknown", "unknown", "unknown",
    # Out of dataset
    "unknown(time)", "unknown(weather)", "unknown(politics)", "unknown(history)", "unknown(science)",
    "quantum_computing", "blockchain", "unknown(history)", "biology", "physics/aviation",
    # Misspelled
    "python", "ml", "javascript", "react+angular", "programming",
    "web_dev", "data_science", "ai", "code_review", "laptop/hardware",
    # Technical
    "api", "docker", "kubernetes", "sql", "git",
    "microservices", "networking", "authentication", "cicd", "design_patterns",
    # Emotional
    "mental_health/sad", "happy/greeting", "career/confused", "anger/support", "loneliness/support",
    "stress/study", "mental_health", "entertainment", "motivation", "joke/humor",
]

results = []
correct = 0
wrong = 0
no_answer = 0
wrong_answers = []

print("=" * 80)
print("100-QUESTION CHATBOT TEST")
print(f"Model: {len(bot.questions)} questions, {len(bot.category_store_map)} categories")
print("=" * 80)

for i, (question, expected) in enumerate(zip(QUESTIONS, EXPECTED)):
    if not question.strip():
        results.append({"q": "(empty)", "expected": expected, "got": "SKIP", "status": "SKIP"})
        continue

    result = bot.get_answer(question)

    if result is None:
        got_cats = "NO_ANSWER"
        got_reply = None
        confidence = 0
    elif result.get("suggestions"):
        got_cats = "SUGGESTIONS"
        got_reply = None
        confidence = 0
    else:
        cats = result.get("categories", [])
        got_cats = ",".join(cats) if cats else str(result.get("intent", "?"))
        got_reply = result.get("reply", "")[:100]
        confidence = result.get("confidence", 0)

    # Simple check: does expected category appear in got_cats?
    expected_lower = expected.lower()
    got_lower = got_cats.lower()

    # Check if any expected keyword is in the result
    expected_parts = expected_lower.replace("+", ",").replace("/", ",").split(",")
    expected_parts = [p.strip().split("(")[0].strip() for p in expected_parts]  # remove (value)

    is_correct = False
    if "unknown" in expected_lower:
        # Should be NO_ANSWER or SUGGESTIONS
        if got_cats in ("NO_ANSWER", "SUGGESTIONS"):
            is_correct = True
    else:
        for part in expected_parts:
            if part and part in got_lower:
                is_correct = True
                break

    status = "OK" if is_correct else "WRONG"
    if got_cats in ("NO_ANSWER", "SUGGESTIONS") and "unknown" not in expected_lower:
        status = "MISS"  # should have answered but didn't

    if status == "OK":
        correct += 1
    elif status == "WRONG":
        wrong += 1
        wrong_answers.append({
            "q": question,
            "expected": expected,
            "got": got_cats,
            "confidence": confidence,
            "reply": got_reply
        })
    else:
        no_answer += 1

    results.append({
        "q": question, "expected": expected, "got": got_cats,
        "confidence": confidence, "status": status
    })

    marker = "Y" if status == "OK" else ("X" if status == "WRONG" else "-")
    print(f"  {i+1:3}. {marker} Q: {question[:45]:45} | Expected: {expected:25} | Got: {got_cats:25} | {confidence:.0%}")

print("\n" + "=" * 80)
print(f"RESULTS: {correct}/100 correct | {wrong} wrong | {no_answer} missed")
print(f"Accuracy: {correct}%")
print("=" * 80)

if wrong_answers:
    print(f"\n--- TOP WRONG ANSWERS ({len(wrong_answers)}) ---")
    for w in wrong_answers[:20]:
        print(f"\n  Q: {w['q']}")
        print(f"  Expected: {w['expected']}")
        print(f"  Got: {w['got']} ({w['confidence']:.0%})")
        if w['reply']:
            print(f"  Reply: {w['reply']}")

# Save full results
with open("test_100q_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "total": 100,
        "correct": correct,
        "wrong": wrong,
        "missed": no_answer,
        "accuracy": correct,
        "details": results,
        "wrong_details": wrong_answers
    }, f, ensure_ascii=False, indent=2)

print(f"\nFull results saved to test_100q_results.json")
