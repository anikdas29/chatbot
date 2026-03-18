"""
100 random out-of-context questions to test chatbot answer quality.
Dataset-driven evaluation (no EXPECTED).
"""

import json
import logging
logging.disable(logging.CRITICAL)

from chatbot import ChatBot

bot = ChatBot()

# =========================
# QUESTIONS (100)
# =========================
QUESTIONS = [

# --- SET 1: Real-world simple ---
"what is docker in simple terms?",
"what is api?",
"what is database?",
"what is cloud computing?",
"what is machine learning?",
"5+7 koto?",
"india er capital ki?",
"tumi ki korte paro?",
"hey",
"thanks a lot",

# --- SET 2: Practical usage ---
"why do we use api?",
"database kothay use hoy?",
"cloud computing keno popular?",
"machine learning diye ki kora jai?",
"wifi kivabe kaj kore?",
"50-20 koto?",
"dhaka kon deshe?",
"tumi ki offline kaj korte paro?",
"how is your day?",
"dhonnobad bhai",

# --- SET 3: Comparison ---
"mysql vs mongodb difference",
"frontend vs backend",
"http vs https",
"ai vs deep learning",
"ram vs rom",
"30% of 150 koto?",
"bangladesh er main export ki?",
"tumi ki chatbot?",
"what are your limitations?",
"good evening",

# --- SET 4: Multi-task ---
"what is api and give example",
"frontend backend difference and which is easier?",
"cloud computing and local server difference",
"machine learning diye earning kora jai?",
"internet slow hole ki ki check korbo?",
"40+60 and 25% of 200",
"bangladesh er capital and currency",
"tumi ki real and ke tomake banai?",
"tell me a fun fact and joke",
"thanks now explain database",

# --- SET 5: Bangla dominant ---
"ami web development shikte chai kotha theke start korbo?",
"backend diye ki kora jai?",
"ai ki manusher job niye nibe?",
"internet off hole ki problem hoy?",
"gach keno oxygen dei?",
"80+20 koto?",
"bangladesh er prime minister ke?",
"tumi ki amar friend hote paro?",
"ami bored lagche ki korbo?",
"onek thanks",

# --- SET 6: Ambiguous ---
"eta explain koro",
"eta keno lage?",
"eta kivabe kaj kore?",
"aro easy kore bolo",
"example dao",
"eta solve koro",
"eta thik naki vul?",
"bujhte parsi na",
"abar bolo",
"short answer dao",

# --- SET 7: Emotional + support ---
"i feel like giving up",
"amar life e pressure onek",
"i don't know what to do next",
"amar kono goal nai",
"i feel ignored",
"ami khub disappointed",
"i am frustrated",
"help me relax",
"i feel empty",
"inspire me",

# --- SET 8: Deep thinking ---
"what is success?",
"can truth change over time?",
"if everything is relative what is absolute?",
"is time real or illusion?",
"what makes a human human?",
"why do we fear death?",
"what is intelligence?",
"is emotion logical?",
"can machines think?",
"what is self-awareness?",

# --- SET 9: Trick / hallucination ---
"who is president of mars?",
"moon er currency ki?",
"bangladesh ki olympic football champion?",
"who invented oxygen in 2023?",
"aliens kon language use kore?",
"earth er king ke?",
"sun ki ice diye toyri?",
"manush ki underwater thakar jonno banano?",
"xyzland bole kono desh ase?",
"who discovered dark fire?",

# --- SET 10: Complex mixed ---
"i am sad explain ai and motivate me",
"what is api and calculate 15+25 and give example",
"i feel lost what should i do in life",
"backend ki and ami keno bujhte pari na",
"who is king of mars and logically explain",
"ami confused frontend naki backend choose korbo",
"explain everything about internet simply",
"prove 2=5 using wrong logic",
"ami lonely but amar family ase keno?",
"tell me about universe and admit if unsure",
]

# =========================
# TEST RUN
# =========================

results = []
strong = 0
weak = 0
low = 0
no_answer = 0

print("=" * 80)
print("100-QUESTION CHATBOT TEST (DATASET MODE)")
print(f"Model: {len(bot.questions)} questions, {len(bot.category_store_map)} categories")
print("=" * 80)

for i, question in enumerate(QUESTIONS):

    if not question.strip():
        results.append({"q": "(empty)", "status": "SKIP"})
        continue

    result = bot.get_answer(question)

    if result is None:
        status = "NO_ANSWER"
        confidence = 0
        reply = None
        no_answer += 1

    elif result.get("suggestions"):
        status = "SUGGESTIONS"
        confidence = 0
        reply = None
        no_answer += 1

    else:
        confidence = result.get("confidence", 0)
        reply = result.get("reply", "")[:100]
        cats = result.get("categories", [])
        got_cats = ",".join(cats) if cats else str(result.get("intent", "?"))

        # Confidence-based scoring
        if confidence >= 0.7:
            status = "STRONG"
            strong += 1
        elif confidence >= 0.4:
            status = "WEAK"
            weak += 1
        else:
            status = "LOW"
            low += 1

    results.append({
        "q": question,
        "status": status,
        "confidence": confidence,
        "reply": reply
    })

    marker = {
        "STRONG": "Y",
        "WEAK": "~",
        "LOW": "!",
        "NO_ANSWER": "X",
        "SUGGESTIONS": "?"
    }.get(status, "-")

    print(f"{i+1:3}. {marker} {question[:45]:45} | {status:10} | {confidence:.0%}")

# =========================
# SUMMARY
# =========================

print("\n" + "=" * 80)
print("RESULT SUMMARY")
print(f"STRONG:     {strong}")
print(f"WEAK:       {weak}")
print(f"LOW:        {low}")
print(f"NO ANSWER:  {no_answer}")
print("=" * 80)

# =========================
# SAVE RESULTS
# =========================

with open("test_100q_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "total": len(QUESTIONS),
        "strong": strong,
        "weak": weak,
        "low": low,
        "no_answer": no_answer,
        "details": results
    }, f, ensure_ascii=False, indent=2)

print("\nSaved: test_100q_results.json")