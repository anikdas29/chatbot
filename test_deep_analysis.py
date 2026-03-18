"""
Deep analysis: 60 real-world questions to find root cause problems.
Simulates a company support bot scenario.
"""
import json
import time
import logging
logging.disable(logging.CRITICAL)

from chatbot import ChatBot

import sys
FAST_MODE = "--fast" in sys.argv

bot = ChatBot()
# Disable TinyLlama for fast testing
if FAST_MODE:
    bot.generator.available = False
    print("[FAST MODE: TinyLlama disabled]")

# 60 questions — mix of exact dataset, rephrased, Bangla, multi-topic, edge cases
QUESTIONS = [
    # === GROUP 1: Simple direct questions (should work perfectly) ===
    ("what is python?", "python", "direct match"),
    ("tell me about javascript", "javascript", "direct match"),
    ("what is machine learning?", "ml", "direct match"),
    ("how to use docker?", "docker_basics", "direct match"),
    ("what is git?", "git", "direct match"),

    # === GROUP 2: Rephrased questions (same intent, different words) ===
    ("i want to learn python, where do i start?", "python", "rephrased"),
    ("can you explain how ML works?", "ml", "rephrased"),
    ("whats the best way to get into web development?", "web_dev", "rephrased"),
    ("how do i make my website look good?", "html_css", "rephrased"),
    ("i need help with my database", "database", "rephrased"),

    # === GROUP 3: Short/Informal questions (like real users type) ===
    ("python", "python", "single word"),
    ("react", "react", "single word"),
    ("help", "help/unknown", "single word"),
    ("sql tips", "sql", "short"),
    ("git basics", "git", "short"),
    ("api kya hai?", "api/rest_api", "informal"),
    ("docker keno use kori?", "docker_basics", "informal"),
    ("css tricks", "html_css/css", "short"),
    ("linux commands", "linux", "short"),
    ("career advice", "career", "short"),

    # === GROUP 4: Company support bot simulation ===
    ("how do i reset my password?", "password/account", "support"),
    ("my app is not loading", "troubleshooting", "support"),
    ("how to contact support?", "contact/support", "support"),
    ("what are your pricing plans?", "pricing", "support"),
    ("how to cancel my subscription?", "subscription", "support"),
    ("where can i find the documentation?", "documentation", "support"),
    ("is there a mobile app?", "mobile_app", "support"),
    ("how to export my data?", "data_export", "support"),
    ("i found a bug", "bug_report", "support"),
    ("how to upgrade my plan?", "subscription/pricing", "support"),

    # === GROUP 5: Similar questions that should get DIFFERENT categories ===
    ("how to cook rice?", "cooking", "similar"),
    ("how to cook a python script?", "python", "tricky - cook=run"),
    ("what is a python snake?", "python(language) or animal?", "ambiguous"),
    ("how to run a program?", "programming", "similar"),
    ("how to run faster?", "fitness/running", "similar"),
    ("java coffee", "java(language) or coffee?", "ambiguous"),
    ("apple products", "apple(tech) or fruit?", "ambiguous"),
    ("how to handle stress at work?", "stress/mental_health", "similar"),
    ("how to handle errors in python?", "python/debugging", "similar"),
    ("spring framework", "spring/java", "similar"),

    # === GROUP 6: Questions that need understanding, not keyword matching ===
    ("i dont know what to do with my life", "career/motivation", "understanding"),
    ("everything keeps going wrong", "mental_health/motivation", "understanding"),
    ("whats the point of learning coding?", "programming/career", "understanding"),
    ("is it too late to switch careers?", "career", "understanding"),
    ("i keep failing my interviews", "interview/career", "understanding"),
    ("my code doesnt work and i dont know why", "debugging", "understanding"),
    ("im overwhelmed with too many technologies", "programming/career", "understanding"),
    ("how do i stay motivated?", "motivation", "understanding"),
    ("nobody uses my app", "marketing/startup", "understanding"),
    ("should i learn AI?", "ai/career", "understanding"),

    # === GROUP 7: Questions with typos and bad grammar ===
    ("hwo to lern pytohn?", "python", "typo"),
    ("waht is machne lerning?", "ml", "typo"),
    ("javscript tutoral pls", "javascript", "typo"),
    ("i wnt to mke a wesbite", "web_dev", "typo"),
    ("hw do i us git?", "git", "typo"),

    # === GROUP 8: Bangla/Banglish ===
    ("python ki?", "python", "bangla"),
    ("amake programming shikhao", "programming", "bangla"),
    ("machine learning kivabe kaj kore?", "ml", "bangla"),
    ("website banabo kivabe?", "web_dev", "bangla"),
    ("career niye confused", "career", "bangla"),
]

results = []
group_stats = {}

print("=" * 90)
print(f"DEEP ANALYSIS: 60 Questions | {len(bot.category_store_map)} categories | LLM: {bot.generator.model_name if bot.generator.available else 'OFF'}")
print("=" * 90)

for i, (question, expected, group) in enumerate(QUESTIONS):
    start = time.time()
    result = bot.get_answer(question)
    elapsed = time.time() - start

    if result is None:
        got_cats = "NO_ANSWER"
        reply = ""
        confidence = 0
        generated = False
    elif result.get("suggestions"):
        got_cats = "SUGGESTIONS"
        reply = ""
        confidence = 0
        generated = False
    else:
        cats = result.get("categories", [])
        got_cats = ",".join(cats)
        reply = result.get("reply", "")
        confidence = result.get("confidence", 0)
        generated = result.get("generated", False)

    # Check if expected category is in results
    expected_parts = [p.strip().lower() for p in expected.replace("+", "/").replace(",", "/").split("/")]
    got_lower = got_cats.lower()

    matched = False
    if "unknown" in expected.lower() or "?" in expected:
        matched = True  # ambiguous, skip
    else:
        for part in expected_parts:
            if part and part in got_lower:
                matched = True
                break

    status = "OK" if matched else "WRONG"
    if got_cats in ("NO_ANSWER", "SUGGESTIONS") and "unknown" not in expected.lower():
        status = "MISS"

    # Track group stats
    group_type = group.split(" ")[0] if " " in group else group
    if group_type not in group_stats:
        group_stats[group_type] = {"total": 0, "ok": 0, "wrong": 0, "miss": 0}
    group_stats[group_type]["total"] += 1
    group_stats[group_type][status.lower()] = group_stats[group_type].get(status.lower(), 0) + 1

    source = "LLM" if generated else "DS"

    entry = {
        "q": question,
        "expected": expected,
        "got": got_cats,
        "confidence": round(confidence, 3),
        "status": status,
        "group": group,
        "source": source,
        "time": round(elapsed, 2),
        "reply_preview": reply[:120] if reply else "",
    }
    results.append(entry)

    mark = "Y" if status == "OK" else ("X" if status == "WRONG" else "-")
    print(f"  {i+1:2}. {mark} [{source:3}] {elapsed:5.1f}s | Q: {question[:40]:40} | Exp: {expected[:20]:20} | Got: {got_cats[:30]:30} | {confidence:.0%}")

# === ANALYSIS ===
print("\n" + "=" * 90)
total_ok = sum(1 for r in results if r["status"] == "OK")
total_wrong = sum(1 for r in results if r["status"] == "WRONG")
total_miss = sum(1 for r in results if r["status"] == "MISS")
print(f"TOTAL: {total_ok}/{len(results)} correct | {total_wrong} wrong | {total_miss} missed")

print("\n--- GROUP ACCURACY ---")
for group, stats in sorted(group_stats.items()):
    ok = stats.get("ok", 0)
    total = stats["total"]
    pct = ok / total * 100 if total > 0 else 0
    print(f"  {group:20} : {ok}/{total} ({pct:.0f}%)")

print("\n--- WRONG ANSWERS DETAIL ---")
for r in results:
    if r["status"] in ("WRONG", "MISS"):
        print(f"\n  Q: {r['q']}")
        print(f"  Expected: {r['expected']}")
        print(f"  Got: {r['got']} ({r['confidence']:.0%})")
        print(f"  Group: {r['group']}")
        if r["reply_preview"]:
            print(f"  Reply: {r['reply_preview']}")

print("\n--- SPEED ANALYSIS ---")
llm_times = [r["time"] for r in results if r["source"] == "LLM"]
ds_times = [r["time"] for r in results if r["source"] == "DS"]
if llm_times:
    print(f"  TinyLlama avg: {sum(llm_times)/len(llm_times):.1f}s (min {min(llm_times):.1f}s, max {max(llm_times):.1f}s)")
if ds_times:
    print(f"  Dataset avg: {sum(ds_times)/len(ds_times):.2f}s")

# Save
with open("test_deep_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "total": len(results),
        "correct": total_ok,
        "wrong": total_wrong,
        "missed": total_miss,
        "group_stats": group_stats,
        "results": results,
    }, f, ensure_ascii=False, indent=2)

print(f"\nSaved to test_deep_results.json")
