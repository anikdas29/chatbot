"""
100 Questions across 100 Different Categories — Real User Style
Tests: typos, informal language, Bangla, short queries, complex questions
"""

import sys
import json
import time

# Each tuple: (question, expected_category, difficulty_tag)
TEST_QUESTIONS = [
    # --- CODING / TECH (40 questions) ---
    ("why python is so populer?", "python", "typo"),
    ("in c# arraylist vs list", "csharp", "comparison"),
    ("what is react hooks?", "react", "direct"),
    ("how to connect mongodb with node?", "mongodb", "multi-topic"),
    ("sql join types explain koro", "sql", "bangla-mix"),
    ("what is REST API?", "rest_api", "direct"),
    ("flask vs django which is better?", "flask_framework", "comparison"),
    ("how javascript async await works?", "javascript", "technical"),
    ("html ar css diye ki ki kora jay?", "html_css", "bangla"),
    ("what is docker and why use it?", "docker_basics", "direct"),
    ("kubernetes ki jinish?", "kubernetes", "bangla"),
    ("how to use git branch?", "git", "direct"),
    ("github pull request kivabe kore?", "github_basics", "bangla"),
    ("java vs kotlin for android", "kotlin", "comparison"),
    ("what is typescript?", "typescript", "direct"),
    ("rust programming language er advantages ki?", "rust", "bangla-mix"),
    ("how to deploy on aws?", "aws", "direct"),
    ("graphql vs rest api difference", "graphql", "comparison"),
    ("what is jwt token?", "jwt", "direct"),
    ("how to setup redis cache?", "redis", "direct"),
    ("postgresql vs mysql which better?", "postgresql", "comparison"),
    ("tailwind css ki?", "tailwind", "bangla"),
    ("what is next.js?", "nextjs", "direct"),
    ("angular vs react vs vue", "angular", "comparison"),
    ("express.js diye server banabo kivabe?", "express", "bangla"),
    ("laravel framework ki?", "laravel", "bangla"),
    ("devops ki and ci cd ki?", "devops", "bangla-mix"),
    ("how to write unit test?", "testing", "direct"),
    ("linux basic commands", "linux", "short"),
    ("what is cybersecurity?", "cybersecurity", "direct"),
    ("data structure ar algorithm shikbo kivabe?", "data_structures", "bangla"),
    ("what is oauth 2.0?", "oauth", "direct"),
    ("c++ vs java performance", "cpp", "comparison"),
    ("flutter diye mobile app banano", "flutter", "bangla"),
    ("swift programming for ios", "swift", "direct"),
    ("ruby on rails ki?", "rails", "bangla"),
    ("how websocket works?", "websocket", "direct"),
    ("webpack vs vite which is faster?", "vite", "comparison"),
    ("php ki ekhono use hoy?", "php", "bangla"),
    ("api ki jinish?", "api", "bangla-short"),

    # --- GENERAL KNOWLEDGE (20 questions) ---
    ("how to improve helth condiiton?", "health", "typo"),
    ("whcih fruite provide best protine?", "nutrition", "typo"),
    ("how to shourdown laptop?", "technology", "typo"),
    ("difference between mobile and laptop", "technology", "comparison"),
    ("why messi is best then ronaldo?", "football", "opinion"),
    ("how to lose weight fast?", "fitness", "direct"),
    ("best way to manage time?", "time_management", "direct"),
    ("ki khele weight kombe?", "diet_plans", "bangla"),
    ("how to deal with stress?", "stress_management", "direct"),
    ("how to save money?", "budgeting", "direct"),
    ("bitcoin ki safe invest?", "crypto", "bangla-mix"),
    ("how to start a business?", "startup", "direct"),
    ("climate change ki?", "environment", "bangla"),
    ("how to cook rice properly?", "cooking", "direct"),
    ("best books for self improvement?", "books", "direct"),
    ("how to take good photos?", "photography", "direct"),
    ("gardening tips for beginners", "gardening", "direct"),
    ("how to learn a new language?", "language", "direct"),
    ("what is meditation?", "meditation", "direct"),
    ("yoga benefits ki ki?", "yoga", "bangla"),

    # --- CAREER / EDUCATION (15 questions) ---
    ("how to write a good resume?", "resume", "direct"),
    ("interview tips for freshers", "interview", "direct"),
    ("freelancing kivabe shuru korbo?", "freelancing", "bangla"),
    ("how to study effectively?", "study", "direct"),
    ("salary negotiation tips", "salary", "direct"),
    ("career change korbo kivabe?", "career", "direct"),
    ("best college for computer science?", "college", "direct"),
    ("motivation nai ki korbo?", "motivation", "bangla"),
    ("how to write good code?", "clean_code", "direct"),
    ("leadership skills improve korbo kivabe?", "leadership", "bangla"),
    ("marketing strategy ki?", "marketing", "direct"),
    ("how to be more productive?", "productivity", "direct"),
    ("exam preparation tips", "exam_preparation", "direct"),
    ("public speaking fear kivabe overcome korbo?", "communication", "bangla"),
    ("how to write technical documents?", "technical_writing", "direct"),

    # --- LIFESTYLE / MISC (15 questions) ---
    ("i feel so lonely", "loneliness", "emotion"),
    ("ami onek depressed feel korchi", "depression_support", "bangla-emotion"),
    ("anger control kivabe korbo?", "anger_management", "bangla"),
    ("how to sleep better at night?", "sleep", "direct"),
    ("pet cat er care kivabe korbo?", "pets", "bangla"),
    ("relationship advice dao", "relationship", "bangla-mix"),
    ("best cricket player of all time?", "cricket", "opinion"),
    ("basketball rules ki?", "basketball", "bangla"),
    ("how to play chess?", "puzzle_games", "direct"),
    ("music shunle ki mental health e help kore?", "music", "bangla"),
    ("fashion trend 2024", "fashion", "direct"),
    ("science facts for kids", "science", "direct"),
    ("history of world war 2", "history", "direct"),
    ("first aid basics", "first_aid", "direct"),
    ("what is cloud computing?", "cloud", "direct"),

    # --- BOT META (10 questions) ---
    ("tumi ki paro?", "bot_capability", "bangla"),
    ("what is your name?", "bot_name", "direct"),
    ("who made you?", "about_bot", "direct"),
    ("tumi ki ai?", "ai", "bangla"),
    ("tell me a quote", "quotes", "direct"),
    ("machine learning ki?", "ml", "bangla"),
    ("deep learning vs machine learning", "dl", "comparison"),
    ("blockchain ki?", "blockchain_basics", "bangla"),
    ("what is quantum computing?", "quantum_computing", "direct"),
    ("solar energy er future ki?", "solar_energy", "bangla"),
]


def run_test():
    """Run all 100 questions and analyze results."""
    sys.path.insert(0, ".")

    # Skip TinyLlama to speed up test — we only care about category detection accuracy
    import chatbot as cb
    _orig_init = cb.TinyLlamaGenerator.__init__
    def _skip_llm(self, *a, **kw):
        self.model = None
        self.available = False
    cb.TinyLlamaGenerator.__init__ = _skip_llm

    from chatbot import ChatBot

    print("=" * 70)
    print("  100 QUESTIONS x 100 CATEGORIES — REAL USER STYLE TEST")
    print("=" * 70)
    print("\nLoading chatbot (LLM skipped for speed)...")

    bot = ChatBot()
    session_id = bot.db.create_session()

    results = []
    correct = 0
    wrong = 0
    wrong_details = []

    print(f"\nRunning {len(TEST_QUESTIONS)} questions...\n")
    print(f"{'#':<4} {'Result':<8} {'Conf':<7} {'Expected':<25} {'Got':<25} {'Question'}")
    print("-" * 120)

    for i, (question, expected, tag) in enumerate(TEST_QUESTIONS, 1):
        start = time.time()
        result = bot.get_answer(question, session_id)
        elapsed = time.time() - start

        if result is None:
            got_cat = "UNKNOWN"
            confidence = 0
            reply = ""
        elif result.get("suggestions"):
            got_cat = "SUGGESTIONS"
            confidence = 0
            reply = ""
        else:
            got_cats = result.get("categories", [])
            got_cat = got_cats[0] if got_cats else result.get("intent", "???")
            confidence = result.get("confidence", 0)
            reply = result.get("reply", "")

        # Check if correct (primary category matches expected)
        is_correct = (got_cat.lower().strip() == expected.lower().strip())

        # Also accept if expected is in any of the returned categories
        if not is_correct and result and result.get("categories"):
            is_correct = expected.lower() in [c.lower() for c in result.get("categories", [])]

        status = "OK" if is_correct else "WRONG"
        if is_correct:
            correct += 1
        else:
            wrong += 1
            wrong_details.append({
                "num": i,
                "question": question,
                "expected": expected,
                "got": got_cat,
                "all_categories": result.get("categories", []) if result else [],
                "confidence": confidence,
                "reply": reply[:100] if reply else "",
                "tag": tag,
                "time_ms": round(elapsed * 1000)
            })

        conf_str = f"{confidence:.0%}" if confidence else "—"
        print(f"{i:<4} {status:<8} {conf_str:<7} {expected:<25} {got_cat:<25} {question[:50]}")

        results.append({
            "num": i,
            "question": question,
            "expected": expected,
            "got": got_cat,
            "correct": is_correct,
            "confidence": confidence,
            "tag": tag,
            "time_ms": round(elapsed * 1000)
        })

    # ============ SUMMARY ============
    print("\n" + "=" * 70)
    print(f"  RESULTS: {correct}/{len(TEST_QUESTIONS)} correct ({correct/len(TEST_QUESTIONS):.0%})")
    print(f"  Wrong: {wrong}")
    print("=" * 70)

    # Break down by tag
    tag_stats = {}
    for r in results:
        tag = r["tag"]
        if tag not in tag_stats:
            tag_stats[tag] = {"total": 0, "correct": 0}
        tag_stats[tag]["total"] += 1
        if r["correct"]:
            tag_stats[tag]["correct"] += 1

    print("\n--- Accuracy by Question Type ---")
    print(f"{'Type':<20} {'Score':<15} {'Accuracy'}")
    print("-" * 50)
    for tag, stats in sorted(tag_stats.items(), key=lambda x: x[1]["correct"]/x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        bar = "#" * round(acc * 10) + "." * (10 - round(acc * 10))
        print(f"{tag:<20} {stats['correct']}/{stats['total']:<12} {bar} {acc:.0%}")

    # Wrong answers detail
    if wrong_details:
        print(f"\n{'=' * 70}")
        print(f"  WRONG ANSWERS — DETAILED ANALYSIS ({len(wrong_details)} questions)")
        print(f"{'=' * 70}\n")

        for w in wrong_details:
            print(f"  Q{w['num']}: \"{w['question']}\"")
            print(f"     Expected: {w['expected']}")
            print(f"     Got:      {w['got']} (all: {w['all_categories']})")
            print(f"     Conf:     {w['confidence']:.0%}")
            print(f"     Reply:    {w['reply'][:80]}...")
            print(f"     Type:     {w['tag']}")
            print(f"     Why wrong: ", end="")

            # Diagnose WHY it's wrong
            if w['got'] == "UNKNOWN":
                print("Bot found NO matching category at all.")
            elif w['got'] == "SUGGESTIONS":
                print("Confidence too low — showed 'Did you mean?' suggestions.")
            elif w['confidence'] < 0.40:
                print(f"Very low confidence ({w['confidence']:.0%}) — weak semantic match.")
            else:
                print(f"Matched '{w['got']}' instead of '{w['expected']}' — semantic confusion between similar topics.")
            print()

    # Save full results to JSON
    output = {
        "total": len(TEST_QUESTIONS),
        "correct": correct,
        "wrong": wrong,
        "accuracy": round(correct / len(TEST_QUESTIONS), 4),
        "by_tag": {tag: {"total": s["total"], "correct": s["correct"],
                         "accuracy": round(s["correct"]/s["total"], 4)}
                   for tag, s in tag_stats.items()},
        "wrong_details": wrong_details,
        "all_results": results
    }

    with open("test_100_categories_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to: test_100_categories_results.json")
    return output


if __name__ == "__main__":
    run_test()
