"""
100 Questions across 100 Different Categories — Real User Style
Tests: typos, informal language, Bangla, short queries, complex questions
Run: python test_100_categories.py          (full mode with TinyLlama)
     python test_100_categories.py --fast   (skip TinyLlama, faster)
"""

import sys
import json
import time
import functools

# Force flush on every print so terminal shows live progress
print = functools.partial(print, flush=True)

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
    fast_mode = "--fast" in sys.argv
    sys.path.insert(0, ".")

    if fast_mode:
        import chatbot as cb
        def _skip_llm(self, *a, **kw):
            self.model = None
            self.available = False
        cb.TinyLlamaGenerator.__init__ = _skip_llm

    from chatbot import ChatBot

    print("=" * 70)
    print("  100 QUESTIONS x 100 CATEGORIES")
    print("=" * 70)
    if fast_mode:
        print("  Mode: FAST (LLM skipped)")
    else:
        print("  Mode: FULL (LLM ON — real bot answers)")
        print("  Note: First load takes ~60-70s, then ~3-8s per question")
    print("\nLoading chatbot...", flush=True)

    load_start = time.time()
    bot = ChatBot()
    load_time = time.time() - load_start
    session_id = bot.db.create_session()
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  LLM: {bot.generator.model_name if bot.generator.available else 'OFF'}")
    print(f"  Questions: {len(bot.questions)}")
    print(f"  Categories: {len(bot.category_store_map)}")

    results = []
    correct = 0
    wrong = 0
    wrong_details = []
    total_time = 0
    output_file = "test_100_categories_results.json"

    def save_progress():
        """Save results after every question so nothing is lost."""
        avg = (total_time / len(results)) if results else 0
        tag_stats = {}
        for r in results:
            t = r["tag"]
            if t not in tag_stats:
                tag_stats[t] = {"total": 0, "correct": 0}
            tag_stats[t]["total"] += 1
            if r["correct"]:
                tag_stats[t]["correct"] += 1
        output = {
            "total_questions": len(TEST_QUESTIONS),
            "completed": len(results),
            "correct": correct,
            "wrong": wrong,
            "accuracy": round(correct / len(results), 4) if results else 0,
            "total_time_s": round(total_time, 1),
            "avg_time_ms": round(avg * 1000),
            "llm_enabled": not fast_mode and llm_status == "ON",
            "by_tag": {t: {"total": s["total"], "correct": s["correct"],
                            "accuracy": round(s["correct"]/s["total"], 4)}
                       for t, s in tag_stats.items()},
            "wrong_details": wrong_details,
            "all_results": results
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nRunning {len(TEST_QUESTIONS)} questions...\n")

    for i, (question, expected, tag) in enumerate(TEST_QUESTIONS, 1):
        start = time.time()
        result = bot.get_answer(question, session_id)
        elapsed = time.time() - start
        total_time += elapsed

        if result is None:
            got_cat = "UNKNOWN"
            confidence = 0
            reply = ""
            generated = False
        elif result.get("suggestions"):
            got_cat = "SUGGESTIONS"
            confidence = 0
            reply = ""
            generated = False
        else:
            got_cats = result.get("categories", [])
            got_cat = got_cats[0] if got_cats else result.get("intent", "???")
            confidence = result.get("confidence", 0)
            reply = result.get("reply", "")
            generated = result.get("generated", False)

        # Check if correct
        is_correct = (got_cat.lower().strip() == expected.lower().strip())
        if not is_correct and result and result.get("categories"):
            is_correct = expected.lower() in [c.lower() for c in result.get("categories", [])]

        if is_correct:
            correct += 1
            status_icon = "+"
        else:
            wrong += 1
            status_icon = "X"
            wrong_details.append({
                "num": i,
                "question": question,
                "expected": expected,
                "got": got_cat,
                "all_categories": result.get("categories", []) if result else [],
                "confidence": confidence,
                "reply": reply,
                "tag": tag,
                "time_ms": round(elapsed * 1000),
                "generated": generated
            })

        # Print like real chat
        conf_pct = f"{confidence:.0%}" if confidence else "---"
        gen_tag = " [LLM]" if generated else ""
        time_str = f"{elapsed:.1f}s"
        acc_pct = f"{correct/i:.0%}"

        print(f"  [{status_icon}] Q{i:>3}/{len(TEST_QUESTIONS)}: {question}")
        if reply:
            short_reply = reply.replace("\n", " ")
            if len(short_reply) > 120:
                short_reply = short_reply[:120] + "..."
            print(f"         Bot: {short_reply}")
        elif got_cat == "SUGGESTIONS":
            print(f"         Bot: (Did you mean? suggestions shown)")
        else:
            print(f"         Bot: (no answer)")
        print(f"         [{got_cat}] {conf_pct}{gen_tag} | {time_str} | Running: {acc_pct}")
        if not is_correct:
            print(f"         Expected: {expected}")
        print()

        results.append({
            "num": i,
            "question": question,
            "expected": expected,
            "got": got_cat,
            "correct": is_correct,
            "confidence": confidence,
            "reply": reply,
            "tag": tag,
            "time_ms": round(elapsed * 1000),
            "generated": generated
        })

        # Save after every question
        save_progress()

    # ============ SUMMARY ============
    avg_time = total_time / len(TEST_QUESTIONS)
    print("=" * 70)
    print(f"  RESULTS: {correct}/{len(TEST_QUESTIONS)} correct ({correct/len(TEST_QUESTIONS):.0%})")
    print(f"  Wrong: {wrong}")
    print(f"  Total time: {total_time:.1f}s | Avg: {avg_time:.2f}s/question")
    print("=" * 70)

    # Break down by tag
    tag_stats = {}
    for r in results:
        t = r["tag"]
        if t not in tag_stats:
            tag_stats[t] = {"total": 0, "correct": 0}
        tag_stats[t]["total"] += 1
        if r["correct"]:
            tag_stats[t]["correct"] += 1

    print("\n--- Accuracy by Question Type ---")
    print(f"{'Type':<20} {'Score':<15} {'Accuracy'}")
    print("-" * 50)
    for t, stats in sorted(tag_stats.items(), key=lambda x: x[1]["correct"]/x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        bar = "#" * round(acc * 10) + "." * (10 - round(acc * 10))
        print(f"{t:<20} {stats['correct']}/{stats['total']:<12} {bar} {acc:.0%}")

    # Wrong answers detail
    if wrong_details:
        print(f"\n{'=' * 70}")
        print(f"  WRONG ANSWERS ({len(wrong_details)})")
        print(f"{'=' * 70}\n")

        for w in wrong_details:
            print(f"  Q{w['num']}: \"{w['question']}\"")
            print(f"     Expected: {w['expected']}")
            print(f"     Got:      {w['got']} (all: {w['all_categories']})")
            print(f"     Conf:     {w['confidence']:.0%}")
            short = w['reply'].replace('\n', ' ')[:100] if w['reply'] else "(none)"
            print(f"     Reply:    {short}")
            gen = " [LLM generated]" if w.get('generated') else ""
            print(f"     Type:     {w['tag']}{gen}")
            print()

    # Final save (already saved incrementally, this is the final version)
    save_progress()
    print(f"\nResults saved to: {output_file}")
    return


if __name__ == "__main__":
    run_test()
