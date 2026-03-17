"""
Quick 10-Question Test — Verify answer quality after pipeline changes
Run: python test_10_quick.py
"""

import sys
import json
import time
import functools

print = functools.partial(print, flush=True)

TEST_QUESTIONS = [
    # 1. Python: basic → intermediate → advanced → expert → real-world
    ("what is python?", "python", "basic"),
    ("python list vs tuple vs set difference ki?", "python", "intermediate"),
    ("python decorator kivabe kaj kore?", "python", "advanced"),
    ("python GIL ki and multithreading e ki problem hoy?", "python", "expert"),
    ("python diye REST API banabo kivabe flask use kore?", "flask_framework", "real-world"),

    # 2. JavaScript: basic → intermediate → advanced → expert → real-world
    ("javascript ki?", "javascript", "basic"),
    ("javascript closure ki jinish?", "javascript", "intermediate"),
    ("javascript event loop and callback queue explain koro", "javascript", "advanced"),
    ("javascript prototype chain and inheritance kivabe kore?", "javascript", "expert"),
    ("react e state management kivabe kore redux chara?", "react", "real-world"),

    # 3. Database: basic → intermediate → advanced → expert → real-world
    ("database ki?", "database", "basic"),
    ("sql vs nosql difference ki?", "sql", "intermediate"),
    ("database indexing kivabe performance improve kore?", "database", "advanced"),
    ("database sharding vs replication difference ki?", "database", "expert"),
    ("mongodb e aggregation pipeline kivabe use korbo?", "mongodb", "real-world"),

    # 4. ML/AI: basic → intermediate → advanced → expert → real-world
    ("machine learning ki?", "ml", "basic"),
    ("supervised vs unsupervised learning er difference ki?", "ml", "intermediate"),
    ("neural network er backpropagation kivabe kaj kore?", "dl", "advanced"),
    ("overfitting reduce korte regularization kivabe help kore?", "ml", "expert"),
    ("python diye image classification model banabo kivabe?", "ml", "real-world"),

    # 5. DevOps: basic → intermediate → advanced → expert → real-world
    ("docker ki?", "docker_basics", "basic"),
    ("docker image vs container difference ki?", "docker_basics", "intermediate"),
    ("dockerfile kivabe optimize korbo layer caching diye?", "docker_basics", "advanced"),
    ("kubernetes pod vs deployment vs service difference ki?", "kubernetes", "expert"),
    ("ci cd pipeline setup korbo github actions diye kivabe?", "ci_cd", "real-world"),

    # 6. Security: basic → intermediate → advanced → expert → real-world
    ("cybersecurity ki?", "cybersecurity", "basic"),
    ("sql injection ki and kivabe prevent korbo?", "cybersecurity", "intermediate"),
    ("oauth 2.0 authorization flow kivabe kaj kore?", "oauth", "advanced"),
    ("jwt token e refresh token vs access token difference ki?", "jwt", "expert"),
    ("https ssl certificate kivabe setup korbo?", "cybersecurity", "real-world"),

    # 7. Web Dev: basic → intermediate → advanced → expert → real-world
    ("html css ki?", "html_css", "basic"),
    ("css flexbox vs grid difference ki?", "html_css", "intermediate"),
    ("responsive design kivabe korbo media query diye?", "html_css", "advanced"),
    ("webpack vs vite difference ki bundling e?", "vite", "expert"),
    ("next.js e server side rendering kivabe kaj kore?", "nextjs", "real-world"),

    # 8. Career: basic → intermediate → advanced → expert → real-world
    ("coding shikbo kivabe?", "coding", "basic"),
    ("data structures and algorithms ki ki shikha dorkar?", "data_structures", "intermediate"),
    ("system design interview e ki ki prepare korbo?", "interview", "advanced"),
    ("microservices architecture er pros and cons ki?", "microservices", "expert"),
    ("freelancing e client kivabe pabo upwork chara?", "freelancing", "real-world"),

    # 9. Cloud: basic → intermediate → advanced → expert → real-world
    ("cloud computing ki?", "cloud", "basic"),
    ("aws vs azure vs gcp konta better?", "aws", "intermediate"),
    ("serverless computing ki and lambda function kivabe kaj kore?", "aws", "advanced"),
    ("terraform diye infrastructure as code kivabe kore?", "terraform", "expert"),
    ("kubernetes e auto scaling kivabe setup korbo?", "kubernetes", "real-world"),

    # 10. Git/Version Control: basic → intermediate → advanced → expert → real-world
    ("git ki jinish?", "git", "basic"),
    ("git merge vs rebase difference ki?", "git", "intermediate"),
    ("git cherry-pick kivabe use korbo?", "git", "advanced"),
    ("git bisect diye bug kivabe khujbo?", "git", "expert"),
    ("github actions e automated testing setup korbo kivabe?", "ci_cd", "real-world"),
]


def run_test():
    fast_mode = "--fast" in sys.argv
    sys.path.insert(0, ".")

    if fast_mode:
        import chatbot as cb
        def _skip_llm(self, *a, **kw):
            self.model = None
            self.available = False
        cb.TinyLlamaGenerator.__init__ = _skip_llm

    from chatbot import ChatBot

    print("=" * 60)
    print("  QUICK 10-QUESTION TEST")
    print("=" * 60)
    mode = "FAST (no LLM)" if fast_mode else "FULL (TinyLlama ON)"
    print(f"  Mode: {mode}")
    print("\nLoading chatbot...")

    bot = ChatBot()
    session_id = bot.db.create_session()
    print(f"  Loaded | TinyLlama: {'ON' if bot.generator.available else 'OFF'}")
    print(f"\nRunning {len(TEST_QUESTIONS)} questions...\n")

    results = []
    correct = 0
    output_file = "test_10_quick_results.json"

    for i, (question, expected, tag) in enumerate(TEST_QUESTIONS, 1):
        start = time.time()
        result = bot.get_answer(question, session_id)
        elapsed = time.time() - start

        if result is None:
            got_cat, confidence, reply, generated = "UNKNOWN", 0, "", False
        elif result.get("suggestions"):
            got_cat, confidence, reply, generated = "SUGGESTIONS", 0, "", False
        else:
            cats = result.get("categories", [])
            got_cat = cats[0] if cats else "???"
            confidence = result.get("confidence", 0)
            reply = result.get("reply", "")
            generated = result.get("generated", False)

        is_correct = (got_cat.lower() == expected.lower())
        if not is_correct and result and result.get("categories"):
            is_correct = expected.lower() in [c.lower() for c in result["categories"]]

        if is_correct:
            correct += 1
        icon = "+" if is_correct else "X"
        gen_tag = " [LLM]" if generated else ""

        print(f"  [{icon}] Q{i}: {question}")
        if reply:
            short = reply.replace("\n", " ")
            if len(short) > 150:
                short = short[:150] + "..."
            print(f"       Bot: {short}")
        else:
            print(f"       Bot: (no answer)")
        print(f"       [{got_cat}] {confidence:.0%}{gen_tag} | {elapsed:.1f}s")
        if not is_correct:
            print(f"       Expected: {expected}")
        print()

        results.append({
            "num": i, "question": question, "expected": expected,
            "got": got_cat, "correct": is_correct, "confidence": confidence,
            "reply": reply, "tag": tag, "generated": generated,
            "time_s": round(elapsed, 1)
        })

        # Save after every question
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"completed": i, "correct": correct,
                        "accuracy": round(correct/i, 4), "results": results},
                       f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"  RESULT: {correct}/{len(TEST_QUESTIONS)} ({correct/len(TEST_QUESTIONS):.0%})")
    print("=" * 60)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
   run_test()
