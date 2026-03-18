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
    # 4. ML/AI: basic → intermediate → advanced → expert → real-world
    ("machine learning ki?", "ml", "basic"),
    ("supervised vs unsupervised learning er difference ki?", "ml", "intermediate"),
    ("neural network er backpropagation kivabe kaj kore?", "dl", "advanced"),
    ("overfitting reduce korte regularization kivabe help kore?", "ml", "expert"),
    ("python diye image classification model banabo kivabe?", "ml", "real-world"),
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
    mode = "FAST (no LLM)" if fast_mode else "FULL (LLM ON)"
    print(f"  Mode: {mode}")
    print("\nLoading chatbot...")

    bot = ChatBot()
    session_id = bot.db.create_session()
    print(f"  Loaded | LLM: {bot.generator.model_name if bot.generator.available else 'OFF'}")
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
