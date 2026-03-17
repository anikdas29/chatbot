"""
Dataset Auto-Updater
1500 category JSON files automatically update korbe Ollama (local LLM) use kore.
Rate limiting + resume support + progress tracking included.
FREE & UNLIMITED — no API key needed, runs fully offline.
"""

import json
import os
import sys
import time
import logging
from pathlib import Path

try:
    import ollama
except ImportError:
    print("❌ ollama package nai! Install koro: pip install ollama")
    sys.exit(1)

# ============================================================
# CONFIG
# ============================================================
DATASET_FOLDERS = ["category_wise_dataset", "coding_dataset"]
MODEL_NAME = "llama3.2"                    # ollama model name
PROGRESS_FILE = "update_progress.json"     # resume support
BATCH_SIZE = 10                            # progress save every 10
DELAY_BETWEEN_CALLS = 0.5                  # local = fast, minimal delay

logging.basicConfig(
    filename="dataset_update.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ============================================================
# PROMPT TEMPLATE (unchanged)
# ============================================================
def build_prompt(category_name, cat_type, existing_questions, existing_answers):
    return f"""You are a dataset builder for a chatbot. I have a category called "{category_name}" (type: {cat_type}).

Current questions ({len(existing_questions)} existing):
{json.dumps(existing_questions[:5], ensure_ascii=False, indent=2)}
(showing first 5 only)

Current answers ({len(existing_answers)} existing):
{json.dumps(existing_answers[:3], ensure_ascii=False, indent=2)}
(showing first 3 only)

Your job: Generate NEW questions and answers to ADD to this category.
Focus on: basic → intermediate → advanced → expert level coverage.
Include both English and Bangla question variations.

RULES:
- Generate exactly 10 new questions and 6 new answers
- Questions: mix of basic, intermediate, advanced
- Answers: detailed, accurate, 2-4 sentences each
- Include Bangla versions of questions (e.g. "X ki?", "X kivabe kaje kore?")
- Do NOT repeat existing content
- Return ONLY valid JSON, no explanation, no markdown

Return this exact format:
{{
  "new_questions": ["q1", "q2", ...],
  "new_answers": ["a1", "a2", ...]
}}"""


# ============================================================
# PROGRESS TRACKER
# ============================================================
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ============================================================
# SINGLE FILE UPDATER
# ============================================================
def update_single_file(client, filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    category = data.get("category", Path(filepath).stem)
    cat_type = data.get("type", "general")
    existing_questions = data.get("questions", [])
    existing_answers = data.get("answers", [])

    # Skip if already has enough content
    if len(existing_questions) >= 30 and len(existing_answers) >= 15:
        logging.info(f"SKIP (already rich): {category}")
        return "skipped"

    prompt = build_prompt(category, cat_type, existing_questions, existing_answers)

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7, "num_predict": 1000}
        )

        response_text = response["message"]["content"].strip()

        # Clean markdown if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        new_data = json.loads(response_text)
        new_questions = new_data.get("new_questions", [])
        new_answers = new_data.get("new_answers", [])

        # Merge — avoid duplicates
        for q in new_questions:
            if q not in existing_questions:
                existing_questions.append(q)

        for a in new_answers:
            if a not in existing_answers:
                existing_answers.append(a)

        data["questions"] = existing_questions
        data["answers"] = existing_answers

        # Save updated file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logging.info(f"UPDATED: {category} | +{len(new_questions)}Q +{len(new_answers)}A")
        return "success"

    except json.JSONDecodeError as e:
        logging.error(f"JSON PARSE ERROR: {category} | {e}")
        return "failed"
    except Exception as e:
        logging.error(f"ERROR: {category} | {e}")
        return "failed"


# ============================================================
# MAIN RUNNER
# ============================================================
def run_updater():
    # ── Check 1: Ollama running? ──
    client = ollama.Client()
    try:
        client.list()
    except Exception:
        print("❌ Ollama chole na! Terminal e run koro: ollama serve")
        sys.exit(1)

    # ── Check 2: Model exists? ──
    models = client.list()
    model_names = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
    # ollama returns names like "llama3.2:latest" — check prefix
    model_found = any(MODEL_NAME in name for name in model_names)
    if not model_found:
        print(f"❌ Model '{MODEL_NAME}' nai! Terminal e run koro: ollama pull {MODEL_NAME}")
        print(f"   Available models: {model_names}")
        sys.exit(1)

    print(f"✅ Ollama running | Model: {MODEL_NAME} | FREE & UNLIMITED")
    print()

    progress = load_progress()
    completed = set(progress["completed"])
    failed = set(progress["failed"])

    # Get all JSON files from all folders
    all_files = []
    for folder in DATASET_FOLDERS:
        if os.path.isdir(folder):
            all_files.extend(list(Path(folder).glob("*.json")))

    total = len(all_files)
    remaining = total - len(completed)
    print(f"Total files: {total}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {remaining}")
    print("=" * 50)

    if remaining == 0:
        print("All files already updated! Delete update_progress.json to re-run.")
        return

    success_count = 0
    fail_count = 0
    skip_count = 0
    start_time = time.time()

    for i, filepath in enumerate(all_files):
        fname = str(filepath)

        # Skip already completed
        if fname in completed:
            continue

        elapsed = time.time() - start_time
        rate = (success_count + skip_count + fail_count) / max(elapsed, 1) * 60
        print(f"[{i+1}/{total}] {filepath.stem}...", end=" ", flush=True)

        result = update_single_file(client, filepath)

        if result == "success":
            completed.add(fname)
            progress["completed"] = list(completed)
            success_count += 1
            print("done")
        elif result == "skipped":
            completed.add(fname)
            progress["completed"] = list(completed)
            skip_count += 1
            print("skip")
        else:
            failed.add(fname)
            progress["failed"] = list(failed)
            fail_count += 1
            print("FAILED")

        # Save progress every BATCH_SIZE files
        if (success_count + fail_count + skip_count) % BATCH_SIZE == 0:
            save_progress(progress)
            done = success_count + skip_count + fail_count
            print(f"\n--- Saved | done:{done}/{remaining} | ok:{success_count} skip:{skip_count} fail:{fail_count} | {rate:.1f}/min ---\n")

        time.sleep(DELAY_BETWEEN_CALLS)

    save_progress(progress)
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"DONE! Updated: {success_count} | Skipped: {skip_count} | Failed: {fail_count}")
    print(f"Time: {total_time/60:.1f} min")

    if failed:
        print(f"\n{len(failed)} files failed. Run again to retry.")


if __name__ == "__main__":
    run_updater()
