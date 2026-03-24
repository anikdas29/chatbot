"""
Dataset Cleanup — Remove duplicate questions, answers, and feelings from all category JSON files.
Cleans questions[], answers[], and feelings[] arrays.
Also removes misclassified feelings (wrong category).

Run: python cleanup_dataset.py
Dry run: python cleanup_dataset.py --dry-run
"""

import json
import os
import re
import sys
from difflib import SequenceMatcher

# Fix Unicode output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8',
                      errors='replace', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8',
                      errors='replace', buffering=1)

def _detect_folders():
    """Auto-detect dataset folders: ISP mode or general purpose mode."""
    if os.path.isdir("general") and os.path.isdir("isp_business"):
        return ["general", "isp_business"]
    folders = []
    for item in os.listdir("."):
        if os.path.isdir(item) and (item.endswith("_dataset") or item in ("general", "isp_business")):
            if any(f.endswith(".json") for f in os.listdir(item)):
                folders.append(item)
    return sorted(folders) if folders else ["category_wise_dataset"]

FOLDERS = _detect_folders()
QUESTION_SIMILARITY_THRESHOLD = 0.85  # Questions >85% similar = duplicate
ANSWER_SIMILARITY_THRESHOLD = 0.80    # Answers >80% similar = duplicate

STOP_WORDS = {
    "what", "is", "how", "to", "the", "a", "an", "ki", "er", "ar", "do",
    "i", "my", "me", "and", "or", "can", "you", "tell", "about", "kore",
    "korbo", "kivabe", "why", "best", "for", "in", "of", "on", "ta", "ke",
    "je", "na", "niye", "hoy", "hole", "theke", "diye", "kono", "amr",
    "tumi", "paro"
}


def strip_punctuation(text):
    """Remove ALL punctuation from text."""
    return re.sub(r'[^\w\s]', '', text)


def normalize(text):
    """Lowercase, strip, remove extra spaces and punctuation for comparison."""
    t = text.lower().strip()
    t = " ".join(t.split())  # collapse whitespace
    # Remove trailing punctuation for comparison
    while t and t[-1] in "?.!,;:":
        t = t[:-1]
    return t


def is_similar(a, b, threshold):
    """Check if two strings are similar above threshold."""
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def dedupe_list(items, threshold):
    """Remove duplicates and near-duplicates from a list of strings.
    Keeps the longer version when two items are similar."""
    if not items:
        return items, [], 0

    seen_normalized = {}  # normalized text -> original text
    unique = []
    removed_items = []
    removed = 0

    for item in items:
        norm = normalize(item)
        if not norm:
            continue

        # Check exact duplicate first
        if norm in seen_normalized:
            removed_items.append(item.strip())
            removed += 1
            continue

        # Check similar
        is_dup = False
        for seen_norm, seen_orig in list(seen_normalized.items()):
            if SequenceMatcher(None, norm, seen_norm).ratio() >= threshold:
                # Keep the longer/better version
                if len(item.strip()) > len(seen_orig.strip()):
                    # Replace with longer version
                    idx = unique.index(seen_orig)
                    removed_items.append(seen_orig)
                    unique[idx] = item.strip()
                    del seen_normalized[seen_norm]
                    seen_normalized[normalize(item)] = item.strip()
                else:
                    removed_items.append(item.strip())
                is_dup = True
                removed += 1
                break

        if not is_dup:
            unique.append(item.strip())
            seen_normalized[norm] = item.strip()

    return unique, removed_items, removed


def get_meaningful_words(text):
    """Extract meaningful (non-stop) words from text after stripping all punctuation."""
    cleaned = strip_punctuation(text.lower())
    words = cleaned.split()
    return {w for w in words if w and w not in STOP_WORDS}


def is_feeling_relevant(feeling_question, category_name, category_questions):
    """Check if a feeling's question has meaningful word overlap with the
    category's questions or category name. Returns True if relevant."""
    feeling_words = get_meaningful_words(feeling_question)
    if not feeling_words:
        return False

    # Build set of meaningful words from category name and all questions
    category_words = set()

    # Words from category name (split underscores)
    cat_name_clean = strip_punctuation(category_name.lower())
    for part in cat_name_clean.replace("_", " ").split():
        if part and part not in STOP_WORDS:
            category_words.add(part)

    # Words from all category questions
    for q in category_questions:
        category_words.update(get_meaningful_words(q))

    if not category_words:
        return True  # If no category words, can't determine, keep it

    # Check overlap
    overlap = feeling_words & category_words
    return len(overlap) > 0


def dedupe_feelings(feelings, q_threshold=0.85, a_threshold=0.80):
    """Deduplicate feelings array (list of {question, answer, ...} dicts).
    Remove entries with duplicate question+answer combos."""
    if not feelings:
        return feelings, [], 0

    seen = []  # list of (norm_q, norm_a) tuples
    unique = []
    removed_items = []
    removed = 0

    for entry in feelings:
        q = entry.get("question", "")
        a = entry.get("answer", "")
        norm_q = normalize(q)
        norm_a = normalize(a)

        if not norm_q:
            continue

        is_dup = False
        for seen_q, seen_a in seen:
            q_sim = SequenceMatcher(None, norm_q, seen_q).ratio()
            if q_sim >= q_threshold:
                # Same question — check if answer is also similar
                a_sim = SequenceMatcher(None, norm_a, seen_a).ratio()
                if a_sim >= a_threshold:
                    is_dup = True
                    removed_items.append(entry)
                    removed += 1
                    break

        if not is_dup:
            unique.append(entry)
            seen.append((norm_q, norm_a))

    return unique, removed_items, removed


def remove_wrong_feelings(feelings, category_name, category_questions):
    """Remove feelings whose question has no meaningful word overlap with the
    category's questions or category name (misclassified entries)."""
    if not feelings:
        return feelings, [], 0

    valid = []
    removed_items = []
    removed = 0

    for entry in feelings:
        q = entry.get("question", "")
        if is_feeling_relevant(q, category_name, category_questions):
            valid.append(entry)
        else:
            removed_items.append(entry)
            removed += 1

    return valid, removed_items, removed


def clean_file(filepath, dry_run=False):
    """Clean a single category JSON file. Returns stats and details."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {"file": os.path.basename(filepath), "changes": False, "details": []}
    category_name = data.get("category", "")

    # --- Dedupe questions array ---
    orig_q = data.get("questions", [])
    clean_q, removed_q_items, q_removed = dedupe_list(orig_q, QUESTION_SIMILARITY_THRESHOLD)
    if q_removed > 0:
        if not dry_run:
            data["questions"] = clean_q
        stats["questions_removed"] = q_removed
        stats["questions_before"] = len(orig_q)
        stats["questions_after"] = len(clean_q)
        stats["changes"] = True
        for item in removed_q_items:
            stats["details"].append(f"  [Q removed] {item}")

    # --- Dedupe answers array ---
    orig_a = data.get("answers", [])
    clean_a, removed_a_items, a_removed = dedupe_list(orig_a, ANSWER_SIMILARITY_THRESHOLD)
    if a_removed > 0:
        if not dry_run:
            data["answers"] = clean_a
        stats["answers_removed"] = a_removed
        stats["answers_before"] = len(orig_a)
        stats["answers_after"] = len(clean_a)
        stats["changes"] = True
        for item in removed_a_items:
            stats["details"].append(f"  [A removed] {item[:80]}...")

    # --- Clean feelings array ---
    orig_f = data.get("feelings", [])
    if orig_f and isinstance(orig_f, list) and len(orig_f) > 0 and isinstance(orig_f[0], dict):
        # Use the cleaned questions list for relevance checking (or original if dry_run)
        reference_questions = clean_q if not dry_run else orig_q

        # Step 1: Remove wrong-category feelings
        after_wrong, wrong_items, wrong_removed = remove_wrong_feelings(
            orig_f, category_name, reference_questions
        )
        if wrong_removed > 0:
            stats["feelings_wrong_removed"] = wrong_removed
            stats["changes"] = True
            for entry in wrong_items:
                stats["details"].append(
                    f"  [F wrong-category] Q: \"{entry.get('question', '')}\" "
                    f"(no overlap with '{category_name}')"
                )

        # Step 2: Dedupe remaining feelings
        clean_f, dup_f_items, f_dup_removed = dedupe_feelings(after_wrong)
        if f_dup_removed > 0:
            stats["feelings_dup_removed"] = f_dup_removed
            stats["changes"] = True
            for entry in dup_f_items:
                stats["details"].append(
                    f"  [F duplicate] Q: \"{entry.get('question', '')[:50]}\" "
                    f"A: \"{entry.get('answer', '')[:50]}...\""
                )

        total_f_removed = wrong_removed + f_dup_removed
        if total_f_removed > 0:
            stats["feelings_removed"] = total_f_removed
            stats["feelings_before"] = len(orig_f)
            stats["feelings_after"] = len(clean_f)
            if not dry_run:
                data["feelings"] = clean_f

    # Save if changed and not dry run
    if stats["changes"] and not dry_run:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    return stats


def main():
    dry_run = "--dry-run" in sys.argv

    total_files = 0
    changed_files = 0
    total_q_removed = 0
    total_a_removed = 0
    total_f_removed = 0
    total_f_wrong = 0
    total_f_dup = 0

    mode = "DRY RUN" if dry_run else "LIVE"
    print("=" * 70)
    print(f"  DATASET CLEANUP — Removing duplicates [{mode}]")
    print("=" * 70)
    if dry_run:
        print("  (No files will be modified)")

    for folder in FOLDERS:
        if not os.path.isdir(folder):
            continue
        print(f"\n--- {folder}/ ---")

        files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
        for fname in files:
            filepath = os.path.join(folder, fname)
            total_files += 1

            stats = clean_file(filepath, dry_run=dry_run)

            if stats["changes"]:
                changed_files += 1
                parts = []
                if "questions_removed" in stats:
                    total_q_removed += stats["questions_removed"]
                    parts.append(
                        f"Q: {stats['questions_before']}->{stats['questions_after']} "
                        f"(-{stats['questions_removed']})"
                    )
                if "answers_removed" in stats:
                    total_a_removed += stats["answers_removed"]
                    parts.append(
                        f"A: {stats['answers_before']}->{stats['answers_after']} "
                        f"(-{stats['answers_removed']})"
                    )
                if "feelings_removed" in stats:
                    total_f_removed += stats["feelings_removed"]
                    f_wrong = stats.get("feelings_wrong_removed", 0)
                    f_dup = stats.get("feelings_dup_removed", 0)
                    total_f_wrong += f_wrong
                    total_f_dup += f_dup
                    detail_parts = []
                    if f_wrong > 0:
                        detail_parts.append(f"{f_wrong} wrong-cat")
                    if f_dup > 0:
                        detail_parts.append(f"{f_dup} dups")
                    removal_detail = ", ".join(detail_parts) if detail_parts else ""
                    parts.append(
                        f"F: {stats['feelings_before']}->{stats['feelings_after']} "
                        f"(-{stats['feelings_removed']}"
                        f"{': ' + removal_detail if removal_detail else ''})"
                    )

                action = "WOULD CLEAN" if dry_run else "CLEANED"
                print(f"  {action} {fname}: {' | '.join(parts)}")

                # Print details in dry run or verbose
                if dry_run and stats["details"]:
                    for detail in stats["details"]:
                        print(detail)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY {'(DRY RUN)' if dry_run else ''}")
    print(f"  Files scanned:            {total_files}")
    print(f"  Files {'would change' if dry_run else 'changed'}:       "
          f"     {changed_files}")
    print(f"  Questions removed:        {total_q_removed}")
    print(f"  Answers removed:          {total_a_removed}")
    print(f"  Feelings removed (total): {total_f_removed}")
    print(f"    - Wrong category:       {total_f_wrong}")
    print(f"    - Duplicates:           {total_f_dup}")
    print(f"  Total items removed:      "
          f"{total_q_removed + total_a_removed + total_f_removed}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
