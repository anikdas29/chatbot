"""
Dataset Cleanup — Remove duplicate questions & answers from all category JSON files.
Cleans both questions[] and feelings[] arrays.
Run: python cleanup_dataset.py
"""

import json
import os
import sys
from difflib import SequenceMatcher

FOLDERS = ["category_wise_dataset", "coding_dataset"]
SIMILARITY_THRESHOLD = 0.85  # Questions >85% similar = duplicate


def normalize(text):
    """Lowercase, strip, remove extra spaces and punctuation for comparison."""
    t = text.lower().strip()
    t = " ".join(t.split())  # collapse whitespace
    # Remove trailing punctuation for comparison
    while t and t[-1] in "?.!,;:":
        t = t[:-1]
    return t


def is_similar(a, b, threshold=SIMILARITY_THRESHOLD):
    """Check if two strings are similar above threshold."""
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def dedupe_list(items, threshold=SIMILARITY_THRESHOLD):
    """Remove duplicates and near-duplicates from a list of strings.
    Keeps the first occurrence (longest version if similar)."""
    if not items:
        return items, 0

    seen_normalized = {}  # normalized text -> original text
    unique = []
    removed = 0

    for item in items:
        norm = normalize(item)
        if not norm:
            continue

        # Check exact duplicate first
        if norm in seen_normalized:
            removed += 1
            continue

        # Check similar
        is_dup = False
        for seen_norm, seen_orig in seen_normalized.items():
            if SequenceMatcher(None, norm, seen_norm).ratio() >= threshold:
                # Keep the longer/better version
                if len(item.strip()) > len(seen_orig.strip()):
                    # Replace with longer version
                    idx = unique.index(seen_orig)
                    unique[idx] = item.strip()
                    del seen_normalized[seen_norm]
                    seen_normalized[normalize(item)] = item.strip()
                is_dup = True
                removed += 1
                break

        if not is_dup:
            unique.append(item.strip())
            seen_normalized[norm] = item.strip()

    return unique, removed


def dedupe_feelings(feelings, q_threshold=0.85, a_threshold=0.80):
    """Deduplicate feelings array (list of {question, answer, ...} dicts).
    Remove entries with duplicate question+answer combos."""
    if not feelings:
        return feelings, 0

    seen = []  # list of (norm_q, norm_a) tuples
    unique = []
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
                    removed += 1
                    break

        if not is_dup:
            unique.append(entry)
            seen.append((norm_q, norm_a))

    return unique, removed


def clean_file(filepath):
    """Clean a single category JSON file. Returns stats."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {"file": os.path.basename(filepath), "changes": False}

    # Dedupe questions array
    orig_q = data.get("questions", [])
    clean_q, q_removed = dedupe_list(orig_q)
    if q_removed > 0:
        data["questions"] = clean_q
        stats["questions_removed"] = q_removed
        stats["questions_before"] = len(orig_q)
        stats["questions_after"] = len(clean_q)
        stats["changes"] = True

    # Dedupe answers array
    orig_a = data.get("answers", [])
    clean_a, a_removed = dedupe_list(orig_a, threshold=0.80)
    if a_removed > 0:
        data["answers"] = clean_a
        stats["answers_removed"] = a_removed
        stats["answers_before"] = len(orig_a)
        stats["answers_after"] = len(clean_a)
        stats["changes"] = True

    # Dedupe feelings array
    orig_f = data.get("feelings", [])
    if orig_f and isinstance(orig_f, list) and len(orig_f) > 0:
        if isinstance(orig_f[0], dict):
            clean_f, f_removed = dedupe_feelings(orig_f)
            if f_removed > 0:
                data["feelings"] = clean_f
                stats["feelings_removed"] = f_removed
                stats["feelings_before"] = len(orig_f)
                stats["feelings_after"] = len(clean_f)
                stats["changes"] = True

    # Save if changed
    if stats["changes"]:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    return stats


def main():
    total_files = 0
    changed_files = 0
    total_q_removed = 0
    total_a_removed = 0
    total_f_removed = 0

    print("=" * 70)
    print("  DATASET CLEANUP — Removing duplicates")
    print("=" * 70)

    for folder in FOLDERS:
        if not os.path.isdir(folder):
            continue
        print(f"\n--- {folder}/ ---")

        files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
        for fname in files:
            filepath = os.path.join(folder, fname)
            total_files += 1

            stats = clean_file(filepath)

            if stats["changes"]:
                changed_files += 1
                parts = []
                if "questions_removed" in stats:
                    total_q_removed += stats["questions_removed"]
                    parts.append(f"Q: {stats['questions_before']}->{stats['questions_after']} (-{stats['questions_removed']})")
                if "answers_removed" in stats:
                    total_a_removed += stats["answers_removed"]
                    parts.append(f"A: {stats['answers_before']}->{stats['answers_after']} (-{stats['answers_removed']})")
                if "feelings_removed" in stats:
                    total_f_removed += stats["feelings_removed"]
                    parts.append(f"F: {stats['feelings_before']}->{stats['feelings_after']} (-{stats['feelings_removed']})")
                print(f"  CLEANED {fname}: {' | '.join(parts)}")

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"  Files scanned: {total_files}")
    print(f"  Files changed: {changed_files}")
    print(f"  Questions removed: {total_q_removed}")
    print(f"  Answers removed: {total_a_removed}")
    print(f"  Feelings removed: {total_f_removed}")
    print(f"  Total duplicates removed: {total_q_removed + total_a_removed + total_f_removed}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
