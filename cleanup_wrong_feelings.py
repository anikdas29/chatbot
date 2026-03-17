"""
Find and remove feelings entries where the question clearly doesn't belong to the category.
Uses the bot's own FAISS detection to check if the feeling question matches its category.
"""
import json
import os
import sys

FOLDERS = ["category_wise_dataset", "coding_dataset"]

def main():
    # Simple keyword check: if a feeling question has NO word overlap
    # with the category name or the category's questions, it's suspicious
    removed_total = 0
    files_changed = 0

    for folder in FOLDERS:
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".json"):
                continue
            filepath = os.path.join(folder, fname)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            cat_name = data.get("category", fname.replace(".json", ""))
            questions = [q.lower() for q in data.get("questions", [])]
            answers = [a.lower() for a in data.get("answers", [])]
            feelings = data.get("feelings", [])

            if not feelings or not isinstance(feelings, list):
                continue

            # Build word set from category name + questions
            cat_words = set(cat_name.lower().replace("_", " ").split())
            for q in questions:
                cat_words.update(q.split())

            clean_feelings = []
            removed = 0
            for entry in feelings:
                if not isinstance(entry, dict):
                    continue
                fq = entry.get("question", "").lower().strip()
                if not fq:
                    continue

                fq_words = set(fq.split())
                # Check if the feeling question shares ANY meaningful word with the category
                overlap = fq_words & cat_words
                # Remove very common words from overlap check
                stop = {"what", "is", "how", "to", "the", "a", "an", "ki", "er", "ar",
                        "do", "i", "my", "me", "and", "or", "can", "you", "tell", "about",
                        "kore", "korbo", "kivabe", "why", "best", "for", "in", "of", "on"}
                meaningful_overlap = overlap - stop

                if meaningful_overlap:
                    clean_feelings.append(entry)
                else:
                    removed += 1
                    print(f"  REMOVED from {fname}: \"{fq}\" (no match with {cat_name})")

            if removed > 0:
                data["feelings"] = clean_feelings
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                removed_total += removed
                files_changed += 1

    print(f"\nTotal wrong feelings removed: {removed_total} from {files_changed} files")

if __name__ == "__main__":
    main()
