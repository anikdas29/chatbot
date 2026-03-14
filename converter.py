"""
Dataset Converter Tool
Online theke download kora dataset ke chatbot er format e convert kore.

Supports:
  1. CSV  (question,answer / multi-turn conversation format)
  2. JSON (various formats)
  3. JSONL (line-by-line JSON)
  4. TXT  (Q: ... A: ... format)

Usage:
  python converter.py input_file.csv
  python converter.py input_file.json
  python converter.py input_file.txt
"""

import csv
import json
import sys
import os
from collections import defaultdict


def read_csv_conversation(filepath):
    """
    Multi-turn conversation CSV format (Kaggle style):
      conversation_id, turn, role, intent, message

    User message = question, Bot message = answer, intent = category
    """
    pairs = []
    conversations = defaultdict(list)

    with open(filepath, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)

        delimiter = "\t" if "\t" in sample else (";" if ";" in sample else ",")
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = [h.lower().strip() for h in reader.fieldnames]

        # Detect if this is multi-turn conversation format
        is_conversation = "role" in headers and "message" in headers

        if not is_conversation:
            # Fall back to simple Q&A CSV
            f.seek(0)
            return read_csv_simple(filepath)

        print("  Detected: Multi-turn conversation format")

        for row in reader:
            conv_id = row.get("conversation_id", row.get("conv_id", ""))
            role = row.get("role", "").strip().lower()
            message = row.get("message", row.get("text", "")).strip()
            intent = row.get("intent", row.get("category", row.get("tag", ""))).strip()
            turn = row.get("turn", "0")

            if conv_id and message:
                conversations[conv_id].append({
                    "turn": float(turn) if turn else 0,
                    "role": role,
                    "message": message,
                    "intent": intent
                })

    # Extract user-bot pairs from conversations
    print(f"  Found: {len(conversations)} conversations")

    for conv_id, turns in conversations.items():
        turns.sort(key=lambda x: x["turn"])

        for i in range(len(turns) - 1):
            if turns[i]["role"] == "user" and turns[i + 1]["role"] == "bot":
                pairs.append({
                    "question": turns[i]["message"],
                    "answer": turns[i + 1]["message"],
                    "category": turns[i]["intent"] or turns[i + 1]["intent"] or None
                })

    return pairs


def read_csv_simple(filepath):
    """
    Simple CSV format:
      question,answer
      question,answer,category
    """
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)

        delimiter = "\t" if "\t" in sample else (";" if ";" in sample else ",")
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = [h.lower().strip() for h in reader.fieldnames]

        # Find question column
        q_col = None
        for h in reader.fieldnames:
            hl = h.lower().strip()
            if hl in ("question", "questions", "query", "text", "input", "utterance", "sentence", "prompt"):
                q_col = h
                break
        if not q_col:
            q_col = reader.fieldnames[0]

        # Find answer column
        a_col = None
        for h in reader.fieldnames:
            hl = h.lower().strip()
            if hl in ("answer", "answers", "response", "reply", "output", "bot_response", "bot"):
                a_col = h
                break
        if not a_col and len(reader.fieldnames) >= 2:
            a_col = reader.fieldnames[1]

        # Find category column
        c_col = None
        for h in reader.fieldnames:
            hl = h.lower().strip()
            if hl in ("category", "intent", "tag", "label", "class", "topic"):
                c_col = h
                break

        print(f"  Detected: Simple Q&A format")
        print(f"    Question: '{q_col}' | Answer: '{a_col}' | Category: '{c_col or 'auto'}'")

        for row in reader:
            q = row.get(q_col, "").strip()
            a = row.get(a_col, "").strip() if a_col else ""
            c = row.get(c_col, "").strip() if c_col else ""

            if q and a:
                pairs.append({
                    "question": q,
                    "answer": a,
                    "category": c if c else None
                })

    return pairs


def read_jsonl(filepath):
    """
    JSONL format - each line is a JSON object
    Supports same fields as conversation CSV
    """
    pairs = []
    conversations = defaultdict(list)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            conv_id = row.get("conversation_id", row.get("conv_id", ""))
            role = row.get("role", "").strip().lower()
            message = row.get("message", row.get("text", "")).strip()
            intent = row.get("intent", row.get("category", "")).strip()
            turn = row.get("turn", 0)

            if conv_id and message:
                conversations[conv_id].append({
                    "turn": float(turn) if turn else 0,
                    "role": role,
                    "message": message,
                    "intent": intent
                })

    print(f"  Found: {len(conversations)} conversations")

    for conv_id, turns in conversations.items():
        turns.sort(key=lambda x: x["turn"])
        for i in range(len(turns) - 1):
            if turns[i]["role"] == "user" and turns[i + 1]["role"] == "bot":
                pairs.append({
                    "question": turns[i]["message"],
                    "answer": turns[i + 1]["message"],
                    "category": turns[i]["intent"] or turns[i + 1]["intent"] or None
                })

    return pairs


def read_json(filepath):
    """
    JSON format support:
      - [{"question": "...", "answer": "..."}]
      - {"intents": [{"tag": "...", "patterns": [...], "responses": [...]}]}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        # Check if it's JSONL (multiple JSON objects, one per line)
        first_line = f.readline().strip()
        second_line = f.readline().strip()
        f.seek(0)

        if first_line.startswith("{") and second_line.startswith("{"):
            print("  Detected: JSONL format")
            return read_jsonl(filepath)

        data = json.load(f)

    pairs = []

    # Format: {"intents": [...]}
    if isinstance(data, dict) and "intents" in data:
        data = data["intents"]
        for item in data:
            tag = item.get("tag", item.get("intent", item.get("category", "")))
            patterns = item.get("patterns", item.get("utterances", item.get("questions", [])))
            responses = item.get("responses", item.get("answers", item.get("response", [])))

            if isinstance(responses, str):
                responses = [responses]
            answer = responses[0] if responses else ""

            for p in patterns:
                pairs.append({
                    "question": p.strip(),
                    "answer": answer.strip(),
                    "category": tag.strip() if tag else None
                })
        return pairs

    # Format: list of objects
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            q = ""
            for key in ["question", "questions", "query", "text", "input", "utterance", "prompt", "sentence", "patterns"]:
                if key in item:
                    val = item[key]
                    q = val[0] if isinstance(val, list) else str(val)
                    break

            a = ""
            for key in ["answer", "answers", "response", "reply", "output", "bot_response", "responses"]:
                if key in item:
                    val = item[key]
                    a = val[0] if isinstance(val, list) else str(val)
                    break

            c = ""
            for key in ["category", "intent", "tag", "label", "class", "topic"]:
                if key in item:
                    c = str(item[key])
                    break

            if q and a:
                pairs.append({
                    "question": q.strip(),
                    "answer": a.strip(),
                    "category": c.strip() if c else None
                })

    return pairs


def read_txt(filepath):
    """TXT format: Q: ... A: ..."""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_q = ""
    current_a = ""

    for line in lines:
        line = line.strip()
        if line.lower().startswith(("q:", "question:")):
            if current_q and current_a:
                pairs.append({"question": current_q, "answer": current_a, "category": None})
            current_q = line.split(":", 1)[1].strip()
            current_a = ""
        elif line.lower().startswith(("a:", "answer:")):
            current_a = line.split(":", 1)[1].strip()

    if current_q and current_a:
        pairs.append({"question": current_q, "answer": current_a, "category": None})

    return pairs


def auto_categorize(pairs):
    """Category na thakle auto generate kore"""
    counter = 0
    for pair in pairs:
        if not pair["category"]:
            words = pair["answer"].lower().split()[:3]
            pair["category"] = "_".join(words)[:20]
            counter += 1
    if counter > 0:
        print(f"  Auto-categorized {counter} entries")
    return pairs


def convert_to_chatbot_format(pairs):
    """Pairs ke chatbot er dataset.json format e convert kore"""
    # Group by category, collect unique questions, keep best answer
    groups = defaultdict(lambda: {"questions": set(), "answers": defaultdict(int)})

    for pair in pairs:
        key = pair["category"]
        groups[key]["questions"].add(pair["question"].lower().strip())
        groups[key]["answers"][pair["answer"]] += 1

    # Convert to list, keep all unique answers per category (sorted by frequency)
    dataset = []
    for category, data in groups.items():
        sorted_answers = sorted(data["answers"], key=data["answers"].get, reverse=True)
        dataset.append({
            "category": category,
            "questions": list(data["questions"]),
            "answers": sorted_answers
        })

    return dataset


def merge_with_existing(new_dataset, existing_path="dataset.json"):
    """Existing dataset er sathe merge kore"""
    existing = []
    if os.path.exists(existing_path):
        with open(existing_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing_categories = {item["category"]: i for i, item in enumerate(existing)}

    added = 0
    updated = 0

    for item in new_dataset:
        if item["category"] in existing_categories:
            idx = existing_categories[item["category"]]
            for q in item["questions"]:
                if q not in existing[idx]["questions"]:
                    existing[idx]["questions"].append(q)
                    updated += 1
            # Merge answers too
            new_answers = item.get("answers", [])
            existing_answers = existing[idx].get("answers", [existing[idx]["answer"]] if "answer" in existing[idx] else [])
            for a in new_answers:
                if a not in existing_answers:
                    existing_answers.append(a)
            existing[idx]["answers"] = existing_answers
        else:
            existing.append(item)
            added += 1

    return existing, added, updated


def convert(input_file, output_file="dataset.json", merge=True):
    """Main converter function"""
    ext = os.path.splitext(input_file)[1].lower()

    print(f"\nReading: {input_file}")
    print(f"File size: {os.path.getsize(input_file) / (1024*1024):.1f} MB")

    if ext in (".csv", ".tsv"):
        pairs = read_csv_conversation(input_file)
    elif ext in (".json", ".jsonl"):
        pairs = read_json(input_file)
    elif ext == ".txt":
        pairs = read_txt(input_file)
    else:
        print(f"Error: '{ext}' format support kori na. CSV, JSON, JSONL, TXT use koro.")
        return

    if not pairs:
        print("Error: Kono data paoa jai ni!")
        return

    print(f"  Extracted: {len(pairs)} question-answer pairs")

    pairs = auto_categorize(pairs)
    dataset = convert_to_chatbot_format(pairs)

    print(f"  Categories: {len(dataset)}")

    if merge and os.path.exists(output_file):
        dataset, added, updated = merge_with_existing(dataset, output_file)
        print(f"\n  Merged with existing dataset:")
        print(f"    New categories added: {added}")
        print(f"    Questions updated: {updated}")
        print(f"    Total categories: {len(dataset)}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    total_q = sum(len(item["questions"]) for item in dataset)
    print(f"\n  Saved to: {output_file}")
    print(f"  Total: {len(dataset)} categories, {total_q} questions")
    print("\n  Done! Ekhon 'python app.py' diye chatbot chalao.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 50)
        print("  Dataset Converter for Mini Chatbot")
        print("=" * 50)
        print("\nUsage: python converter.py <input_file> [output_file]")
        print("\nExamples:")
        print("  python converter.py chatbot_conversations.csv")
        print("  python converter.py chatbot_conversations.jsonl")
        print("  python converter.py faq_data.csv")
        print("  python converter.py intents.json")
        print("  python converter.py qa_pairs.txt")
        print("\nSupported formats: CSV, JSON, JSONL, TXT")
        print("Supports: Kaggle multi-turn conversation datasets")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "dataset.json"
        convert(input_file, output_file)
