# Chatbot Dataset Manager

## Project
This is a Mini Chatbot project with NLP + ML + Self Learning. Run with `python app.py` (Flask web UI on port 5000).

## Key Files
- `dataset.json` — Main dataset (categories, questions, answers)
- `CATEGORIES.md` — Category name list for quick reference
- `chatbot.py` — Bot engine (NLP, ML, feedback, learning)
- `app.py` — Flask API + Web UI
- `templates/index.html` — Chat UI
- `static/style.css` — Styles
- `feedback.json` — User feedback log

## Dataset Format
`dataset.json` is a JSON array. Each entry:
```json
{
    "category": "category_name",
    "questions": ["question 1", "question 2", "question 3", "question 4", "question 5"],
    "answers": ["answer 1", "answer 2", "answer 3", "answer 4"]
}
```

## Adding Data to Dataset
When the user says "add [topic/category]" or "add data about [topic]" or similar:

1. **Read `CATEGORIES.md`** to check if the category already exists
2. **Read `dataset.json`** to see the current data
3. **If category EXISTS**: Add more questions and/or answers to that existing category entry in `dataset.json`. Do NOT create a duplicate category.
4. **If category is NEW**: Create a new entry with:
   - `category`: lowercase, no spaces (use underscore), short name
   - `questions`: At least 5 varied questions users might ask (mix English + Bangla if relevant)
   - `answers`: At least 4 different answers (varied tone, different details, helpful)
5. **Update `CATEGORIES.md`** — add the new category name with incremented number
6. **Update `CATEGORIES.md` header** — update Total Categories count
7. After adding, show the user what was added (category, questions count, answers count)

### Rules for generating Q&A:
- Questions should be natural, like how a real user would type (short, informal, sometimes misspelled)
- Answers should be informative, varied, and 1-3 sentences each
- Each answer should give slightly different info or tone so the bot doesn't feel repetitive
- Mix English and Bangla (Banglish) where appropriate
- Keep answers factual and helpful
- Minimum 5 questions, minimum 4 answers per category
