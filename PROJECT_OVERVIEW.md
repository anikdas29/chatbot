# Mini Chatbot - Full Codebase Overview

## Project Summary
NLP + ML powered self-learning chatbot with Flask web UI. No third-party AI API needed - shob kichu local e run hoy. User er question bujhe answer dey, ar notun jinish shikhte pare real-time e.

---

## Tech Stack
| Component       | Technology                          |
|-----------------|-------------------------------------|
| Backend         | Python 3.13, Flask 3.0+             |
| ML/NLP          | scikit-learn (TF-IDF, LogisticRegression) |
| Spell Correction| difflib (get_close_matches)         |
| Frontend        | HTML, CSS, Vanilla JavaScript       |
| Data Storage    | JSON (dataset.json)                 |
| Logging         | Python logging (chatbot.log)        |

---

## File Structure

```
chatbot/
├── app.py                  # Flask server - API endpoints + Web UI serve
├── chatbot.py              # Core chatbot engine - NLP, ML, Self-Learning
├── templates/
│   └── index.html          # Chat Web UI (HTML + inline JS)
├── static/
│   └── style.css           # Dark theme chat UI styling
├── dataset.json            # Main dataset - 56 categories, 250+ questions
├── requirements.txt        # Dependencies: flask, scikit-learn
├── build_dataset.py        # Dataset builder from Kaggle CSV + custom data
├── build_clean.py          # Clean dataset builder from cached kaggle intents
├── converter.py            # Universal dataset converter (CSV/JSON/JSONL/TXT)
├── _kaggle_intents.json    # Cached Kaggle intents (55 categories)
├── chatbot_conversations.csv  # Raw Kaggle conversation dataset
├── chatbot_conversations.json # Conversation data (JSON format)
├── Conversation.csv        # Additional conversation data
├── chatbot.log             # Runtime logs
└── README.md               # Old README (needs update)
```

---

## Core Files Breakdown

### 1. `app.py` (Flask Server)
- **Purpose:** Web server + REST API
- **Routes:**
  | Method | Endpoint      | Description                        |
  |--------|---------------|------------------------------------|
  | GET    | `/`           | Chat Web UI (index.html)           |
  | GET    | `/api/health` | Health check                       |
  | POST   | `/api/chat`   | Send message, get bot reply        |
  | POST   | `/api/learn`  | Teach bot new question-answer pair |
- **How it works:**
  - `/api/chat` - user message pathay, bot `get_answer()` call kore. Answer na janlbe `needs_learning: true` return kore.
  - `/api/learn` - question, category, answer niye bot ke shikhay. Bot retrain hoy instantly.
- **Run:** `python app.py` (port 5000, debug mode)

---

### 2. `chatbot.py` (Core Engine)
Main brain of the chatbot. 3 ta class ache:

#### Class: `SpellCorrector`
- Known words er sathe match kore spelling fix kore
- `difflib.get_close_matches` use kore (cutoff: 0.75)
- Example: "helo" -> "hello", "pyhton" -> "python"

#### Class: `SentimentDetector`
- Simple keyword-based sentiment detection
- 3 categories: **angry**, **happy**, **neutral**
- Angry words: "kharap", "baje", "worst", "bad", "error", etc.
- Happy words: "great", "awesome", "valo", "bhalo", etc.
- Response adjust kore sentiment onujayi:
  - Angry -> empathy message add kore
  - Happy -> appreciation message add kore

#### Class: `ChatBot` (Main Class)
- **`__init__`** - Dataset load, data prepare, spell corrector build, ML train, TF-IDF train
- **`_prepare_data()`** - Dataset theke questions, labels, category_answers extract kore
- **`_train_intent_classifier()`** - LogisticRegression + TF-IDF pipeline (ngram 1-2). Only trains jodi categories < 60% of total questions
- **`_train_tfidf()`** - TF-IDF vectorizer fit kore (cosine similarity er jonno)
- **`get_answer(user_question)`** - Main answer logic:
  1. Text clean (lowercase, remove punctuation)
  2. Spell correction
  3. Sentiment detect
  4. TF-IDF similarity check (primary)
  5. ML intent classification (if available)
  6. Decision: ML first (confidence >= 0.55), then TF-IDF (score >= 0.20), otherwise return None
  7. Sentiment-adjusted response return
- **`learn(question, category, answer)`** - Notun data add kore, dataset save kore, full retrain kore
- **`_tfidf_fallback()`** - Cosine similarity diye top 5 match dekhe, non-conversational intent prefer kore
- **`retrain()`** - Full model retrain (data prepare + spell corrector + classifier + tfidf)
- **CLI mode:** `run_chatbot()` function - terminal e chatbot chalano jay

---

### 3. `templates/index.html` (Chat UI)
- Dark theme chat interface
- Features:
  - Real-time message send/receive
  - Learning form - bot answer na janle user shikhate pare (category + answer input)
  - Skip option for learning
  - Enter key support
  - Auto-scroll to latest message
- JavaScript handles:
  - `/api/chat` POST call
  - `/api/learn` POST call (when teaching)
  - Dynamic message bubble creation
  - Learn form show/hide toggle

---

### 4. `static/style.css` (Styling)
- Dark theme: `#0f0f23` background, `#1a1a2e` chat container
- Purple gradient header & user bubbles (`#667eea` -> `#764ba2`)
- Orange gradient learn button (`#ffa726` -> `#ff7043`)
- 420x650px fixed chat container
- Custom scrollbar styling
- Responsive message bubbles (max-width: 80%)

---

### 5. `converter.py` (Dataset Converter Tool)
- Universal converter - online dataset ke chatbot format e convert kore
- **Supported formats:**
  | Format | Detection                                     |
  |--------|-----------------------------------------------|
  | CSV    | Simple Q&A or Multi-turn conversation (Kaggle)|
  | JSON   | Flat list or `{"intents": [...]}` format       |
  | JSONL  | Line-by-line JSON objects                      |
  | TXT    | `Q: ... A: ...` format                         |
- **Auto-detection:** Column names automatically detect kore (question/answer/category er different naming conventions)
- **Auto-categorize:** Category na thakle answer er first 3 words diye category generate kore
- **Merge support:** Existing dataset er sathe merge korte pare
- **Usage:** `python converter.py input_file.csv [output_file.json]`

---

### 6. `build_dataset.py` (Full Dataset Builder)
- Kaggle `chatbot_conversations.csv` theke intents extract kore
- 55 ta intent er jonno manually written answers ache
- Extra questions add kore per intent (better matching er jonno)
- 4 ta custom Bangla categories add kore: about_bot, bot_name, bot_capability, thanks
- Output: `dataset.json`

---

### 7. `build_clean.py` (Clean Dataset Builder)
- `_kaggle_intents.json` (cached) theke dataset build kore
- Same intent_answers ar extra_questions as build_dataset.py (shorter answers)
- Faster - CSV parse korte hoy na

---

### 8. `dataset.json` (Main Dataset)
- **56 categories**, **250+ questions**
- Format per entry:
  ```json
  {
      "category": "greeting",
      "questions": ["hello", "hi", "how are you", ...],
      "answer": "Hello! How can I help you today?"
  }
  ```
- Categories include: AI, ML, DL, coding, education, health, fitness, food, finance, career, motivation, books, music, sports, gaming, entertainment, travel, weather, sleep, emotions, etc.
- 4 custom Bangla bot categories: about_bot, bot_name, bot_capability, thanks

---

## How It Works (Flow)

```
User types message
       |
       v
  [app.py] /api/chat receives message
       |
       v
  [chatbot.py] ChatBot.get_answer()
       |
       ├── 1. Clean text (lowercase, remove punctuation)
       ├── 2. Spell correction (difflib)
       ├── 3. Sentiment detection (keyword matching)
       ├── 4. TF-IDF cosine similarity (primary matching)
       ├── 5. ML intent classification (LogisticRegression)
       └── 6. Decision:
            ├── ML confidence >= 0.55 → use ML prediction
            ├── TF-IDF score >= 0.20 → use TF-IDF match
            └── Neither → return None (needs_learning)
       |
       v
  Answer found? ──Yes──> Adjust for sentiment → Return reply
       |
       No
       v
  Frontend shows learning form
       |
       v
  User teaches: category + answer
       |
       v
  [app.py] /api/learn → ChatBot.learn()
       |
       ├── Add to dataset (existing or new category)
       ├── Save dataset.json
       └── Full retrain (data + spell + classifier + tfidf)
```

---

## Dependencies
```
flask>=3.0.0
scikit-learn>=1.4.0
```
Built-in modules used: `json`, `logging`, `re`, `difflib`, `csv`, `os`, `sys`, `collections`

---

## Run Instructions
```bash
# Install dependencies
pip install -r requirements.txt

# Run web UI (port 5000)
python app.py

# Run CLI mode
python chatbot.py

# Convert external dataset
python converter.py input_file.csv

# Rebuild dataset from Kaggle CSV
python build_dataset.py

# Rebuild dataset from cached intents
python build_clean.py
```

---

## Key Design Decisions
1. **No external AI API** - Shob kichu local ML (TF-IDF + LogisticRegression) diye hoy
2. **Self-learning** - Bot real-time e notun jinish shikhte pare, dataset.json e save hoy
3. **Dual matching** - ML classifier + TF-IDF fallback (confidence thresholds diye decide kore)
4. **Spell correction** - User er typo handle kore automatically
5. **Sentiment awareness** - Angry/happy user ke different tone e respond kore
6. **Auto retrain** - Prottek learn er por full model retrain hoy
7. **Bangla + English** - Mixed language support (Banglish responses)
