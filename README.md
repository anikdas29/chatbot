# 🤖 Customer Support Chatbot

An AI-powered customer support chatbot built with Python and the Anthropic Claude API.

## Features

- 💬 Multi-turn conversation memory
- 🌐 REST API via Flask
- 🖥️ CLI mode for terminal use
- 🔒 Secure API key handling via `.env`

## Project Structure

```
customer-support-chatbot/
├── chatbot.py        # Core chatbot logic + CLI
├── app.py            # Flask REST API
├── requirements.txt  # Dependencies
├── .env.example      # Environment variable template
└── README.md
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-support-chatbot.git
cd customer-support-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

### CLI Mode
```bash
python chatbot.py
```

### API Mode
```bash
python app.py
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`      | API info    |
| GET    | `/health`| Health check|
| POST   | `/chat`  | Send message|
| POST   | `/reset` | Reset session|

#### Example Request
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your business hours?", "session_id": "user123"}'
```

#### Example Response
```json
{
  "session_id": "user123",
  "reply": "Our business hours are Monday to Friday, 9 AM to 6 PM. How can I help you today?",
  "turn": 1
}
```

## License

MIT
