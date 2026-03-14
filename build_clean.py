"""Build clean dataset from cached kaggle intents"""
import json

with open("_kaggle_intents.json", "r") as f:
    kaggle = json.load(f)

intent_answers = {
    "greeting": "Hello! How can I help you today?",
    "farewell": "Goodbye! Have a great day! Feel free to come back anytime.",
    "ai": "AI (Artificial Intelligence) is the simulation of human intelligence by machines. It includes machine learning, deep learning, NLP, and more.",
    "ml": "Machine Learning is a subset of AI where computers learn from data. Types: supervised, unsupervised, reinforcement learning. Tools: scikit-learn, TensorFlow, PyTorch.",
    "dl": "Deep Learning uses neural networks with many layers. CNNs for images, RNNs for sequences, Transformers for text. Frameworks: TensorFlow, PyTorch.",
    "coding": "Start with Python (beginner-friendly), then JavaScript, Java, or C++. Practice on LeetCode, HackerRank, or build real projects.",
    "coding_errors": "Debugging tips: Read error messages carefully, check syntax, use print statements, Google the error, check Stack Overflow.",
    "education": "For self-learning: Coursera, Udemy, Khan Academy. Practice consistently, build projects, join communities.",
    "technology": "Key tech areas: AI/ML, Cloud Computing, Cybersecurity, Blockchain, IoT. Stay updated through tech blogs.",
    "science": "Science branches: Physics, Chemistry, Biology, Astronomy. Science drives innovation and technology.",
    "math": "Key areas: Algebra, Calculus, Statistics, Probability, Linear Algebra. Practice regularly!",
    "finance": "Tips: Monthly budget, save 20% income, invest early, emergency fund, avoid debt, compound interest.",
    "health": "Tips: Exercise 30 min daily, 8 glasses water, balanced meals, 7-8 hours sleep, meditation.",
    "fitness": "Start with walks, bodyweight exercises (push-ups, squats, planks), stay consistent, track progress.",
    "food": "Breakfast: Oatmeal, eggs, fruits. Lunch: Rice with veggies. Snacks: Nuts, yogurt. Dinner: Light meals.",
    "sleep": "Tips: Consistent schedule, no screens before bed, dark cool room, no caffeine after 2 PM, 7-8 hours.",
    "motivation": "Every expert was once a beginner! Set small goals, celebrate wins, learn from failures. Progress not perfection!",
    "motivation_daily": "You are capable of amazing things! Take one step forward today. Consistency beats intensity!",
    "motivation_strong": "You didn't come this far to only come this far. Pain is temporary, quitting is forever. Push harder!",
    "quotes": "'The only way to do great work is to love what you do.' - Steve Jobs\n'Be the change you wish to see.' - Gandhi",
    "books": "Recommended: Atomic Habits, Think and Grow Rich, The Alchemist, Deep Work, Sapiens.",
    "music": "For focus: Lo-fi, Classical. For energy: Pop, Rock. For relaxation: Jazz, Acoustic.",
    "sports": "Popular: Cricket, Football, Basketball, Tennis. Stay updated on ESPN or Google Sports.",
    "gaming": "PC: Valorant, Minecraft, GTA V. Mobile: PUBG, Free Fire. Console: God of War, Zelda.",
    "entertainment": "Movies: IMDb top rated. Shows: Breaking Bad, Money Heist, Stranger Things.",
    "news": "Check Google News, BBC, CNN for updates. Stay informed but avoid overload!",
    "weather": "Check your local weather app or Google Weather for accurate updates.",
    "travel": "Plan ahead, book early, pack light, try local food, use Google Maps.",
    "shopping": "Compare prices, wait for sales, read reviews, set budget, check return policies.",
    "career": "Identify strengths, build skills, network, keep resume updated, practice interviews!",
    "interview": "Research company, practice questions, dress well, be confident, ask questions, follow up.",
    "resume": "Keep 1-2 pages, action verbs, quantify achievements, tailor for each job, proofread!",
    "salary": "Research rates on Glassdoor, know your worth, let employer offer first, negotiate with data.",
    "business": "Identify a problem, validate idea, start small (MVP), manage finances, build online presence.",
    "marketing": "SEO, Social Media, Content Marketing, Email Marketing, Analytics. Track and optimize!",
    "productivity": "Pomodoro (25 min work, 5 min break), prioritize tasks, minimize distractions, plan ahead.",
    "study": "Active recall, spaced repetition, teach others, Pomodoro, quiet environment, past papers.",
    "college": "Attend classes, build relationships, join clubs, build portfolio, network with seniors.",
    "projects": "Web: Portfolio, Blog. Python: Chatbot, Scraper. Mobile: Todo app. AI: Image classifier.",
    "habits": "Wake early, exercise daily, read 30 min, drink water, plan day, limit social media.",
    "emotions": "Its okay to feel. Talk to someone, journal, deep breathing, walk in nature. Seek help if needed.",
    "relationship": "Communication is key, listen actively, respect boundaries, be honest, support growth.",
    "communication": "Listen more, be clear, eye contact, practice public speaking.",
    "psychology": "Mindfulness, emotional intelligence, manage stress, sleep well. Therapy is strength.",
    "philosophy": "Big questions: meaning of life, consciousness, morality. Explore Socrates, Aristotle, Kant.",
    "history": "Ancient civilizations, World Wars, Industrial Revolution. Learn from the past!",
    "geography": "Largest country: Russia. Longest river: Nile. Highest peak: Everest. 195 countries.",
    "security": "Strong passwords, 2FA, avoid suspicious links, update software, VPN, backup data.",
    "cloud": "IaaS, PaaS, SaaS. Providers: AWS, Azure, GCP. Start with AWS Free Tier.",
    "networking": "IP Address, Router, DNS, HTTP/HTTPS. Learn: Cisco CCNA, CompTIA Network+.",
    "datascience": "Learn Python + SQL, Statistics, Visualization, ML. Tools: Jupyter, Pandas, NumPy.",
    "android": "Learn Kotlin/Java, Android Studio, Activities, Fragments, Material Design.",
    "ios": "Learn Swift, Xcode, UIKit/SwiftUI. Follow Apple Guidelines.",
    "life": "Focus on what you can control, be kind, keep learning, build relationships.",
    "general": "Keep asking questions - curiosity drives achievement!",
}

extra_questions = {
    "ai": ["what is ai", "tell me about ai", "artificial intelligence", "ai ki", "ai explain"],
    "ml": ["what is machine learning", "explain ml", "machine learning ki"],
    "dl": ["what is deep learning", "explain deep learning"],
    "greeting": ["how are you", "how are you doing", "whats up", "hey there", "hi there"],
    "farewell": ["take care", "see you tomorrow", "bye bye"],
    "coding": ["how to code", "learn programming", "best programming language"],
    "education": ["what is python", "how to study", "how to learn", "online courses"],
    "fitness": ["how to lose weight", "gym tips", "home workout"],
    "health": ["how to be healthy", "stay healthy"],
    "finance": ["how to invest", "save money", "how to save money"],
    "motivation": ["i need motivation", "feeling unmotivated", "inspire me"],
    "emotions": ["i feel sad", "i am depressed", "feeling lonely", "i am stressed"],
    "weather": ["weather today", "is it raining", "temperature today"],
    "sports": ["cricket score", "football news", "who won the match"],
    "books": ["suggest a book", "good books", "reading list"],
    "music": ["suggest songs", "best music", "playlist"],
    "news": ["latest news", "today news", "headlines"],
    "shopping": ["where to buy", "best deals", "online shopping"],
    "travel": ["where to visit", "best places", "holiday destination"],
    "sleep": ["cant sleep", "how to sleep fast", "insomnia"],
    "career": ["how to get job", "job tips"],
    "technology": ["latest tech", "tech news"],
}

dataset = []
for intent, questions in kaggle.items():
    answer = intent_answers.get(intent, f"I can help you with {intent}.")
    all_q = list(set(q.lower() for q in questions))
    if intent in extra_questions:
        for eq in extra_questions[intent]:
            if eq not in all_q:
                all_q.append(eq)
    dataset.append({"category": intent, "questions": all_q, "answers": [answer]})

# Custom
dataset.extend([
    {"category": "about_bot", "questions": ["tumi ke", "who are you", "what are you", "are you a bot", "apni ke"], "answers": ["Ami ekta NLP + ML powered chatbot. 55+ topic e answer dite pari!"]},
    {"category": "bot_name", "questions": ["tomar nam ki", "what is your name", "whats your name", "name bolo"], "answers": ["Amar nam Mini Bot!"]},
    {"category": "bot_capability", "questions": ["ki korte paro", "what can you do", "help", "help me", "features"], "answers": ["Ami 55+ topic e help korte pari: AI, ML, Coding, Education, Health, Fitness, Finance, Career, Motivation, Books, Music, Sports, Gaming, Travel, Weather ar onek kichu!"]},
    {"category": "thanks", "questions": ["thanks", "thank you", "dhonnobad", "tnx", "thx"], "answers": ["Welcome! Abar kichu lagleo jiggesh koro!"]},
])

with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

total_q = sum(len(i["questions"]) for i in dataset)
print(f"Categories: {len(dataset)}")
print(f"Total questions: {total_q}")
