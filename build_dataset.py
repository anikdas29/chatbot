"""
Build clean dataset from Kaggle intents + original dataset
"""

import csv
import json
from collections import defaultdict

# ============================================
# Kaggle dataset - intents + questions only
# ============================================
kaggle_intents = defaultdict(set)
with open("chatbot_conversations.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["role"] == "user":
            kaggle_intents[row["intent"]].add(row["message"].strip().lower())

print(f"Kaggle: {len(kaggle_intents)} intents loaded")

# Proper answers for each intent
intent_answers = {
    "greeting": "Hello! How can I help you today?",
    "farewell": "Goodbye! Have a great day! Feel free to come back anytime.",
    "ai": "AI (Artificial Intelligence) is the simulation of human intelligence by machines. It includes machine learning, deep learning, NLP, and more. It's used in chatbots, self-driving cars, recommendation systems, and many other applications.",
    "ml": "Machine Learning is a subset of AI where computers learn from data without being explicitly programmed. Types include supervised learning, unsupervised learning, and reinforcement learning. Popular tools: scikit-learn, TensorFlow, PyTorch.",
    "dl": "Deep Learning uses neural networks with many layers to learn complex patterns. CNNs are used for images, RNNs/LSTMs for sequences, and Transformers for text. Popular frameworks: TensorFlow, PyTorch, Keras.",
    "coding": "Programming is the process of writing instructions for computers. Start with Python (beginner-friendly), then explore JavaScript, Java, or C++. Practice on LeetCode, HackerRank, or build real projects.",
    "coding_errors": "Common debugging tips:\n- Read the error message carefully\n- Check for typos and syntax errors\n- Use print statements to trace the issue\n- Google the error message\n- Check Stack Overflow\n- Use a debugger tool",
    "education": "Education is the key to success! For self-learning:\n- Use platforms like Coursera, Udemy, Khan Academy\n- Practice consistently\n- Build projects to apply knowledge\n- Join communities and study groups",
    "technology": "Technology is constantly evolving. Key areas:\n- AI & Machine Learning\n- Cloud Computing (AWS, Azure, GCP)\n- Cybersecurity\n- Blockchain\n- IoT (Internet of Things)\nStay updated through tech blogs and newsletters.",
    "science": "Science explains how the world works through observation and experimentation. Key branches: Physics, Chemistry, Biology, Astronomy, Earth Science. Science drives innovation and technology advancement.",
    "math": "Mathematics is the foundation of science and technology. Key areas:\n- Algebra & Calculus\n- Statistics & Probability\n- Linear Algebra\n- Discrete Mathematics\nPractice regularly and use visual tools to understand concepts.",
    "finance": "Financial tips:\n- Create a monthly budget\n- Save at least 20% of income\n- Start investing early (mutual funds, stocks)\n- Build an emergency fund\n- Avoid unnecessary debt\n- Learn about compound interest",
    "health": "Health tips for a better life:\n- Exercise 30 minutes daily\n- Drink 8 glasses of water\n- Eat balanced meals with fruits and vegetables\n- Get 7-8 hours of sleep\n- Manage stress through meditation\n- Regular health checkups",
    "fitness": "Fitness routine suggestions:\n- Start with 15-minute walks\n- Do bodyweight exercises (push-ups, squats, planks)\n- Stretch before and after workouts\n- Stay consistent, not intense\n- Track your progress\n- Rest days are important too!",
    "food": "Food suggestions:\n- Breakfast: Oatmeal, eggs, fruits\n- Lunch: Rice/roti with vegetables and protein\n- Snacks: Nuts, yogurt, fruits\n- Dinner: Light meals, salads, soup\nStay hydrated and eat mindfully!",
    "sleep": "Better sleep tips:\n- Maintain a consistent sleep schedule\n- Avoid screens 1 hour before bed\n- Keep room dark and cool\n- No caffeine after 2 PM\n- Try reading or meditation before sleep\n- Aim for 7-8 hours of sleep",
    "motivation": "Remember: Every expert was once a beginner. Keep going!\n- Set small, achievable goals\n- Celebrate small wins\n- Surround yourself with positive people\n- Learn from failures, don't fear them\n- Progress, not perfection!",
    "motivation_daily": "Today's motivation: You are capable of amazing things! Take one step forward today, no matter how small. Consistency beats intensity. You've got this!",
    "motivation_strong": "Listen up! You didn't come this far to only come this far. Pain is temporary, but quitting lasts forever. Get up, push harder, and prove everyone wrong. You're stronger than you think!",
    "quotes": "Here are some inspiring quotes:\n- 'The only way to do great work is to love what you do.' - Steve Jobs\n- 'Be the change you wish to see in the world.' - Gandhi\n- 'Success is not final, failure is not fatal.' - Churchill",
    "books": "Recommended books:\n- 'Atomic Habits' by James Clear\n- 'Think and Grow Rich' by Napoleon Hill\n- 'The Alchemist' by Paulo Coelho\n- 'Deep Work' by Cal Newport\n- 'Sapiens' by Yuval Noah Harari",
    "music": "Music recommendations:\n- For focus: Lo-fi, Classical, Ambient\n- For energy: Pop, Rock, Electronic\n- For relaxation: Jazz, Acoustic, Nature sounds\nExplore different genres on Spotify or YouTube Music!",
    "sports": "Sports keep you active and healthy! Popular sports:\n- Cricket, Football, Basketball, Tennis\n- Stay updated on scores through ESPN or Google Sports\n- Try playing a sport regularly for fitness and fun",
    "gaming": "Gaming recommendations:\n- PC: Valorant, Minecraft, GTA V\n- Mobile: PUBG, Free Fire, Among Us\n- Console: God of War, Zelda, FIFA\nGaming is fun but balance it with other activities!",
    "entertainment": "Entertainment suggestions:\n- Movies: Check IMDb top rated\n- TV Shows: Breaking Bad, Money Heist, Stranger Things\n- Music: Spotify playlists\n- Podcasts: Joe Rogan, Lex Fridman\nEnjoy but maintain balance!",
    "news": "For latest news, check:\n- Google News for quick updates\n- BBC, CNN for international news\n- Local news channels for regional updates\nStay informed but avoid news overload!",
    "weather": "For accurate weather updates, check your local weather app or Google Weather. General tips:\n- Carry an umbrella during monsoon\n- Stay hydrated in summer\n- Layer up in winter",
    "travel": "Travel tips:\n- Plan ahead and book early for better deals\n- Research your destination\n- Pack light but smart\n- Keep important documents safe\n- Try local food and culture\n- Use Google Maps for navigation",
    "shopping": "Smart shopping tips:\n- Compare prices across platforms\n- Wait for sales and festivals for big discounts\n- Read reviews before buying\n- Set a budget and stick to it\n- Check return policies",
    "career": "Career advice:\n- Identify your strengths and interests\n- Build skills through courses and projects\n- Network with professionals\n- Keep your resume updated\n- Practice for interviews\n- Never stop learning!",
    "interview": "Interview tips:\n- Research the company thoroughly\n- Practice common questions\n- Dress professionally\n- Be confident but humble\n- Ask thoughtful questions\n- Follow up with a thank you email",
    "resume": "Resume writing tips:\n- Keep it 1-2 pages max\n- Use action verbs (built, designed, managed)\n- Quantify achievements\n- Tailor it for each job\n- Include: Contact, Summary, Experience, Skills, Education\n- Proofread carefully!",
    "salary": "Salary negotiation tips:\n- Research market rates on Glassdoor/LinkedIn\n- Know your worth and skills\n- Let the employer make the first offer\n- Negotiate confidently with data\n- Consider total compensation (benefits, growth)",
    "business": "Starting a business:\n- Identify a problem to solve\n- Validate your idea with potential customers\n- Start small (MVP approach)\n- Manage finances carefully\n- Build an online presence\n- Stay adaptable and keep learning",
    "marketing": "Digital marketing basics:\n- SEO: Optimize for search engines\n- Social Media: Engage on Instagram, Facebook, LinkedIn\n- Content Marketing: Blog posts, videos\n- Email Marketing: Build an email list\n- Analytics: Track and optimize campaigns",
    "productivity": "Productivity tips:\n- Use the Pomodoro technique (25 min work, 5 min break)\n- Prioritize tasks (important vs urgent)\n- Minimize distractions\n- Batch similar tasks together\n- Take regular breaks\n- Plan your day the night before",
    "study": "Effective study tips:\n- Use active recall and spaced repetition\n- Teach what you learn to someone\n- Take breaks (Pomodoro method)\n- Minimize phone distractions\n- Study in a quiet environment\n- Practice with past papers",
    "college": "College tips:\n- Attend classes regularly\n- Build relationships with professors\n- Join clubs and societies\n- Start building your portfolio early\n- Balance academics and extracurriculars\n- Network with seniors and alumni",
    "projects": "Project ideas:\n- Web: Portfolio site, Blog, E-commerce\n- Python: Chatbot, Web scraper, Data analysis\n- Mobile: To-do app, Weather app, Chat app\n- AI/ML: Image classifier, Sentiment analyzer\nBuild and showcase on GitHub!",
    "habits": "Good habits to build:\n- Wake up early\n- Exercise daily\n- Read for 30 minutes\n- Drink enough water\n- Plan your day\n- Limit social media\n- Practice gratitude",
    "emotions": "It's okay to feel emotions. Tips:\n- Talk to someone you trust\n- Write in a journal\n- Practice deep breathing\n- Go for a walk in nature\n- Remember: bad days don't mean a bad life\n- Seek professional help if needed",
    "relationship": "Relationship tips:\n- Communication is key\n- Listen actively\n- Respect boundaries\n- Be honest and trustworthy\n- Support each other's growth\n- It's okay to disagree respectfully",
    "communication": "Communication tips:\n- Listen more than you speak\n- Be clear and concise\n- Make eye contact\n- Practice public speaking\n- Read books on communication\n- Join Toastmasters or debate clubs",
    "psychology": "Psychology insights:\n- Practice mindfulness for mental clarity\n- Understand cognitive biases\n- Build emotional intelligence\n- Manage stress through healthy habits\n- Sleep well for better mental health\n- Seek therapy when needed - it's a sign of strength",
    "philosophy": "Philosophy explores life's big questions:\n- What is the meaning of life?\n- What is consciousness?\n- What is morality?\nExplore thinkers like Socrates, Aristotle, Kant, and modern philosophers.",
    "history": "History helps us understand the present:\n- Ancient civilizations: Egypt, Greece, Rome\n- World Wars shaped modern geopolitics\n- Industrial Revolution transformed society\n- Learn from the past to build a better future",
    "geography": "Geography facts:\n- Largest country: Russia\n- Largest ocean: Pacific Ocean\n- Longest river: Nile (6,650 km)\n- Highest peak: Mount Everest (8,849m)\n- 7 continents, 195 countries",
    "security": "Cybersecurity tips:\n- Use strong, unique passwords\n- Enable two-factor authentication\n- Don't click suspicious links\n- Keep software updated\n- Use a VPN on public WiFi\n- Backup your data regularly",
    "cloud": "Cloud computing basics:\n- IaaS, PaaS, SaaS models\n- Major providers: AWS, Azure, Google Cloud\n- Benefits: Scalability, cost-effective, reliability\n- Learn: Start with AWS Free Tier",
    "networking": "Networking basics:\n- IP Address: Your device's identity on a network\n- Router: Connects devices to the internet\n- DNS: Translates domain names to IP addresses\n- HTTP/HTTPS: Web communication protocols",
    "datascience": "Data Science roadmap:\n- Learn Python and SQL\n- Statistics and Probability\n- Data visualization (Matplotlib, Seaborn)\n- Machine Learning (scikit-learn)\n- Tools: Jupyter Notebook, Pandas, NumPy\n- Practice on Kaggle datasets",
    "android": "Android development:\n- Learn Kotlin (recommended) or Java\n- Use Android Studio IDE\n- Understand Activities, Fragments, Intents\n- Material Design for UI\n- Resources: Android Developers official docs",
    "ios": "iOS development:\n- Learn Swift programming language\n- Use Xcode IDE\n- Understand UIKit and SwiftUI\n- Follow Apple Human Interface Guidelines\n- Resources: Apple Developer documentation",
    "life": "Life advice:\n- Focus on what you can control\n- Be kind to yourself and others\n- Keep learning and growing\n- Build meaningful relationships\n- Take care of your health\n- Find purpose in what you do",
    "general": "Here's something interesting: The human brain processes information at about 120 bits per second. Keep asking questions - curiosity is the engine of achievement!",
}

# Extra questions to add per intent for better matching
extra_questions = {
    "ai": ["what is ai", "what is ai?", "tell me about ai", "artificial intelligence", "what is artificial intelligence", "ai explain koro", "ai ki"],
    "ml": ["what is ml", "what is machine learning", "explain machine learning", "machine learning ki", "ml explain koro"],
    "dl": ["what is dl", "what is deep learning", "explain deep learning", "deep learning ki"],
    "greeting": ["how are you", "how are you doing", "whats up", "howdy", "hey there", "good day", "namaste"],
    "farewell": ["take care", "see you tomorrow", "good night", "bye bye"],
    "coding": ["how to code", "learn programming", "best programming language", "coding tips", "programming tips"],
    "education": ["what is python", "how to study", "how to learn", "online courses", "best courses"],
    "technology": ["what is ai?", "latest tech", "new technology", "tech news", "what is blockchain"],
    "fitness": ["how to lose weight", "gym tips", "home workout", "yoga tips"],
    "health": ["how to be healthy", "health tips", "stay healthy", "medical advice"],
    "food": ["what to eat", "healthy food", "diet plan", "breakfast ideas", "lunch ideas"],
    "finance": ["how to invest", "save money", "budget tips", "stock market", "crypto"],
    "career": ["how to get a job", "job tips", "career path", "switch career"],
    "motivation": ["i need motivation", "feeling unmotivated", "inspire me", "i feel low"],
    "emotions": ["i feel sad", "i am depressed", "feeling lonely", "i am stressed", "anxiety help"],
    "weather": ["weather today", "is it raining", "temperature today", "weather forecast"],
    "sports": ["cricket score", "football news", "who won the match", "sports news"],
    "gaming": ["best games 2024", "mobile games", "pc games", "gaming tips"],
    "books": ["suggest a book", "good books", "reading list", "novel suggestions"],
    "music": ["suggest songs", "best music", "playlist", "new songs"],
    "news": ["latest news", "today news", "what happened today", "headlines"],
    "shopping": ["where to buy", "best deals", "online shopping", "discount"],
    "travel": ["where to visit", "travel plans", "best places", "holiday destination"],
    "sleep": ["cant sleep", "insomnia help", "sleep better", "how to sleep fast"],
}

# ============================================
# Build clean dataset
# ============================================
dataset = []

for intent, questions in kaggle_intents.items():
    answer = intent_answers.get(intent, f"I can help you with {intent}. What would you like to know?")
    all_questions = list(questions)

    # Add extra questions
    if intent in extra_questions:
        for eq in extra_questions[intent]:
            if eq.lower() not in all_questions:
                all_questions.append(eq.lower())

    dataset.append({
        "category": intent,
        "questions": all_questions,
        "answer": answer
    })

# Add original custom categories (from your first dataset)
custom = [
    {
        "category": "about_bot",
        "questions": ["tumi ke", "tui ke", "apni ke", "who are you", "what are you", "tumi ki bot", "are you a bot", "are you human"],
        "answer": "Ami ekta NLP-powered chatbot. Ami tomader question er answer dei intelligent matching use kore. No AI API needed!"
    },
    {
        "category": "bot_name",
        "questions": ["tomar nam ki", "tor nam ki", "apnar nam ki", "what is your name", "whats your name", "name bolo"],
        "answer": "Amar nam Mini Bot! Ami NLP + Machine Learning use kore tomader question bujhi ar answer dei."
    },
    {
        "category": "bot_capability",
        "questions": ["ki korte paro", "ki help korbe", "what can you do", "ki ki jano", "features ki ki", "help", "help me"],
        "answer": "Ami 55+ topic e help korte pari:\n- AI, ML, Deep Learning\n- Coding, Education\n- Health, Fitness, Food\n- Finance, Career, Business\n- Motivation, Books, Music\n- Sports, Gaming, Entertainment\n- Travel, Shopping, Weather\nShudhu question likho!"
    },
    {
        "category": "thanks",
        "questions": ["thanks", "thank you", "dhonnobad", "tnx", "thx", "thanks a lot", "onek dhonnobad"],
        "answer": "Welcome! Abar kichu lagleo jiggesh koro. Tomader help korte pere khushi holam!"
    }
]

for item in custom:
    dataset.append(item)

# Save
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

total_q = sum(len(item["questions"]) for item in dataset)
print(f"\nFinal dataset:")
print(f"  Categories: {len(dataset)}")
print(f"  Total questions: {total_q}")
print(f"  Done!")
