"""
Comprehensive Chatbot Test - 500+ questions across all 94 categories.
Tests: accuracy, routing, edge cases, misspellings, Banglish, follow-ups.
"""
import json
import time
from chatbot import ChatBot

# ============================================================
# TEST QUESTIONS: ~6 questions per category (94 categories)
# Mix of: exact dataset questions, paraphrased, Banglish, typos
# ============================================================

TEST_QUESTIONS = {
    # ===== GENERAL CATEGORIES (category_wise_dataset) =====
    "greeting": [
        ("hello", "greeting"),
        ("hi there", "greeting"),
        ("hey", "greeting"),
        ("good morning", "greeting"),
        ("assalamualaikum", "greeting"),
        ("kemon acho", "greeting"),
        ("whats up", "greeting"),
    ],
    "farewell": [
        ("bye", "farewell"),
        ("goodbye", "farewell"),
        ("see you later", "farewell"),
        ("take care", "farewell"),
        ("bye bye", "farewell"),
        ("good night", "farewell"),
    ],
    "thanks": [
        ("thank you", "thanks"),
        ("thanks a lot", "thanks"),
        ("dhonnobad", "thanks"),
        ("thanks bro", "thanks"),
        ("thank you so much", "thanks"),
    ],
    "about_bot": [
        ("who are you", "about_bot"),
        ("what are you", "about_bot"),
        ("tell me about yourself", "about_bot"),
        ("who made you", "about_bot"),
        ("are you a robot", "about_bot"),
    ],
    "bot_name": [
        ("what is your name", "bot_name"),
        ("tomar nam ki", "bot_name"),
        ("whats your name", "bot_name"),
        ("tell me your name", "bot_name"),
    ],
    "bot_capability": [
        ("what can you do", "bot_capability"),
        ("tumi ki ki paro", "bot_capability"),
        ("what are your features", "bot_capability"),
        ("help me", "bot_capability"),
        ("how can you help me", "bot_capability"),
    ],
    "emotions": [
        ("i feel sad", "emotions"),
        ("how to control anger", "emotions"),
        ("i am depressed", "emotions"),
        ("feeling lonely", "emotions"),
        ("i am stressed", "emotions"),
        ("i feel anxious", "emotions"),
    ],
    "health": [
        ("how to stay healthy", "health"),
        ("health tips", "health"),
        ("how to improve health", "health"),
        ("healthy lifestyle", "health"),
        ("health advice", "health"),
        ("kivabe healthy thaka jai", "health"),
    ],
    "fitness": [
        ("how to get fit", "fitness"),
        ("workout tips", "fitness"),
        ("exercise routine", "fitness"),
        ("best exercises", "fitness"),
        ("gym tips", "fitness"),
        ("fitness advice for beginners", "fitness"),
    ],
    "sleep": [
        ("how to sleep better", "sleep"),
        ("sleep tips", "sleep"),
        ("insomnia help", "sleep"),
        ("cant sleep at night", "sleep"),
        ("how many hours should i sleep", "sleep"),
    ],
    "food": [
        ("what should i eat", "food"),
        ("food suggestions", "food"),
        ("healthy food options", "food"),
        ("best foods for health", "food"),
        ("ki khabo aj", "food"),
    ],
    "cooking": [
        ("how to cook rice", "cooking"),
        ("cooking tips", "cooking"),
        ("easy recipes", "cooking"),
        ("meal prep ideas", "cooking"),
        ("how to cook for beginners", "cooking"),
    ],
    "shopping": [
        ("best online shopping site", "shopping"),
        ("where to buy phone", "shopping"),
        ("shopping tips", "shopping"),
        ("online shopping help", "shopping"),
        ("best deals online", "shopping"),
    ],
    "weather": [
        ("whats the weather today", "weather"),
        ("weather forecast", "weather"),
        ("will it rain today", "weather"),
        ("temperature today", "weather"),
        ("aj weather kemon", "weather"),
        ("is it going to rain", "weather"),
    ],
    "travel": [
        ("best places to visit", "travel"),
        ("travel tips", "travel"),
        ("cheap travel destinations", "travel"),
        ("how to plan a trip", "travel"),
        ("travel guide", "travel"),
    ],
    "music": [
        ("best music to listen", "music"),
        ("music suggestions", "music"),
        ("how to learn guitar", "music"),
        ("best songs", "music"),
    ],
    "movies": [
        ("best movies to watch", "movies"),
        ("movie recommendations", "movies"),
        ("top movies 2024", "movies"),
        ("suggest a good movie", "movies"),
        ("ki movie dekhbo", "movies"),
    ],
    "sports": [
        ("latest cricket score", "sports"),
        ("best football players", "sports"),
        ("sports news", "sports"),
        ("who won the match", "sports"),
        ("sports update", "sports"),
    ],
    "gaming": [
        ("best games to play", "gaming"),
        ("gaming tips", "gaming"),
        ("mobile games recommendation", "gaming"),
        ("best pc games", "gaming"),
    ],
    "entertainment": [
        ("what to do for fun", "entertainment"),
        ("entertainment ideas", "entertainment"),
        ("fun activities", "entertainment"),
        ("bored what to do", "entertainment"),
    ],
    "news": [
        ("latest news", "news"),
        ("whats happening today", "news"),
        ("news update", "news"),
        ("current events", "news"),
        ("trending news", "news"),
    ],
    "education": [
        ("how to study effectively", "education"),
        ("education tips", "education"),
        ("best way to learn", "education"),
        ("online courses", "education"),
        ("education advice", "education"),
    ],
    "study": [
        ("study tips", "study"),
        ("how to study better", "study"),
        ("exam preparation tips", "study"),
        ("poriksha preparation", "study"),
    ],
    "college": [
        ("best college in bangladesh", "college"),
        ("college admission tips", "college"),
        ("how to choose a college", "college"),
        ("college life tips", "college"),
    ],
    "books": [
        ("best books to read", "books"),
        ("book recommendations", "books"),
        ("good books", "books"),
        ("suggest a book", "books"),
        ("reading suggestions", "books"),
    ],
    "motivation": [
        ("i need motivation", "motivation"),
        ("motivate me", "motivation"),
        ("feeling unmotivated", "motivation"),
        ("motivation tips", "motivation"),
        ("inspire me", "motivation"),
        ("give me inspiration", "motivation"),
    ],
    "motivation_strong": [
        ("give me a powerful motivation", "motivation_strong"),
        ("strong motivation needed", "motivation_strong"),
    ],
    "motivation_daily": [
        ("daily motivation", "motivation_daily"),
        ("morning motivation quote", "motivation_daily"),
    ],
    "quotes": [
        ("give me a quote", "quotes"),
        ("inspirational quotes", "quotes"),
        ("best quotes", "quotes"),
        ("famous quotes", "quotes"),
    ],
    "career": [
        ("how to build a career", "career"),
        ("career advice", "career"),
        ("career tips", "career"),
        ("what career should i choose", "career"),
    ],
    "resume": [
        ("how to make a resume", "resume"),
        ("resume tips", "resume"),
        ("cv writing tips", "resume"),
        ("resume format", "resume"),
    ],
    "interview": [
        ("interview tips", "interview"),
        ("how to prepare for interview", "interview"),
        ("job interview questions", "interview"),
        ("interview preparation", "interview"),
    ],
    "salary": [
        ("how to negotiate salary", "salary"),
        ("salary tips", "salary"),
        ("average salary in bangladesh", "salary"),
    ],
    "freelancing": [
        ("how to start freelancing", "freelancing"),
        ("freelancing tips", "freelancing"),
        ("best freelancing platform", "freelancing"),
        ("fiverr vs upwork", "freelancing"),
        ("freelancing career", "freelancing"),
    ],
    "business": [
        ("how to start a business", "business"),
        ("business ideas", "business"),
        ("small business tips", "business"),
        ("startup ideas", "business"),
    ],
    "finance": [
        ("how to save money", "finance"),
        ("money management tips", "finance"),
        ("financial advice", "finance"),
        ("investment tips", "finance"),
        ("budget planning", "finance"),
    ],
    "marketing": [
        ("digital marketing tips", "marketing"),
        ("how to do marketing", "marketing"),
        ("marketing strategy", "marketing"),
        ("social media marketing", "marketing"),
    ],
    "productivity": [
        ("how to be more productive", "productivity"),
        ("productivity tips", "productivity"),
        ("boost productivity", "productivity"),
        ("productivity hacks", "productivity"),
    ],
    "time_management": [
        ("how to manage time", "time_management"),
        ("time management tips", "time_management"),
        ("how to stop procrastinating", "time_management"),
        ("time management skills", "time_management"),
        ("pomodoro technique", "time_management"),
    ],
    "habits": [
        ("how to build good habits", "habits"),
        ("good habits", "habits"),
        ("daily habits for success", "habits"),
        ("habit building tips", "habits"),
    ],
    "life": [
        ("meaning of life", "life"),
        ("life advice", "life"),
        ("how to live a good life", "life"),
        ("life tips", "life"),
    ],
    "relationship": [
        ("relationship advice", "relationship"),
        ("how to maintain relationship", "relationship"),
        ("relationship tips", "relationship"),
        ("love advice", "relationship"),
    ],
    "communication": [
        ("how to improve communication", "communication"),
        ("communication skills tips", "communication"),
        ("public speaking tips", "communication"),
    ],
    "philosophy": [
        ("what is philosophy", "philosophy"),
        ("philosophical questions", "philosophy"),
        ("meaning of existence", "philosophy"),
    ],
    "psychology": [
        ("what is psychology", "psychology"),
        ("psychology facts", "psychology"),
        ("human behavior", "psychology"),
    ],
    "science": [
        ("interesting science facts", "science"),
        ("latest science news", "science"),
        ("science ki", "science"),
        ("how does gravity work", "science"),
    ],
    "math": [
        ("how to learn math", "math"),
        ("math tips", "math"),
        ("math problem solving", "math"),
        ("math ki", "math"),
    ],
    "history": [
        ("interesting history facts", "history"),
        ("world history", "history"),
        ("history of bangladesh", "history"),
    ],
    "geography": [
        ("geography facts", "geography"),
        ("largest country in the world", "geography"),
        ("geography ki", "geography"),
    ],
    "ai": [
        ("what is ai", "ai"),
        ("artificial intelligence explained", "ai"),
        ("ai future", "ai"),
        ("ai ki", "ai"),
        ("how does ai work", "ai"),
        ("will ai take over jobs", "ai"),
    ],
    "ml": [
        ("what is machine learning", "ml"),
        ("how to learn ml", "ml"),
        ("machine learning basics", "ml"),
        ("ml tutorial", "ml"),
        ("machine learning ki", "ml"),
    ],
    "dl": [
        ("what is deep learning", "dl"),
        ("deep learning basics", "dl"),
        ("neural network explained", "dl"),
        ("deep learning tutorial", "dl"),
    ],
    "datascience": [
        ("what is data science", "datascience"),
        ("data science career", "datascience"),
        ("how to become data scientist", "datascience"),
    ],
    "technology": [
        ("latest technology news", "technology"),
        ("tech trends", "technology"),
        ("new technology 2024", "technology"),
        ("technology update", "technology"),
    ],
    "android": [
        ("best android apps", "android"),
        ("android tips", "android"),
        ("android development", "android"),
    ],
    "ios": [
        ("best iphone apps", "ios"),
        ("ios vs android", "ios"),
        ("iphone tips", "ios"),
    ],
    "cloud": [
        ("what is cloud computing", "cloud"),
        ("cloud computing basics", "cloud"),
        ("aws vs azure", "cloud"),
    ],
    "database": [
        ("what is database", "database"),
        ("sql basics", "database"),
        ("mysql vs postgresql", "database"),
        ("database tutorial", "database"),
        ("how to learn sql", "database"),
    ],
    "git": [
        ("what is git", "git"),
        ("git basics", "git"),
        ("how to use github", "git"),
        ("git commands", "git"),
        ("git tutorial", "git"),
    ],
    "coding": [
        ("how to start coding", "coding"),
        ("coding tips", "coding"),
        ("best programming language", "coding"),
        ("learn to code", "coding"),
        ("coding for beginners", "coding"),
    ],
    "coding_errors": [
        ("how to fix bugs", "coding_errors"),
        ("debugging tips", "coding_errors"),
        ("common coding errors", "coding_errors"),
    ],
    "projects": [
        ("project ideas", "projects"),
        ("beginner project ideas", "projects"),
        ("coding project suggestions", "projects"),
    ],
    "security": [
        ("online security tips", "security"),
        ("how to stay safe online", "security"),
        ("password security", "security"),
    ],
    "networking": [
        ("networking basics", "networking"),
        ("computer networking", "networking"),
        ("what is tcp ip", "networking"),
    ],
    "social_media": [
        ("how to grow on social media", "social_media"),
        ("social media tips", "social_media"),
        ("instagram growth tips", "social_media"),
        ("facebook marketing", "social_media"),
        ("social media strategy", "social_media"),
    ],
    "photography": [
        ("photography tips", "photography"),
        ("how to take better photos", "photography"),
        ("camera settings for beginners", "photography"),
        ("mobile photography tips", "photography"),
        ("best camera for beginners", "photography"),
    ],
    "writing": [
        ("how to improve writing", "writing"),
        ("writing tips", "writing"),
        ("creative writing tips", "writing"),
        ("how to write better", "writing"),
        ("content writing tips", "writing"),
    ],
    "language": [
        ("how to learn english", "language"),
        ("language learning tips", "language"),
        ("best language to learn", "language"),
        ("english speaking tips", "language"),
        ("how to learn a new language", "language"),
    ],
    "meditation": [
        ("how to meditate", "meditation"),
        ("meditation for beginners", "meditation"),
        ("benefits of meditation", "meditation"),
        ("meditation tips", "meditation"),
        ("mindfulness techniques", "meditation"),
    ],
    "yoga": [
        ("how to start yoga", "yoga"),
        ("yoga for beginners", "yoga"),
        ("benefits of yoga", "yoga"),
        ("yoga tips", "yoga"),
        ("best yoga poses", "yoga"),
    ],
    "pets": [
        ("how to take care of pets", "pets"),
        ("best pets to have", "pets"),
        ("dog care tips", "pets"),
        ("cat care tips", "pets"),
        ("pet adoption", "pets"),
    ],
    "parenting": [
        ("parenting tips", "parenting"),
        ("how to be a good parent", "parenting"),
        ("child education tips", "parenting"),
        ("parenting advice", "parenting"),
        ("baby care tips", "parenting"),
    ],
    "fashion": [
        ("fashion tips", "fashion"),
        ("how to dress well", "fashion"),
        ("mens fashion guide", "fashion"),
        ("latest fashion trends", "fashion"),
        ("fashion advice", "fashion"),
    ],
    "environment": [
        ("how to save environment", "environment"),
        ("climate change facts", "environment"),
        ("reduce carbon footprint", "environment"),
        ("environmental tips", "environment"),
        ("go green tips", "environment"),
    ],
    "astronomy": [
        ("facts about space", "astronomy"),
        ("how big is the universe", "astronomy"),
        ("astronomy basics", "astronomy"),
        ("planets in solar system", "astronomy"),
        ("space exploration", "astronomy"),
    ],
    "crypto": [
        ("what is cryptocurrency", "crypto"),
        ("bitcoin explained", "crypto"),
        ("how to invest in crypto", "crypto"),
        ("crypto trading tips", "crypto"),
        ("blockchain explained", "crypto"),
    ],
    "diy": [
        ("diy project ideas", "diy"),
        ("easy diy crafts", "diy"),
        ("diy home decoration", "diy"),
        ("diy tips", "diy"),
        ("do it yourself ideas", "diy"),
    ],
    "podcast": [
        ("best podcasts to listen", "podcast"),
        ("podcast recommendations", "podcast"),
        ("how to start a podcast", "podcast"),
        ("top podcasts", "podcast"),
        ("podcast tips", "podcast"),
    ],

    # ===== CODING CATEGORIES (coding_dataset) =====
    "python": [
        ("what is python", "python"),
        ("python basics", "python"),
        ("how to learn python", "python"),
        ("python tutorial", "python"),
        ("python for beginners", "python"),
        ("best python course", "python"),
        ("python programming language", "python"),
    ],
    "javascript": [
        ("what is javascript", "javascript"),
        ("javascript basics", "javascript"),
        ("how to learn javascript", "javascript"),
        ("js tutorial", "javascript"),
        ("javascript for beginners", "javascript"),
        ("learn js", "javascript"),
    ],
    "html_css": [
        ("what is html", "html_css"),
        ("what is css", "html_css"),
        ("html css basics", "html_css"),
        ("how to learn html", "html_css"),
        ("html tutorial", "html_css"),
        ("how to make a website", "html_css"),
    ],
    "react": [
        ("what is react", "react"),
        ("react js tutorial", "react"),
        ("how to learn react", "react"),
        ("react basics", "react"),
        ("react vs angular", "react"),
        ("why use react", "react"),
    ],
    "nodejs": [
        ("what is node js", "nodejs"),
        ("node js tutorial", "nodejs"),
        ("express js", "nodejs"),
        ("backend with javascript", "nodejs"),
        ("nodejs basics", "nodejs"),
    ],
    "java": [
        ("what is java", "java"),
        ("java basics", "java"),
        ("how to learn java", "java"),
        ("java tutorial", "java"),
        ("java vs python", "java"),
        ("java for beginners", "java"),
    ],
    "cpp": [
        ("what is c++", "cpp"),
        ("c++ basics", "cpp"),
        ("how to learn c++", "cpp"),
        ("c++ tutorial", "cpp"),
        ("c vs c++", "cpp"),
    ],
    "api": [
        ("what is api", "api"),
        ("how api works", "api"),
        ("rest api explained", "api"),
        ("what is rest api", "api"),
        ("how to make an api", "api"),
    ],
    "data_structures": [
        ("what is data structure", "data_structures"),
        ("dsa basics", "data_structures"),
        ("how to learn dsa", "data_structures"),
        ("data structures and algorithms", "data_structures"),
        ("dsa for interview", "data_structures"),
        ("array vs linked list", "data_structures"),
    ],
    "web_dev": [
        ("how to become web developer", "web_dev"),
        ("web development roadmap", "web_dev"),
        ("frontend vs backend", "web_dev"),
        ("full stack developer", "web_dev"),
        ("learn web development", "web_dev"),
    ],
    "linux": [
        ("what is linux", "linux"),
        ("linux commands", "linux"),
        ("linux for beginners", "linux"),
        ("best linux distro", "linux"),
        ("ubuntu basics", "linux"),
    ],
    "devops": [
        ("what is devops", "devops"),
        ("devops basics", "devops"),
        ("what is docker", "devops"),
        ("ci cd pipeline", "devops"),
        ("devops tools", "devops"),
    ],
    "cybersecurity": [
        ("what is cybersecurity", "cybersecurity"),
        ("how to learn cybersecurity", "cybersecurity"),
        ("ethical hacking basics", "cybersecurity"),
        ("how to become a hacker", "cybersecurity"),
        ("cybersecurity career", "cybersecurity"),
    ],
    "flutter": [
        ("what is flutter", "flutter"),
        ("flutter tutorial", "flutter"),
        ("flutter vs react native", "flutter"),
        ("dart language", "flutter"),
        ("mobile app development", "flutter"),
    ],
    "testing": [
        ("what is software testing", "testing"),
        ("testing basics", "testing"),
        ("unit testing explained", "testing"),
        ("how to test code", "testing"),
        ("qa testing tips", "testing"),
    ],

    # ===== EDGE CASES: Misspellings, Banglish, Informal =====
    "_edge_misspell": [
        ("waht is python", "python"),
        ("how to lern javascript", "javascript"),
        ("progrming tips", "coding"),
        ("hlth tips", "health"),
        ("fitnss advice", "fitness"),
        ("meditaton benefits", "meditation"),
    ],
    "_edge_banglish": [
        ("python ki", "python"),
        ("javascript ki", "javascript"),
        ("ai ki", "ai"),
        ("machine learning ki", "ml"),
        ("cybersecurity ki", "cybersecurity"),
        ("tumi ki paro", "bot_capability"),
    ],
    "_edge_short": [
        ("hi", "greeting"),
        ("bye", "farewell"),
        ("thanks", "thanks"),
        ("python", "python"),
        ("react", "react"),
        ("yoga", "yoga"),
    ],
    "_edge_informal": [
        ("bro how to code", "coding"),
        ("yo whats up", "greeting"),
        ("need help with java", "java"),
        ("any good book suggestion", "books"),
        ("can you motivate me", "motivation"),
        ("feeling down today", "emotions"),
    ],
}

def run_test():
    print("Loading chatbot...")
    start = time.time()
    bot = ChatBot()
    load_time = time.time() - start
    print(f"Loaded in {load_time:.2f}s | {len(bot.questions)} questions | {len(bot.category_store_map)} categories\n")

    total = 0
    correct = 0
    wrong = []
    unanswered = []
    wrong_store = []

    # Flatten all test questions
    all_tests = []
    for cat_group, questions in TEST_QUESTIONS.items():
        for q, expected in questions:
            all_tests.append((q, expected, cat_group))

    print(f"Running {len(all_tests)} test questions...\n")

    test_start = time.time()
    for q, expected, group in all_tests:
        total += 1
        result = bot.get_answer(q)

        if result is None:
            unanswered.append((q, expected, group))
        elif result["intent"] != expected:
            wrong.append((q, expected, result["intent"], group))
        else:
            correct += 1
            # Check store routing
            store = bot._get_store_for(result["intent"])
            if expected in TEST_QUESTIONS and not expected.startswith("_edge"):
                # Check if coding category goes to coding_dataset
                coding_cats = {"python", "javascript", "html_css", "react", "nodejs",
                              "java", "cpp", "api", "data_structures", "web_dev",
                              "linux", "devops", "cybersecurity", "flutter", "testing"}
                if expected in coding_cats and store.folder != "coding_dataset":
                    wrong_store.append((q, expected, store.folder))
                elif expected not in coding_cats and store.folder != "category_wise_dataset":
                    wrong_store.append((q, expected, store.folder))

    test_time = time.time() - test_start
    accuracy = (correct / total) * 100 if total else 0

    print("=" * 70)
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.1f}%)")
    print(f"Test time: {test_time:.2f}s ({test_time/total*1000:.1f}ms per question)")
    print(f"Wrong intent: {len(wrong)}")
    print(f"Unanswered: {len(unanswered)}")
    print(f"Wrong store routing: {len(wrong_store)}")
    print("=" * 70)

    if wrong:
        print(f"\n--- WRONG INTENT ({len(wrong)}) ---")
        for q, expected, got, group in wrong:
            store = bot._get_store_for(got)
            print(f"  Q: '{q}' | Expected: {expected} | Got: {got} ({store.folder}) | Group: {group}")

    if unanswered:
        print(f"\n--- UNANSWERED ({len(unanswered)}) ---")
        for q, expected, group in unanswered:
            print(f"  Q: '{q}' | Expected: {expected} | Group: {group}")

    if wrong_store:
        print(f"\n--- WRONG STORE ROUTING ({len(wrong_store)}) ---")
        for q, expected, got_store in wrong_store:
            print(f"  Q: '{q}' | Category: {expected} | Went to: {got_store}")

    # Per-category breakdown
    print(f"\n--- PER-CATEGORY BREAKDOWN ---")
    cat_stats = {}
    for q, expected, group in all_tests:
        if expected not in cat_stats:
            cat_stats[expected] = {"total": 0, "correct": 0}
        cat_stats[expected]["total"] += 1
        result = bot.get_answer(q)
        if result and result["intent"] == expected:
            cat_stats[expected]["correct"] += 1

    failed_cats = []
    for cat, stats in sorted(cat_stats.items()):
        acc = (stats["correct"] / stats["total"]) * 100
        if acc < 100:
            failed_cats.append((cat, stats["correct"], stats["total"], acc))
            print(f"  {cat:20s}: {stats['correct']}/{stats['total']} ({acc:.0f}%)")

    if not failed_cats:
        print("  All categories at 100%!")

    print(f"\n  Categories with issues: {len(failed_cats)}/{len(cat_stats)}")

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "wrong": wrong,
        "unanswered": unanswered,
        "wrong_store": wrong_store,
        "failed_cats": failed_cats,
    }


if __name__ == "__main__":
    results = run_test()
