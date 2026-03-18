"""
1000 Questions across ALL Categories — Real User Style
Tests: typos, informal language, Bangla, short queries, complex questions
Run: python test_1000.py          (full mode with TinyLlama)
     python test_1000.py --fast   (skip TinyLlama, faster)
"""

import sys
import json
import time
import functools

# Force flush on every print so terminal shows live progress
print = functools.partial(print, flush=True)

# Each tuple: (question, expected_category, difficulty_tag)
TEST_QUESTIONS = [
    # ===================================================================
    # CODING / TECH — coding_dataset categories (65 categories, ~130 Qs)
    # ===================================================================

    # python
    ("why python is so populer?", "python", "typo"),
    # react
    ("what is react hooks?", "react", "direct"),
    # javascript
    ("how javascript async await works?", "javascript", "technical"),
    # html_css
    ("html ar css diye ki ki kora jay?", "html_css", "bangla"),
    # sql
    ("sql join types explain koro", "sql", "bangla-mix"),
    # mongodb
    ("how to connect mongodb with node?", "mongodb", "multi-topic"),
    # rest_api
    ("what is REST API?", "rest_api", "direct"),
    # flask_framework
    ("flask vs django which is better?", "flask_framework", "comparison"),
    # typescript
    ("what is typescript?", "typescript", "direct"),
    # rust
    ("rust programming language er advantages ki?", "rust", "bangla-mix"),
    # aws
    ("how to deploy on aws?", "aws", "direct"),
    # graphql
    ("graphql vs rest api difference", "graphql", "comparison"),
    # jwt
    ("what is jwt token?", "jwt", "direct"),
    # redis
    ("how to setup redis cache?", "redis", "direct"),
    # postgresql
    ("postgresql vs mysql which better?", "postgresql", "comparison"),
    # tailwind
    ("tailwind css ki?", "tailwind", "bangla"),
    # nextjs
    ("what is next.js?", "nextjs", "direct"),
    # angular
    ("angular vs react vs vue", "angular", "comparison"),
    # express
    ("express.js diye server banabo kivabe?", "express", "bangla"),
    # laravel
    ("laravel framework ki?", "laravel", "bangla"),
    # devops
    ("devops ki and ci cd ki?", "devops", "bangla-mix"),
    # testing
    ("how to write unit test?", "testing", "direct"),
    # linux
    ("linux basic commands", "linux", "short"),
    # cybersecurity
    ("what is cybersecurity?", "cybersecurity", "direct"),
    # data_structures
    ("data structure ar algorithm shikbo kivabe?", "data_structures", "bangla"),
    # oauth
    ("what is oauth 2.0?", "oauth", "direct"),
    # cpp
    ("c++ vs java performance", "cpp", "comparison"),
    # flutter
    ("flutter diye mobile app banano", "flutter", "bangla"),
    # swift
    ("swift programming for ios", "swift", "direct"),
    # rails
    ("ruby on rails ki?", "rails", "bangla"),
    # websocket
    ("how websocket works?", "websocket", "direct"),
    # vite
    ("webpack vs vite which is faster?", "vite", "comparison"),
    # php
    ("php ki ekhono use hoy?", "php", "bangla"),
    # api
    ("api ki jinish?", "api", "bangla"),
    # java
    ("java vs python konta shikbo age?", "java", "bangla-mix"),
    # csharp
    ("in c# arraylist vs list", "csharp", "comparison"),
    # go_lang
    ("golang ki fast?", "go_lang", "bangla-mix"),
    # kotlin
    ("java vs kotlin for android", "kotlin", "comparison"),
    # scala
    ("what is scala programming?", "scala", "direct"),
    # perl
    ("perl still used in 2024?", "perl", "direct"),
    # r_language
    ("r language for data science", "r_language", "direct"),
    # matlab
    ("matlab vs python for engineering", "matlab", "comparison"),
    # julia
    ("julia language ki fast?", "julia", "bangla-mix"),
    # haskell
    ("haskell functional programming", "haskell", "direct"),
    # elixir
    ("elixir language ki?", "elixir", "bangla"),
    # clojure
    ("clojure vs other lisps", "clojure", "comparison"),
    # lua
    ("lua scripting for games", "lua", "direct"),
    # dart
    ("dart language flutter er jonno?", "dart", "bangla"),
    # solidity
    ("solidity smart contract kivabe likhi?", "solidity", "bangla"),
    # grpc
    ("grpc vs rest api", "grpc", "comparison"),
    # nuxtjs
    ("nuxt.js ki vue er framework?", "nuxtjs", "bangla-mix"),
    # svelte
    ("svelte vs react performance", "svelte", "comparison"),
    # vue
    ("vue.js tutorial for beginners", "vue", "direct"),
    # bootstrap
    ("bootstrap css framework ki?", "bootstrap", "bangla"),
    # sass
    ("sass vs css difference", "sass", "comparison"),
    # django
    ("django framework shikhi kivabe?", "django", "bangla"),
    # less
    ("less css preprocessor ki?", "less", "bangla"),
    # fastapi
    ("fastapi vs flask which faster?", "fastapi", "comparison"),
    # spring_boot
    ("spring boot microservice tutorial", "spring_boot", "direct"),
    # dotnet
    (".net framework ki?", "dotnet", "bangla"),
    # azure
    ("azure vs aws konta better?", "azure", "comparison"),
    # nodejs
    ("node.js backend development", "nodejs", "direct"),
    # web_dev
    ("web development roadmap 2024", "web_dev", "direct"),
    # ruby
    ("ruby programming basics", "ruby", "direct"),
    # webpack
    ("webpack configuration kivabe kore?", "webpack", "bangla"),

    # ===================================================================
    # CATEGORY_WISE_DATASET — Main categories (~870 questions)
    # ===================================================================

    # --- HEALTH & WELLNESS ---
    ("how to improve helth condiiton?", "health", "typo"),
    ("whcih fruite provide best protine?", "nutrition", "typo"),
    ("ki khele weight kombe?", "diet_plans", "bangla"),
    ("how to lose weight fast?", "fitness", "direct"),
    ("yoga benefits ki ki?", "yoga", "bangla"),
    ("meditation kivabe korbo?", "meditation", "bangla"),
    ("how to sleep better at night?", "sleep", "direct"),
    ("skincare routine for oily skin", "skincare", "direct"),
    ("vitamin d er upokaarita ki?", "vitamins", "bangla"),
    ("protein powder khaoa uchit?", "protein", "bangla"),
    ("diabetes control kivabe korbo?", "diabetes", "bangla"),
    ("blood pressure high hole ki korbo?", "blood_pressure", "direct"),
    ("cholesterol komanor upay", "cholesterol", "bangla"),
    ("how to deal with headache?", "headache", "direct"),
    ("back pain remedy at home", "back_pain", "direct"),
    ("eye care tips for computer users", "eye_care", "direct"),
    ("dental care basics", "dental_care", "direct"),
    ("ear care tips ki?", "ear_care", "bangla"),
    ("heart health improve korbo kivabe?", "heart_health", "bangla"),
    ("kidney health maintain korar tips", "kidney_health", "bangla"),
    ("liver health improve korar upay", "liver_health", "bangla"),
    ("bone health er jonno ki khai?", "bone_health", "bangla"),
    ("joint pain er remedy ki?", "joint_pain", "bangla"),
    ("asthma control tips", "asthma", "direct"),
    ("allergy hle ki korbo?", "allergy", "bangla"),
    ("migraine er treatment ki?", "migraine", "bangla"),
    ("insomnia cure kivabe?", "insomnia", "bangla"),
    ("sleep apnea ki jinish?", "sleep_apnea", "bangla"),
    ("snoring stop korar upay", "snoring", "bangla"),
    ("sunburn treatment at home", "sunburn", "direct"),
    ("dehydration symptoms ki ki?", "dehydration", "bangla"),
    ("food poisoning hole ki korbo?", "food_poisoning", "bangla"),
    ("how to treat burns?", "burns", "direct"),
    ("fracture hole first aid ki?", "fracture", "direct"),
    ("snake bite first aid", "snake_bite", "direct"),
    ("insect bite treatment", "insect_bite", "direct"),
    ("drowning rescue kivabe kore?", "drowning", "bangla"),
    ("frostbite treatment tips", "frostbite", "direct"),
    ("heatstroke er lakkhon ki?", "heatstroke", "bangla"),
    ("altitude sickness symptoms", "altitude_sickness", "direct"),
    ("motion sickness er remedy", "motion_sickness", "bangla"),
    ("cpr kivabe dei?", "cpr", "bangla"),
    ("first aid kit e ki ki thakbe?", "first_aid", "bangla"),

    # --- MENTAL HEALTH & EMOTIONS ---
    ("i feel so lonely", "loneliness", "emotion"),
    ("ami onek depressed feel korchi", "depression_support", "bangla-emotion"),
    ("anger control kivabe korbo?", "anger_management", "bangla"),
    ("anxiety attack hole ki korbo?", "anxiety", "bangla"),
    ("how to deal with stress?", "stress_management", "direct"),
    ("burnout feel korchi ki korbo?", "burnout", "emotion"),
    ("imposter syndrome ki?", "imposter_syndrome", "bangla"),
    ("ptsd er treatment ki?", "ptsd", "bangla"),
    ("grief counseling ki?", "grief", "direct"),
    ("loss er por kivabe cope korbo?", "loss", "bangla-emotion"),
    ("how to build self confidence?", "self_confidence", "direct"),
    ("self care routine ki?", "self_care", "bangla"),
    ("mindfulness practice kivabe shuru korbo?", "mindfulness", "bangla"),
    ("therapy ki help kore?", "therapy", "direct"),
    ("counseling er upokaarita ki?", "counseling", "bangla"),
    ("how to manage emotions?", "emotions", "direct"),
    ("mental health improve korar upay", "mental_health", "bangla"),
    ("homesickness dur korar upay", "homesickness", "bangla"),

    # --- CAREER & EDUCATION ---
    ("how to write a good resume?", "resume", "direct"),
    ("interview tips for freshers", "interview", "direct"),
    ("freelancing kivabe shuru korbo?", "freelancing", "bangla"),
    ("how to study effectively?", "study", "direct"),
    ("salary negotiation tips", "salary", "direct"),
    ("career change korbo kivabe?", "career", "bangla"),
    ("best college for computer science?", "college", "direct"),
    ("motivation nai ki korbo?", "motivation", "bangla"),
    ("leadership skills improve korbo kivabe?", "leadership", "bangla"),
    ("marketing strategy ki?", "marketing", "direct"),
    ("how to be more productive?", "productivity", "direct"),
    ("exam preparation tips", "exam_preparation", "direct"),
    ("public speaking fear kivabe overcome korbo?", "public_speaking", "bangla"),
    ("how to write technical documents?", "technical_writing", "direct"),
    ("cover letter kivabe likhi?", "cover_letter", "bangla"),
    ("job interview er preparation ki?", "job_interview", "bangla"),
    ("scholarship kivabe pabo?", "scholarship", "bangla"),
    ("study abroad er process ki?", "study_abroad", "bangla"),
    ("study techniques for exam", "study_techniques", "direct"),
    ("academic writing tips", "academic_writing", "direct"),
    ("research paper kivabe likhi?", "research_paper", "bangla"),
    ("thesis writing tips", "thesis", "direct"),
    ("dissertation ki?", "dissertation", "bangla"),
    ("resume writing service lagbe?", "resume_writing", "direct"),
    ("salary negotiation tactics", "salary_negotiation", "direct"),
    ("career coaching ki?", "career_coaching", "bangla"),
    ("mentorship er importance ki?", "mentorship", "bangla"),

    # --- BUSINESS & FINANCE ---
    ("how to start a business?", "startup", "direct"),
    ("how to save money?", "budgeting", "direct"),
    ("bitcoin ki safe invest?", "crypto", "bangla-mix"),
    ("stock market e invest korbo kivabe?", "stock_market", "bangla"),
    ("mutual fund ki?", "mutual_funds", "bangla"),
    ("how to do affiliate marketing?", "affiliate_marketing", "direct"),
    ("dropshipping business ki?", "dropshipping", "bangla"),
    ("ecommerce site kivabe banabo?", "ecommerce", "bangla"),
    ("seo tips for beginners", "seo", "direct"),
    ("email marketing strategy", "email_marketing", "direct"),
    ("content creation tips", "content_creation", "direct"),
    ("social media marketing", "social_media", "direct"),
    ("how to do freelance tax?", "freelance_tax", "direct"),
    ("business plan kivabe likhi?", "business_plan", "bangla"),
    ("business model canvas ki?", "business_model", "bangla"),
    ("entrepreneurship shuru korbo kivabe?", "entrepreneurship", "bangla"),
    ("small business tips", "small_business", "direct"),
    ("franchise business ki?", "franchise", "bangla"),
    ("real estate investment tips", "real_estate", "direct"),
    ("insurance ki lagbe?", "insurance", "direct"),
    ("credit card tips and tricks", "credit_card", "direct"),
    ("credit score improve korbo kivabe?", "credit_score", "bangla"),
    ("debit card vs credit card", "debit_card", "comparison"),
    ("debt management tips", "debt_management", "direct"),
    ("taxation basics in bangladesh", "taxation", "direct"),
    ("accounting basics for beginners", "accounting", "direct"),
    ("bookkeeping ki?", "bookkeeping", "bangla"),
    ("auditing process ki?", "auditing", "bangla"),
    ("invoice kivabe banai?", "invoice", "bangla"),
    ("bonds investment ki safe?", "bonds", "bangla-mix"),
    ("angel investing ki?", "angel_investing", "bangla"),
    ("venture capital ki?", "venture_capital", "bangla"),
    ("ipo ki jinish?", "ipo", "bangla"),
    ("defi decentralized finance ki?", "defi", "bangla"),
    ("nft ki jinish actually?", "nft", "bangla"),
    ("personal branding tips", "personal_branding", "direct"),
    ("branding strategy for business", "branding_strategy", "direct"),
    ("pricing strategy set korbo kivabe?", "pricing_strategy", "bangla"),
    ("revenue model ki?", "revenue_model", "bangla"),
    ("subscription model business", "subscription_model", "direct"),
    ("competitor analysis korbo kivabe?", "competitor_analysis", "bangla"),
    ("market research basics", "market_research", "direct"),
    ("customer retention strategy", "customer_retention", "direct"),
    ("customer service improve korbo kivabe?", "customer_service", "bangla"),
    ("sales technique best practices", "sales_technique", "direct"),
    ("cold calling tips", "cold_calling", "direct"),
    ("elevator pitch ki?", "elevator_pitch", "bangla"),
    ("pitch deck banabo kivabe?", "pitch_deck", "bangla"),
    ("swot analysis ki?", "swot_analysis", "bangla"),
    ("crm software ki?", "crm", "bangla"),
    ("supply chain management ki?", "supply_chain", "bangla"),
    ("logistics business ki?", "logistics", "bangla"),
    ("warehousing tips", "warehousing", "direct"),
    ("inventory management basics", "inventory_management", "direct"),
    ("import export business ki?", "import_export", "bangla"),
    ("wholesale business tips", "wholesale", "direct"),
    ("retail business strategy", "retail", "direct"),
    ("gig economy ki?", "gig_economy", "bangla"),

    # --- TECHNOLOGY & GADGETS ---
    ("difference between mobile and laptop", "technology", "comparison"),
    ("smart home automation ki?", "smart_home", "bangla"),
    ("iot devices ki ki ache?", "iot", "bangla"),
    ("electric vehicles future ki?", "electric_vehicles", "bangla"),
    ("wearable tech trends", "wearable_tech", "direct"),
    ("bluetooth technology ki?", "bluetooth", "bangla"),
    ("nfc payment ki?", "nfc", "bangla"),
    ("gps technology ki?", "gps", "bangla"),
    ("wifi security tips", "wifi_security", "direct"),
    ("vpn ki and keno use korbo?", "vpn", "bangla-mix"),
    ("antivirus software lagbe ki?", "antivirus", "direct"),
    ("cloud storage best options", "cloud_storage", "direct"),
    ("backup strategies for data", "backup_strategies", "direct"),
    ("raspberry pi projects", "raspberry_pi", "direct"),
    ("arduino project ideas", "arduino", "direct"),
    ("3d printing ki?", "3d_printing", "bangla"),
    ("drone photography tips", "drone", "direct"),
    ("robotics basics", "robotics", "direct"),
    ("nanotechnology ki?", "nanotechnology", "bangla"),
    ("battery technology future", "battery_technology", "direct"),
    ("barcode vs qr code", "barcode", "comparison"),
    ("qr code generator ki?", "qr_code", "bangla"),
    ("rfid technology ki?", "rfid", "bangla"),

    # --- FOOD & COOKING ---
    ("how to cook rice properly?", "cooking", "direct"),
    ("best biryani recipe", "biryani", "direct"),
    ("pasta recipe easy", "pasta", "direct"),
    ("pizza dough recipe", "pizza", "direct"),
    ("burger banabo kivabe?", "burger", "bangla"),
    ("cake baking tips", "cake", "direct"),
    ("cookies recipe simple", "cookies", "direct"),
    ("bread baking at home", "bread", "direct"),
    ("pie recipe easy", "pie", "direct"),
    ("desserts recipe ideas", "desserts", "direct"),
    ("ice cream banabo kivabe?", "ice_cream", "bangla"),
    ("chocolate recipe homemade", "chocolate", "direct"),
    ("candy making at home", "candy", "direct"),
    ("cheese types and uses", "cheese", "direct"),
    ("sushi ki?", "sushi", "bangla"),
    ("curry recipe bangladeshi", "curry", "direct"),
    ("street food ideas", "street_food", "direct"),
    ("baking tips for beginners", "baking", "direct"),
    ("smoothie recipe healthy", "smoothies", "direct"),
    ("juicing benefits ki?", "juicing", "bangla"),
    ("spices used in cooking", "spices", "direct"),
    ("herbs for cooking", "herbs", "direct"),
    ("coffee brewing methods", "coffee", "direct"),
    ("tea types and benefits", "tea", "direct"),
    ("cocktails recipe easy", "cocktails", "direct"),
    ("beer types explained", "beer", "direct"),
    ("wine basics for beginners", "wine", "direct"),
    ("whiskey types ki ki?", "whiskey", "bangla"),
    ("bbq tips and tricks", "bbq", "direct"),
    ("food preservation methods", "food_preservation", "direct"),
    ("food safety guidelines", "food_safety", "direct"),
    ("food truck business ki?", "food_truck", "bangla"),
    ("bakery business tips", "bakery", "direct"),
    ("catering business ki?", "catering", "bangla"),
    ("restaurant management tips", "restaurant", "direct"),
    ("bartending basics", "bartending", "direct"),
    ("bangladeshi food recipes", "bangladeshi_food", "direct"),
    ("indian food popular dishes", "indian_food", "direct"),
    ("chinese food at home", "chinese_food", "direct"),
    ("japanese food culture", "japanese_food", "direct"),
    ("italian food basics", "italian_food", "direct"),
    ("thai food recipes", "thai_food", "direct"),
    ("french food classics", "french_food", "direct"),
    ("mexican food favorites", "mexican_food", "direct"),
    ("korean food popular dishes", "korean_food", "direct"),
    ("turkish food recipes", "turkish_food", "direct"),

    # --- SPORTS & FITNESS ---
    ("why messi is best then ronaldo?", "football", "opinion"),
    ("best cricket player of all time?", "cricket", "opinion"),
    ("basketball rules ki?", "basketball", "bangla"),
    ("tennis tips for beginners", "tennis", "direct"),
    ("badminton rules explained", "badminton", "direct"),
    ("volleyball court size ki?", "volleyball", "bangla"),
    ("rugby ki?", "rugby", "bangla"),
    ("golf basics for beginners", "golf", "direct"),
    ("baseball rules ki?", "baseball", "bangla"),
    ("boxing training tips", "boxing", "direct"),
    ("wrestling moves basic", "wrestling", "direct"),
    ("mma training ki?", "mma", "bangla"),
    ("table tennis tips", "table_tennis", "direct"),
    ("swimming tips for beginners", "swimming", "direct"),
    ("running tips for marathon", "running", "direct"),
    ("marathon training plan", "marathon", "direct"),
    ("cycling benefits ki?", "cycling", "bangla"),
    ("skateboarding basics", "skateboarding", "direct"),
    ("skiing tips for beginners", "skiing", "direct"),
    ("snowboarding basics", "snowboarding", "direct"),
    ("surfing tips for newbies", "surfing", "direct"),
    ("archery basics ki?", "archery", "bangla"),
    ("fencing sport ki?", "fencing", "bangla"),
    ("horse riding lessons", "horse_riding", "direct"),
    ("rock climbing tips", "rock_climbing", "direct"),
    ("parkour basics", "parkour", "direct"),
    ("ice skating tips", "ice_skating", "direct"),
    ("scuba diving ki?", "scuba_diving", "bangla"),
    ("gym workout routine", "gym_workout", "direct"),
    ("home workout without equipment", "home_workout", "direct"),
    ("stretching exercises", "stretching", "direct"),
    ("flexibility improve korbo kivabe?", "flexibility", "bangla"),
    ("sports photography tips", "sports_photography", "direct"),
    ("polo sport ki?", "polo", "bangla"),

    # --- TRAVEL & ADVENTURE ---
    ("best travel destinations 2024", "travel", "direct"),
    ("solo travel tips", "solo_travel", "direct"),
    ("backpacking essentials", "backpacking", "direct"),
    ("adventure travel destinations", "adventure_travel", "direct"),
    ("camping tips for beginners", "camping", "direct"),
    ("hiking trails near me", "hiking", "direct"),
    ("road trip planning tips", "road_trip", "direct"),
    ("flight booking cheap tickets", "flight_booking", "direct"),
    ("hotel booking tips", "hotel_booking", "direct"),
    ("airbnb tips for hosts", "airbnb", "direct"),
    ("hostel vs hotel konta better?", "hostel", "comparison"),
    ("bus travel tips", "bus_travel", "direct"),
    ("train travel experience", "train_travel", "direct"),
    ("cruise ship vacation", "cruise", "direct"),
    ("ferry travel tips", "ferry", "direct"),
    ("beach vacation best places", "beach_vacation", "direct"),
    ("mountain trip planning", "mountain_trip", "direct"),
    ("desert safari experience", "desert_safari", "direct"),
    ("jungle safari tips", "jungle_safari", "direct"),
    ("safari photography tips", "safari_photography", "direct"),
    ("eco tourism ki?", "eco_tourism", "bangla"),
    ("heritage tourism places", "heritage_tourism", "direct"),
    ("religious tourism destinations", "religious_tourism", "direct"),
    ("family vacation planning", "family_vacation", "direct"),
    ("group travel tips", "group_travel", "direct"),
    ("honeymoon destinations best", "honeymoon", "direct"),
    ("digital nomad lifestyle ki?", "digital_nomad", "bangla"),
    ("expat life challenges", "expat_life", "direct"),
    ("car rental tips", "car_rental", "direct"),
    ("ride sharing apps ki ki ache?", "ride_sharing", "bangla"),
    ("carpooling benefits ki?", "carpooling", "bangla"),
    ("scooter rental for travel", "scooter", "direct"),
    ("motorcycle tour tips", "motorcycle", "direct"),
    ("yacht vacation ki?", "yacht", "bangla"),
    ("boat trip planning", "boat", "direct"),
    ("hot air balloon ride", "hot_air_balloon", "direct"),
    ("bungee jumping ki safe?", "bungee_jumping", "bangla-mix"),
    ("zip lining experience", "zip_lining", "direct"),
    ("paragliding ki?", "paragliding", "bangla"),
    ("skydiving first time tips", "skydiving", "direct"),
    ("river rafting experience", "river_rafting", "direct"),
    ("cave exploration tips", "cave_exploration", "direct"),
    ("volcano tour ki safe?", "volcano_tour", "bangla-mix"),
    ("northern lights dekha jay kothay?", "northern_lights", "bangla"),
    ("whale watching best spots", "whale_watching", "direct"),
    ("coral diving tips", "coral_diving", "direct"),
    ("treasure hunt games", "treasure_hunt", "direct"),
    ("escape room tips", "escape_room", "direct"),
    ("national parks to visit", "national_parks", "direct"),
    ("airport tips for first time flyers", "airport_tips", "direct"),
    ("passport process ki?", "passport", "bangla"),
    ("visa process explained", "visa_process", "direct"),
    ("visa tips for students", "visa_tips", "direct"),
    ("customs rules when traveling", "customs", "direct"),
    ("travel photography tips", "travel_photography", "direct"),

    # --- SCIENCE & NATURE ---
    ("science facts for kids", "science", "direct"),
    ("physics basic concepts", "physics", "direct"),
    ("chemistry interesting facts", "chemistry", "direct"),
    ("biology basics ki?", "biology", "bangla"),
    ("astronomy interesting facts", "astronomy", "direct"),
    ("astrophysics ki?", "astrophysics", "bangla"),
    ("space exploration latest news", "space_exploration", "direct"),
    ("mars mission update", "mars_mission", "direct"),
    ("black holes ki?", "black_holes", "bangla"),
    ("big bang theory explained", "big_bang", "direct"),
    ("quantum computing ki?", "quantum_computing", "direct"),
    ("nanotechnology applications", "nanotechnology", "direct"),
    ("genetics basics", "genetics", "direct"),
    ("evolution theory ki?", "evolution", "bangla"),
    ("ecology and environment", "ecology", "direct"),
    ("marine biology facts", "marine_biology", "direct"),
    ("microbiology basics", "microbiology", "direct"),
    ("botany plant science", "botany", "direct"),
    ("zoology animal science ki?", "zoology", "bangla"),
    ("epidemiology ki?", "epidemiology", "bangla"),
    ("virology basics", "virology", "direct"),
    ("immunology ki?", "immunology", "bangla"),
    ("bacteriology basics", "bacteriology", "direct"),
    ("parasitology ki?", "parasitology", "bangla"),
    ("biotechnology applications", "biotechnology", "direct"),
    ("material science ki?", "material_science", "bangla"),
    ("weather forecast basics", "weather", "direct"),
    ("climate change effects", "climate_change", "direct"),
    ("coral reef conservation", "coral_reef", "direct"),
    ("endangered species list", "endangered_species", "direct"),
    ("wildlife conservation", "wildlife", "direct"),
    ("ocean cleanup efforts", "ocean_cleanup", "direct"),
    ("plastic pollution problem", "plastic_pollution", "direct"),
    ("recycling tips", "recycling", "direct"),
    ("composting at home", "composting", "direct"),
    ("zero waste lifestyle", "zero_waste", "direct"),
    ("conservation efforts", "conservation", "direct"),
    ("reforestation projects", "reforestation", "direct"),
    ("deforestation effects ki?", "deforestation", "bangla"),
    ("wetlands importance ki?", "wetlands", "bangla"),
    ("soil health tips", "soil_health", "direct"),

    # --- ENERGY ---
    ("solar energy er future ki?", "solar_energy", "bangla"),
    ("wind energy basics", "wind_energy", "direct"),
    ("hydropower ki?", "hydropower", "bangla"),
    ("nuclear energy pros and cons", "nuclear_energy", "comparison"),
    ("geothermal energy ki?", "geothermal", "bangla"),
    ("tidal energy basics", "tidal_energy", "direct"),
    ("biomass energy ki?", "biomass", "bangla"),
    ("hydrogen fuel cell technology", "hydrogen_fuel", "direct"),
    ("fuel cell car ki?", "fuel_cell", "bangla"),
    ("energy storage solutions", "energy_storage", "direct"),
    ("energy efficiency tips", "energy_efficiency", "direct"),
    ("renewable energy future", "renewable_energy", "direct"),
    ("smart grid technology", "smart_grid", "direct"),
    ("power grid basics", "power_grid", "direct"),

    # --- AI & ML ---
    ("tumi ki ai?", "ai", "bangla"),
    ("machine learning ki?", "ml", "bangla"),
    ("deep learning vs machine learning", "dl", "comparison"),
    ("nlp natural language processing ki?", "nlp", "bangla"),
    ("computer vision applications", "computer_vision", "direct"),
    ("reinforcement learning ki?", "reinforcement_learning", "bangla"),
    ("gan generative adversarial network", "gan", "direct"),
    ("llm large language model ki?", "llm", "bangla"),
    ("prompt engineering tips", "prompt_engineering", "direct"),
    ("fine tuning models ki?", "fine_tuning", "bangla"),
    ("transfer learning explained", "transfer_learning", "direct"),
    ("gpt model ki?", "gpt_model", "bangla"),
    ("bert model explained", "bert", "direct"),
    ("attention mechanism ki?", "attention_mechanism", "bangla"),
    ("transformer model architecture", "transformer_model", "direct"),
    ("diffusion model ki?", "diffusion_model", "bangla"),
    ("image generation with ai", "image_generation", "direct"),
    ("text classification basics", "text_classification", "direct"),
    ("sentiment analysis ki?", "sentiment_analysis", "bangla"),
    ("named entity recognition", "named_entity", "direct"),
    ("machine translation ki?", "machine_translation", "bangla"),
    ("speech recognition technology", "speech_recognition", "direct"),
    ("speech to text ki?", "speech_to_text", "bangla"),
    ("text to speech technology", "text_to_speech", "direct"),
    ("question answering system", "question_answering", "direct"),
    ("summarization with ai", "summarization", "direct"),
    ("paraphrasing tools ki?", "paraphrasing", "bangla"),
    ("tokenization in nlp", "tokenization", "technical"),
    ("embeddings ki?", "embeddings", "bangla"),
    ("recommendation system ki?", "recommendation_system", "bangla"),
    ("chatbot development ki?", "chatbot_development", "bangla"),
    ("mlops ki?", "mlops", "bangla"),
    ("datascience shikbo kivabe?", "datascience", "bangla"),
    ("data visualization tools", "data_visualization", "direct"),
    ("classification vs regression", "classification", "comparison"),
    ("regression analysis ki?", "regression", "bangla"),
    ("clustering algorithms", "clustering", "direct"),
    ("time series analysis", "time_series", "direct"),
    ("pytorch vs tensorflow", "pytorch", "comparison"),
    ("tensorflow tutorial", "tensorflow", "direct"),
    ("keras basics ki?", "keras", "bangla"),
    ("opencv for image processing", "opencv", "direct"),

    # --- ENGINEERING ---
    ("civil engineering basics", "civil_engineering", "direct"),
    ("mechanical engineering ki?", "mechanical_engineering", "bangla"),
    ("electrical engineering basics", "electrical_engineering", "direct"),
    ("chemical engineering ki?", "chemical_engineering", "bangla"),
    ("aerospace engineering ki?", "aerospace_engineering", "bangla"),
    ("biomedical engineering applications", "biomedical_engineering", "direct"),
    ("marine engineering ki?", "marine_engineering", "bangla"),
    ("mining engineering basics", "mining_engineering", "direct"),
    ("petroleum engineering ki?", "petroleum_engineering", "bangla"),

    # --- DESIGN & CREATIVE ---
    ("graphic design basics", "graphic_design", "direct"),
    ("ui design principles", "ui_design", "direct"),
    ("ux design process ki?", "ux_design", "bangla"),
    ("web design tips", "web_design", "direct"),
    ("logo design ki?", "logo_design", "bangla"),
    ("color theory basics", "color_theory", "direct"),
    ("typography fundamentals", "typography", "direct"),
    ("branding for startups", "branding", "direct"),
    ("wireframing tools ki ki?", "wireframing", "bangla"),
    ("prototyping with figma", "prototyping", "direct"),
    ("figma tips and tricks", "figma", "direct"),
    ("photoshop basics", "photoshop", "direct"),
    ("illustrator tutorial ki?", "illustrator", "bangla"),
    ("canva design tips", "canva", "direct"),
    ("premiere pro video editing", "premiere_pro", "direct"),
    ("after effects motion graphics", "after_effects", "direct"),
    ("lightroom photo editing", "lightroom", "direct"),
    ("usability testing ki?", "usability_testing", "bangla"),
    ("3d modeling basics", "3d_modeling", "direct"),
    ("animation fundamentals", "animation", "direct"),
    ("motion graphics ki?", "motion_graphics", "bangla"),
    ("vfx visual effects basics", "vfx", "direct"),
    ("digital art tips", "digital_art", "direct"),
    ("pixel art ki?", "pixel_art", "bangla"),
    ("character design basics", "character_design", "direct"),
    ("interior design tips", "interior_design", "direct"),
    ("landscape design basics", "landscape_design", "direct"),
    ("architecture design principles", "architecture", "direct"),
    ("sustainable design ki?", "sustainable_design", "bangla"),

    # --- PHOTOGRAPHY ---
    ("how to take good photos?", "photography", "direct"),
    ("portrait photography tips", "portrait_photography", "direct"),
    ("landscape photography basics", "landscape_photography", "direct"),
    ("street photography ki?", "street_photography", "bangla"),
    ("food photography tips", "food_photography", "direct"),
    ("wildlife photography basics", "wildlife_photography", "direct"),
    ("fashion photography ki?", "fashion_photography", "bangla"),
    ("wedding photography tips", "wedding_photography", "direct"),
    ("product photography basics", "product_photography", "direct"),
    ("macro photography ki?", "macro_photography", "bangla"),
    ("underwater photography tips", "underwater_photography", "direct"),
    ("astrophotography basics", "astrophotography", "direct"),
    ("drone photography ki?", "drone_photography", "bangla"),
    ("film photography basics", "film_photography", "direct"),
    ("photo editing tips", "photo_editing", "direct"),
    ("darkroom processing ki?", "darkroom", "bangla"),
    ("color grading basics", "color_grading", "direct"),

    # --- MUSIC & ENTERTAINMENT ---
    ("music shunle ki mental health e help kore?", "music", "bangla"),
    ("guitar shikbo kivabe?", "guitar", "bangla"),
    ("piano basics for beginners", "piano", "direct"),
    ("drums learning tips", "drums", "direct"),
    ("violin shikhi kivabe?", "violin", "bangla"),
    ("singing tips for beginners", "singing", "direct"),
    ("music production basics", "music_production", "direct"),
    ("djing ki?", "djing", "bangla"),
    ("hip hop music history", "hip_hop", "direct"),
    ("rock music er evolution", "rock_music", "bangla"),
    ("classical music ki?", "classical_music", "bangla"),
    ("bangla music best songs", "bangla_music", "direct"),
    ("movies recommendation", "movies", "direct"),
    ("anime best shows", "anime", "direct"),
    ("manga reading ki?", "manga", "bangla"),
    ("webtoon ki?", "webtoon", "bangla"),
    ("bollywood best movies", "bollywood", "direct"),
    ("gaming tips", "gaming", "direct"),
    ("board games family night", "board_games", "direct"),
    ("card games rules", "card_games", "direct"),
    ("puzzle games brain training", "puzzle_games", "direct"),
    ("chess strategy tips", "chess", "direct"),
    ("karaoke tips for fun", "karaoke", "direct"),
    ("standup comedy ki?", "standup_comedy", "bangla"),
    ("theater acting basics", "theater", "direct"),
    ("acting tips for beginners", "acting", "direct"),
    ("film making basics", "film_making", "direct"),
    ("cinematography ki?", "cinematography", "bangla"),
    ("screenplay writing tips", "screenplay", "direct"),
    ("screenwriting basics", "screenwriting", "direct"),
    ("documentary making ki?", "documentary", "bangla"),
    ("short film ideas", "short_film", "direct"),
    ("video editing basics", "video_editing", "direct"),
    ("sound design ki?", "sound_design", "bangla"),
    ("podcast how to start?", "podcast", "direct"),
    ("podcasting equipment", "podcasting", "direct"),
    ("vlogging tips for youtube", "vlogging", "direct"),
    ("voice training ki?", "voice_training", "bangla"),

    # --- WRITING & LANGUAGE ---
    ("how to improve writing skills?", "writing", "direct"),
    ("fiction writing tips", "fiction_writing", "direct"),
    ("poetry ki?", "poetry", "bangla"),
    ("storytelling techniques", "storytelling", "direct"),
    ("copywriting basics", "copywriting", "direct"),
    ("journalism career ki?", "journalism", "bangla"),
    ("editing tips for writers", "editing", "direct"),
    ("proofreading skills", "proofreading", "direct"),
    ("translation career ki?", "translation", "bangla"),
    ("blogging tips for beginners", "blogging", "direct"),
    ("fan fiction ki?", "fan_fiction", "bangla"),
    ("bibliography format", "bibliography", "direct"),
    ("citation styles explained", "citation", "direct"),
    ("plagiarism ki?", "plagiarism", "bangla"),
    ("how to learn a new language?", "language", "direct"),
    ("sign language basics", "sign_language", "direct"),
    ("braille reading ki?", "braille", "bangla"),

    # --- LIFESTYLE ---
    ("fashion trend 2024", "fashion", "direct"),
    ("makeup tips for beginners", "makeup", "direct"),
    ("haircare routine", "haircare", "direct"),
    ("nail care tips", "nail_care", "direct"),
    ("tattoo ki safe?", "tattoo", "bangla-mix"),
    ("piercing aftercare tips", "piercing", "direct"),
    ("perfume selection tips", "perfume", "direct"),
    ("watches collection ki?", "watches", "bangla"),
    ("jewelry buying guide", "jewelry", "direct"),
    ("luxury goods ki ki?", "luxury_goods", "bangla"),
    ("minimalism lifestyle", "minimalism", "direct"),
    ("decluttering tips", "decluttering", "direct"),
    ("morning routine tips", "morning_routine", "direct"),
    ("evening routine ideas", "evening_routine", "direct"),
    ("habit tracking ki?", "habit_tracking", "bangla"),
    ("habits building tips", "habits", "direct"),
    ("goal setting framework", "goal_setting", "direct"),
    ("time management tips", "time_management", "direct"),
    ("work life balance kivabe maintain korbo?", "work_life_balance", "bangla"),
    ("remote work tips", "remote_work", "direct"),
    ("coworking space ki?", "coworking", "bangla"),
    ("dating tips", "dating", "direct"),
    ("relationship advice dao", "relationship", "bangla-mix"),
    ("marriage tips for couples", "marriage", "direct"),
    ("divorce process ki?", "divorce", "bangla"),
    ("wedding planning checklist", "wedding_planning", "direct"),
    ("parenting tips", "parenting", "direct"),
    ("baby care basics", "baby_care", "direct"),
    ("breastfeeding tips", "breastfeeding", "direct"),
    ("toddler activities", "toddler", "direct"),
    ("child development stages", "child_development", "direct"),
    ("teenage problems ki?", "teenage", "bangla"),
    ("adolescence challenges", "adolescence", "direct"),
    ("puberty ki?", "puberty", "bangla"),
    ("sex education basics", "sex_education", "direct"),
    ("contraception options", "contraception", "direct"),
    ("pregnancy care tips", "pregnancy", "direct"),
    ("childbirth preparation", "childbirth", "direct"),
    ("fertility tips", "fertility", "direct"),
    ("ivf process ki?", "ivf", "bangla"),
    ("surrogacy ki?", "surrogacy", "bangla"),
    ("adoption process", "adoption", "direct"),
    ("foster care ki?", "foster_care", "bangla"),
    ("single parenting tips", "single_parenting", "direct"),
    ("elder care ki?", "elder_care", "bangla"),
    ("retirement planning", "retirement", "direct"),
    ("pension system ki?", "pension", "bangla"),
    ("estate planning basics", "estate_planning", "direct"),
    ("will writing ki?", "will_writing", "bangla"),
    ("funeral planning ki?", "funeral_planning", "bangla"),

    # --- HOME & DIY ---
    ("diy projects for home", "diy", "direct"),
    ("gardening tips for beginners", "gardening", "direct"),
    ("indoor plants care", "indoor_plants", "direct"),
    ("plant care basics", "plant_care", "direct"),
    ("succulent care ki?", "succulent", "bangla"),
    ("flower arrangement ideas", "flower_arrangement", "direct"),
    ("woodworking basics", "woodworking", "direct"),
    ("sewing for beginners", "sewing", "direct"),
    ("knitting basics ki?", "knitting", "bangla"),
    ("embroidery tips", "embroidery", "direct"),
    ("pottery making ki?", "pottery", "bangla"),
    ("ceramics basics", "ceramics", "direct"),
    ("painting techniques", "painting", "direct"),
    ("calligraphy ki?", "calligraphy", "bangla"),
    ("origami for beginners", "origami", "direct"),
    ("sculpture basics ki?", "sculpture", "bangla"),
    ("plumbing basics", "plumbing", "direct"),
    ("electricity basics at home", "electricity_basics", "direct"),
    ("car maintenance tips", "car_maintenance", "direct"),
    ("bicycle repair at home", "bicycle_repair", "direct"),
    ("tire change kivabe kore?", "tire_change", "bangla"),
    ("electronics repair basics", "electronics", "direct"),

    # --- PETS & ANIMALS ---
    ("pet cat er care kivabe korbo?", "pets", "bangla"),
    ("bird watching ki?", "bird_watching", "bangla"),
    ("aquaculture basics", "aquaculture", "direct"),
    ("fishery management ki?", "fishery", "bangla"),
    ("beekeeping basics", "beekeeping", "direct"),
    ("dairy farming ki?", "dairy_farming", "bangla"),
    ("poultry farming basics", "poultry", "direct"),
    ("animal rights ki?", "animal_rights", "bangla"),

    # --- SOCIAL & LEGAL ---
    ("human rights ki ki?", "human_rights", "bangla"),
    ("child rights ki?", "child_rights", "bangla"),
    ("women safety tips", "women_safety", "direct"),
    ("domestic violence help", "domestic_violence", "direct"),
    ("sexual harassment reporting", "sexual_harassment", "direct"),
    ("bullying stop korbo kivabe?", "bullying", "bangla"),
    ("cyber bullying ki?", "cyber_bullying", "bangla"),
    ("racism ki?", "racism", "bangla"),
    ("gender equality importance", "gender_equality", "direct"),
    ("disability rights ki?", "disability_rights", "bangla"),
    ("diversity and inclusion", "diversity", "direct"),
    ("cultural awareness tips", "cultural_awareness", "direct"),
    ("consumer rights ki?", "consumer_rights", "bangla"),
    ("patient rights ki?", "patient_rights", "bangla"),
    ("labor law basics", "labor_law", "direct"),
    ("minimum wage ki?", "minimum_wage", "bangla"),
    ("trade union ki?", "trade_union", "bangla"),
    ("legal basics for everyone", "legal_basics", "direct"),
    ("copyright law ki?", "copyright", "bangla"),
    ("trademark registration", "trademark", "direct"),
    ("patent filing ki?", "patents", "bangla"),
    ("intellectual property ki?", "intellectual_property", "bangla"),
    ("contracts basics", "contracts", "direct"),
    ("democracy ki?", "democracy", "bangla"),
    ("elections process ki?", "elections", "bangla"),
    ("voting importance", "voting", "direct"),
    ("political parties ki?", "political_parties", "bangla"),
    ("political science basics", "political_science", "direct"),
    ("governance ki?", "governance", "bangla"),
    ("corruption prevention", "corruption", "direct"),
    ("transparency in government", "transparency", "direct"),
    ("whistleblowing ki?", "whistleblowing", "bangla"),
    ("activism ki?", "activism", "bangla"),
    ("protest rights ki?", "protest", "bangla"),

    # --- INTERNATIONAL ORGANIZATIONS ---
    ("united nations ki kore?", "united_nations", "bangla"),
    ("nato ki?", "nato", "bangla"),
    ("european union ki?", "european_union", "bangla"),
    ("imf ki?", "imf", "bangla"),
    ("world bank er role ki?", "world_bank", "bangla"),
    ("wto ki?", "wto", "bangla"),
    ("unicef ki kore?", "unicef", "bangla"),
    ("red cross ki?", "red_cross", "bangla"),
    ("who world health organization ki?", "who", "bangla"),
    ("ngo ki?", "ngo", "bangla"),

    # --- MATH ---
    ("math basics for students", "math", "direct"),
    ("algebra shikbo kivabe?", "algebra", "bangla"),
    ("calculus ki?", "calculus", "bangla"),
    ("geometry basics", "geometry", "direct"),
    ("trigonometry ki?", "trigonometry", "bangla"),
    ("statistics basics", "statistics", "direct"),
    ("probability ki?", "probability", "bangla"),
    ("linear algebra ki?", "linear_algebra", "bangla"),
    ("differential equations ki?", "differential_equations", "bangla"),
    ("number theory ki?", "number_theory", "bangla"),
    ("set theory basics", "set_theory", "direct"),
    ("graph theory ki?", "graph_theory", "bangla"),
    ("combinatorics ki?", "combinatorics", "bangla"),
    ("topology ki?", "topology", "bangla"),
    ("arithmetic basics for kids", "arithmetic", "direct"),
    ("discrete math ki?", "discrete_math", "bangla"),
    ("mathematical modeling ki?", "mathematical_modeling", "bangla"),
    ("operations research ki?", "operations_research", "bangla"),
    ("optimization techniques", "optimization", "direct"),
    ("game theory ki?", "game_theory", "bangla"),
    ("logical reasoning tips", "logical_reasoning", "direct"),

    # --- DEVOPS & CLOUD ---
    ("docker basics ki?", "docker_basics", "bangla"),
    ("kubernetes ki jinish?", "kubernetes", "bangla"),
    ("git version control", "git", "direct"),
    ("github basics ki?", "github_basics", "bangla"),
    ("github actions ci cd", "github_actions", "direct"),
    ("gitlab ci ki?", "gitlab_ci", "bangla"),
    ("jenkins pipeline ki?", "jenkins", "bangla"),
    ("terraform infrastructure as code", "terraform", "direct"),
    ("ansible automation ki?", "ansible", "bangla"),
    ("ci cd pipeline ki?", "ci_cd", "bangla"),
    ("cloud computing basics", "cloud", "direct"),
    ("cloud security ki?", "cloud_security", "bangla"),
    ("containers ki?", "containers", "bangla"),
    ("microservices architecture", "microservices", "direct"),
    ("serverless computing ki?", "serverless", "bangla"),
    ("edge computing ki?", "edge_computing", "bangla"),
    ("monitoring tools ki?", "monitoring", "bangla"),
    ("logging best practices", "logging", "direct"),
    ("alerting system ki?", "alerting", "bangla"),
    ("incident management process", "incident_management", "direct"),
    ("disaster recovery plan ki?", "disaster_recovery", "bangla"),
    ("version control basics", "version_control", "direct"),
    ("agile methodology ki?", "agile", "bangla"),
    ("scrum framework ki?", "scrum", "bangla"),
    ("kanban board ki?", "kanban", "bangla"),

    # --- DATA & TOOLS ---
    ("database basics", "database", "direct"),
    ("elasticsearch ki?", "elasticsearch", "bangla"),
    ("kafka streaming ki?", "kafka", "bangla"),
    ("hadoop big data ki?", "hadoop", "bangla"),
    ("spark data processing", "spark", "direct"),
    ("airflow workflow ki?", "airflow", "bangla"),
    ("rabbitmq ki?", "rabbitmq", "bangla"),
    ("power bi dashboard ki?", "power_bi", "bangla"),
    ("tableau visualization", "tableau", "direct"),
    ("looker analytics ki?", "looker", "bangla"),
    ("excel tips and tricks", "excel", "direct"),
    ("google sheets formula ki?", "google_sheets", "bangla"),
    ("google docs tips", "google_docs", "direct"),
    ("powerpoint presentation tips", "powerpoint", "direct"),
    ("word processing basics", "word_processing", "direct"),
    ("pdf tools ki ki ache?", "pdf_tools", "bangla"),
    ("notion organization tips", "notion", "direct"),
    ("obsidian note taking ki?", "obsidian", "bangla"),
    ("evernote vs notion", "evernote", "comparison"),
    ("trello project management", "trello", "direct"),
    ("asana task management ki?", "asana", "bangla"),
    ("jira for teams ki?", "jira", "bangla"),
    ("slack tool ki?", "slack_tool", "bangla"),
    ("zoom tool tips", "zoom_tool", "direct"),
    ("teams collaboration ki?", "teams", "bangla"),
    ("discord server setup", "discord", "direct"),
    ("todoist task management", "todoist", "direct"),
    ("roam research ki?", "roam_research", "bangla"),
    ("bitbucket ki?", "bitbucket", "bangla"),

    # --- SECURITY ---
    ("ethical hacking ki?", "ethical_hacking", "bangla"),
    ("penetration testing basics", "penetration_testing", "direct"),
    ("bug bounty ki?", "bug_bounty", "bangla"),
    ("malware types ki ki?", "malware", "bangla"),
    ("ransomware attack ki?", "ransomware", "bangla"),
    ("phishing attack ki?", "phishing", "bangla"),
    ("social engineering ki?", "social_engineering", "bangla"),
    ("firewall ki?", "firewall", "bangla"),
    ("encryption basics", "encryption", "direct"),
    ("ssl tls ki?", "ssl_tls", "bangla"),
    ("network security ki?", "network_security", "bangla"),
    ("endpoint security ki?", "endpoint_security", "bangla"),
    ("password security tips", "password_security", "direct"),
    ("two factor authentication ki?", "two_factor_auth", "bangla"),
    ("identity theft prevention", "identity_theft", "direct"),
    ("data protection basics", "data_protection", "direct"),
    ("privacy tips online", "privacy", "direct"),
    ("online safety tips", "online_safety", "direct"),
    ("dark web ki?", "dark_web", "bangla"),
    ("tor browser ki?", "tor_browser", "bangla"),
    ("scam awareness tips", "scam_awareness", "direct"),
    ("proxy server ki?", "proxy", "bangla"),

    # --- SOCIAL MEDIA ---
    ("instagram tips for growth", "instagram_tips", "direct"),
    ("youtube tips for creators", "youtube_tips", "direct"),
    ("tiktok tips for viral", "tiktok_tips", "direct"),
    ("facebook tips for business", "facebook_tips", "direct"),
    ("twitter tips for engagement", "twitter_tips", "direct"),
    ("linkedin tips for job seekers", "linkedin_tips", "direct"),
    ("pinterest for business ki?", "pinterest", "bangla"),
    ("reddit tips for karma", "reddit_tips", "direct"),
    ("twitch streaming ki?", "twitch", "bangla"),

    # --- HEALTH PRACTICES ---
    ("ayurveda treatment ki?", "ayurveda", "bangla"),
    ("homeopathy ki kore?", "homeopathy", "bangla"),
    ("acupuncture ki?", "acupuncture", "bangla"),
    ("chiropractic treatment ki?", "chiropractic", "bangla"),
    ("naturopathy ki?", "naturopathy", "bangla"),
    ("physiotherapy ki?", "physiotherapy", "bangla"),
    ("aromatherapy ki?", "aromatherapy", "bangla"),
    ("essential oils uses ki?", "essential_oils", "bangla"),
    ("massage types and benefits", "massage", "direct"),
    ("spa treatments ki?", "spa", "bangla"),

    # --- EMERGENCY & SAFETY ---
    ("ambulance number ki?", "ambulance", "bangla"),
    ("emergency number ki?", "emergency_number", "bangla"),
    ("fire safety tips", "fire_safety", "direct"),
    ("earthquake safety tips", "earthquake_safety", "direct"),
    ("flood safety ki?", "flood_safety", "bangla"),
    ("hurricane safety tips", "hurricane_safety", "direct"),
    ("tornado safety ki?", "tornado_safety", "bangla"),
    ("tsunami warning ki?", "tsunami", "bangla"),
    ("volcanic eruption safety ki?", "volcanic_eruption", "bangla"),
    ("landslide safety tips", "landslide", "direct"),
    ("drought preparedness", "drought", "direct"),
    ("emergency planning ki?", "emergency_planning", "bangla"),
    ("road safety rules ki?", "road_safety", "bangla"),
    ("traffic rules ki?", "traffic_rules", "bangla"),
    ("driving license process ki?", "driving_license", "bangla"),
    ("car insurance ki?", "car_insurance", "bangla"),

    # --- BOT META ---
    ("tumi ki paro?", "bot_capability", "bangla"),
    ("what is your name?", "bot_name", "direct"),
    ("who made you?", "about_bot", "direct"),
    ("tell me a quote", "quotes", "direct"),
    ("blockchain ki?", "blockchain_basics", "bangla"),

    # --- MISCELLANEOUS TOPICS ---
    ("history of world war 2", "history", "direct"),
    ("philosophy ki?", "philosophy", "bangla"),
    ("psychology basics", "psychology", "direct"),
    ("sociology ki?", "sociology", "bangla"),
    ("anthropology ki?", "anthropology", "bangla"),
    ("archaeology ki?", "archaeology", "bangla"),
    ("economics basics", "economics", "direct"),
    ("macroeconomics ki?", "macroeconomics", "bangla"),
    ("microeconomics ki?", "microeconomics", "bangla"),
    ("behavioral economics ki?", "behavioral_economics", "bangla"),
    ("geography interesting facts", "geography", "direct"),
    ("cartography ki?", "cartography", "bangla"),
    ("geolocation technology ki?", "geolocation", "bangla"),
    ("maps history", "maps", "direct"),
    ("navigation basics", "navigation", "direct"),
    ("information science ki?", "information_science", "bangla"),
    ("library science ki?", "library_science", "bangla"),
    ("education system ki?", "education", "bangla"),

    # --- AGRICULTURE ---
    ("agriculture basics", "agriculture", "direct"),
    ("organic farming ki?", "organic_farming", "bangla"),
    ("hydroponics farming ki?", "hydroponics", "bangla"),
    ("aquaponics ki?", "aquaponics", "bangla"),
    ("vertical farming ki?", "vertical_farming", "bangla"),
    ("precision agriculture ki?", "precision_agriculture", "bangla"),
    ("permaculture ki?", "permaculture", "bangla"),
    ("mushroom farming ki?", "mushroom_farming", "bangla"),
    ("lab grown meat ki?", "lab_grown_meat", "bangla"),
    ("food tech innovations ki?", "food_tech", "bangla"),

    # --- MATERIALS & INDUSTRY ---
    ("steel industry ki?", "steel", "bangla"),
    ("aluminum uses ki?", "aluminum", "bangla"),
    ("copper applications ki?", "copper", "bangla"),
    ("gold investment ki?", "gold", "bangla"),
    ("diamond ki?", "diamond", "bangla"),
    ("gemstones types ki?", "gemstones", "bangla"),
    ("rubber production ki?", "rubber", "bangla"),
    ("glass manufacturing ki?", "glass", "bangla"),
    ("cement industry ki?", "cement", "bangla"),
    ("textile industry ki?", "textile", "bangla"),
    ("leather industry ki?", "leather", "bangla"),
    ("cotton industry ki?", "cotton_industry", "bangla"),
    ("silk production ki?", "silk_production", "bangla"),
    ("paper industry ki?", "paper_industry", "bangla"),
    ("plastic pollution ki?", "plastic_surgery", "bangla"),

    # --- PAYMENT & BANKING ---
    ("mobile banking ki?", "mobile_banking", "bangla"),
    ("digital wallet ki?", "digital_wallet", "bangla"),
    ("upi payment ki?", "upi_payment", "bangla"),
    ("remittance ki?", "remittance", "bangla"),
    ("hawala system ki?", "hawala", "bangla"),
    ("microfinance ki?", "microfinance", "bangla"),
    ("cooperative banking ki?", "cooperative", "bangla"),
    ("health insurance ki?", "health_insurance", "bangla"),
    ("student loan ki?", "student_loan", "bangla"),
    ("bankruptcy ki?", "bankruptcy", "bangla"),

    # --- PROJECT MANAGEMENT & METHODOLOGY ---
    ("project management basics", "project_management", "direct"),
    ("design patterns ki?", "design_patterns", "bangla"),
    ("system design ki?", "system_design", "bangla"),
    ("code review best practices", "code_review", "direct"),
    ("pair programming ki?", "pair_programming", "bangla"),
    ("clean code principles", "clean_code", "direct"),
    ("coding best practices", "coding", "direct"),
    ("competitive programming ki?", "competitive_programming", "bangla"),
    ("hackathon tips", "hackathon", "direct"),
    ("open source contribution", "open_source", "direct"),
    ("stackoverflow ki?", "stackoverflow", "bangla"),
    ("documentation writing ki?", "documentation", "bangla"),
    ("coding errors common ki?", "coding_errors", "bangla"),
    ("six sigma ki?", "six_sigma", "bangla"),
    ("lean manufacturing ki?", "lean_manufacturing", "bangla"),
    ("quality control ki?", "quality_control", "bangla"),

    # --- COMMUNICATION & SOFT SKILLS ---
    ("communication skills improve korbo", "communication", "bangla"),
    ("negotiation skills ki?", "negotiation", "bangla"),
    ("negotiation tactics ki ki?", "negotiation_tactics", "bangla"),
    ("conflict resolution ki?", "conflict_resolution", "bangla"),
    ("teamwork er importance ki?", "teamwork", "bangla"),
    ("decision making tips", "decision_making", "direct"),
    ("critical thinking ki?", "critical_thinking", "bangla"),
    ("creative thinking ki?", "creative_thinking", "bangla"),
    ("problem solving techniques", "problem_solving", "direct"),
    ("brainstorming session tips", "brainstorming", "direct"),
    ("mind mapping ki?", "mind_mapping", "bangla"),
    ("note taking tips", "note_taking", "direct"),
    ("speed reading ki?", "speed_reading", "bangla"),
    ("memory techniques ki?", "memory_techniques", "bangla"),
    ("etiquette basics ki?", "etiquette", "bangla"),
    ("body language tips", "body_language", "direct"),

    # --- SPIRITUAL & MYSTICAL ---
    ("astrology ki real?", "astrology", "bangla-mix"),
    ("numerology ki?", "numerology", "bangla"),
    ("tarot card reading ki?", "tarot", "bangla"),
    ("palmistry ki?", "palmistry", "bangla"),
    ("feng shui home tips", "feng_shui", "direct"),
    ("vastu shastra ki?", "vastu", "bangla"),
    ("superstitions ki?", "superstitions", "bangla"),
    ("dream interpretation ki?", "dream_interpretation", "bangla"),
    ("lucid dreaming ki?", "lucid_dreaming", "bangla"),
    ("mythology interesting stories", "mythology", "direct"),
    ("folklore stories ki?", "folklore", "bangla"),
    ("fairy tales for kids", "fairy_tales", "direct"),
    ("urban legends ki?", "urban_legends", "bangla"),
    ("conspiracy theories ki?", "conspiracy_theories", "bangla"),
    ("paranormal ki?", "paranormal", "bangla"),
    ("ufo sightings ki real?", "ufo", "bangla-mix"),
    ("aliens exist kore ki?", "aliens", "bangla-mix"),
    ("time travel ki possible?", "time_travel", "bangla-mix"),
    ("parallel universe ki?", "parallel_universe", "bangla"),
    ("simulation theory ki?", "simulation_theory", "bangla"),
    ("transhumanism ki?", "transhumanism", "bangla"),
    ("artificial consciousness ki?", "artificial_consciousness", "bangla"),

    # --- SOCIAL LIFE & MISC ---
    ("how to make friends?", "friendship", "direct"),
    ("goodbye message ki likhi?", "farewell", "bangla"),
    ("greeting messages ideas", "greeting", "direct"),
    ("thanks message ki likhi?", "thanks", "bangla"),
    ("best books for reading", "books", "direct"),
    ("book club ki?", "book_club", "bangla"),
    ("audiobook recommendations", "audiobook", "direct"),
    ("ebook readers best", "ebook", "direct"),
    ("library ki?", "library", "bangla"),
    ("digital library ki?", "digital_library", "bangla"),
    ("life advice general", "life", "direct"),
    ("general knowledge quiz", "general", "direct"),
    ("news sources best", "news", "direct"),
    ("entertainment recommendations", "entertainment", "direct"),
    ("trivia facts interesting", "trivia", "direct"),

    # --- MOTIVATION ---
    ("daily motivation quotes", "motivation_daily", "direct"),
    ("strong motivation speech", "motivation_strong", "direct"),

    # --- FOOD SPECIAL DIETS ---
    ("keto diet ki?", "keto_diet", "bangla"),
    ("intermittent fasting ki?", "intermittent_fasting", "bangla"),
    ("veganism ki?", "veganism", "bangla"),
    ("veganism ethics ki?", "veganism_ethics", "bangla"),
    ("vegetarian diet benefits", "vegetarian", "direct"),
    ("supplements ki ki nibo?", "supplements", "bangla"),
    ("creatine ki safe?", "creatine", "bangla-mix"),
    ("pre workout drink ki?", "pre_workout", "bangla"),
    ("post workout nutrition", "post_workout", "direct"),
    ("nootropics ki?", "nootropics", "bangla"),
    ("weight loss tips quick", "weight_loss", "direct"),
    ("weight gain tips healthy", "weight_gain", "direct"),

    # --- EMERGING TECH ---
    ("metaverse ki?", "metaverse", "bangla"),
    ("web3 ki?", "web3", "bangla"),
    ("nft art ki?", "nft_art", "bangla"),
    ("augmented reality ki?", "augmented_reality", "bangla"),
    ("virtual reality gaming", "virtual_reality", "direct"),
    ("blockchain basics", "blockchain_basics", "direct"),

    # --- CLOUD SERVICES ---
    ("iaas ki?", "iaas", "bangla"),
    ("paas ki?", "paas", "bangla"),
    ("saas ki?", "saas", "bangla"),
    ("gcp google cloud ki?", "gcp", "bangla"),

    # --- MISC TOPICS ---
    ("heritage preservation ki?", "heritage", "bangla"),
    ("culture shock ki?", "culture_shock", "bangla"),
    ("comic books recommendations", "comic_books", "direct"),
    ("cosplay ki?", "cosplay", "bangla"),
    ("magic tricks for beginners", "magic_tricks", "direct"),
    ("juggling ki?", "juggling", "bangla"),
    ("circus history ki?", "circus", "bangla"),
    ("clown ki?", "clown", "bangla"),
    ("puppetry ki?", "puppetry", "bangla"),
    ("ventriloquism ki?", "ventriloquism", "bangla"),
    ("improv comedy ki?", "improv", "bangla"),
    ("mime art ki?", "mime", "bangla"),
    ("sketching tips for beginners", "sketching", "direct"),
    ("art history interesting", "art_history", "direct"),
    ("art collection ki?", "art_collection", "bangla"),
    ("antiques collection ki?", "antiques", "bangla"),
    ("auction bidding ki?", "auction", "bangla"),
    ("geocaching ki?", "geocaching", "bangla"),
    ("orienteering ki?", "orienteering", "bangla"),
    ("biohacking ki?", "biohacking", "bangla"),
    ("blood donation ki?", "blood_donation", "bangla"),
    ("organ donation ki?", "organ_donation", "bangla"),
    ("vaccination importance ki?", "vaccination", "bangla"),
    ("telemedicine ki?", "telemedicine", "bangla"),
    ("medical tourism ki?", "medical_tourism", "bangla"),
    ("medical devices ki?", "medical_devices", "bangla"),
    ("medical ethics ki?", "medical_ethics", "bangla"),
    ("pharmacy basics ki?", "pharmacy", "bangla"),
    ("nursing career ki?", "nursing", "bangla"),
    ("hospital management ki?", "hospital_management", "bangla"),
    ("healthcare system ki?", "healthcare_system", "bangla"),
    ("public health ki?", "public_health", "bangla"),
    ("pandemic preparedness ki?", "pandemic", "bangla"),
    ("epidemic ki?", "epidemic", "bangla"),

    # --- URBAN & PLANNING ---
    ("urban planning ki?", "urban_planning", "bangla"),
    ("green building ki?", "green_building", "bangla"),
    ("public transport ki?", "public_transport", "bangla"),
    ("smart grid ki?", "smart_grid", "bangla"),

    # --- MANAGEMENT & HR ---
    ("risk management ki?", "risk_management", "bangla"),
    ("crisis management ki?", "crisis_management", "bangla"),
    ("event planning tips", "event_planning", "direct"),
    ("knowledge management ki?", "knowledge_management", "bangla"),
    ("workplace culture ki?", "workplace_culture", "bangla"),
    ("workplace harassment ki?", "workplace_harassment", "bangla"),
    ("loyalty program ki?", "loyalty_program", "bangla"),
    ("distribution channels ki?", "distribution", "bangla"),
    ("telemarketing ki?", "telemarketing", "bangla"),
    ("door to door sales ki?", "door_to_door", "bangla"),

    # --- MISC TECHNOLOGY ---
    ("android tips and tricks", "android", "direct"),
    ("ios tips for iphone", "ios", "direct"),
    ("mobile design best practices", "mobile_design", "direct"),
    ("accessibility in web design", "accessibility", "direct"),
    ("automation tools ki?", "automation", "bangla"),
    ("cnc machining ki?", "cnc", "bangla"),
    ("laser cutting ki?", "laser_cutting", "bangla"),
    ("3d printing projects", "3d_printing", "direct"),
    ("telescope for astronomy", "telescope", "direct"),
    ("satellite technology ki?", "satellite", "bangla"),
    ("drone regulations ki?", "drone", "bangla"),

    # --- HOBBY & LEISURE ---
    ("swimming pool maintenance", "swimming", "direct"),
    ("surfing lessons where?", "surfing", "direct"),
    ("fishing tips for beginners", "fishery", "direct"),
    ("photography equipment ki lagbe?", "photography", "bangla"),
    ("painting supplies for beginners", "painting", "direct"),
    ("cooking class online", "cooking", "direct"),

    # --- UNIQUE MISC ---
    ("procrastination ki?", "procrastination", "bangla"),
    ("gratitude practice ki?", "gratitude", "bangla"),
    ("journaling benefits ki?", "journaling", "bangla"),
    ("debate skills improve korbo", "debate", "bangla"),
    ("diplomacy ki?", "diplomacy", "bangla"),
    ("immigration process ki?", "immigration", "bangla"),
    ("refugee rights ki?", "human_rights", "bangla"),
    ("shopping tips smart", "shopping", "direct"),
    ("food ki?", "food", "bangla"),
    ("sports general", "sports", "direct"),

    # --- UNIQUE TOPICS CONTINUED ---
    ("rural life ki?", "rural_life", "bangla"),
    ("mobility solutions ki?", "mobility", "bangla"),
    ("game design basics", "game_design", "direct"),
    ("level design ki?", "level_design", "bangla"),
    ("unity game development", "unity_game", "direct"),
    ("unreal engine ki?", "unreal_engine", "bangla"),
    ("blender 3d modeling", "blender_3d", "direct"),
    ("simulation in gaming ki?", "simulation", "bangla"),

    # --- ADDITIONAL TYPO QUESTIONS ---
    ("hwo to lern pytohn?", "python", "typo"),
    ("javscript tutorail for bignners", "javascript", "typo"),
    ("rreact hooks tutoriel", "react", "typo"),
    ("waht is mahcine lerning?", "ml", "typo"),
    ("how to shourdown laptop?", "technology", "typo"),
    ("whats artifical inteligence?", "ai", "typo"),
    ("databaes managment sysytem ki?", "database", "typo"),

    # --- ADDITIONAL INFORMAL & SHORT ---
    ("bro how to code", "coding", "informal"),
    ("tell me about ai", "ai", "informal"),
    ("yo whats blockchain?", "blockchain_basics", "informal"),
    ("dude explain git to me", "git", "informal"),
    ("hey what is cloud?", "cloud", "informal"),
    ("sql tips", "sql", "short"),
    ("git basics", "git", "short"),
    ("css tips", "html_css", "short"),
    ("api basics", "api", "short"),

    # --- ADDITIONAL COMPARISON ---
    ("react vs angular", "react", "comparison"),
    ("python vs java", "python", "comparison"),
    ("messi vs ronaldo", "football", "opinion"),
    ("aws vs azure vs gcp", "aws", "comparison"),
    ("docker vs kubernetes", "docker_basics", "comparison"),

    # --- ADDITIONAL MULTI-TOPIC ---
    ("python flask deployment", "flask_framework", "multi-topic"),
    ("machine learning with python", "ml", "multi-topic"),
    ("docker kubernetes deployment", "kubernetes", "multi-topic"),

    # --- MORE BANGLA ---
    ("python ki?", "python", "bangla"),
    ("ami depressed", "depression_support", "bangla-emotion"),
    ("kemon achen?", "greeting", "bangla"),
    ("dhonnobad", "thanks", "bangla"),
    ("ghumote parchi na", "insomnia", "bangla-emotion"),

    # --- ADDITIONAL UNIQUE CATEGORIES ---
    ("surveying basics ki?", "surveying", "bangla"),
    ("archive management ki?", "archive", "bangla"),
    ("museum visit tips", "museum", "direct"),
    ("museum curation ki?", "museum_curation", "bangla"),
    ("peer review process ki?", "peer_review", "bangla"),
    ("competitive programming tips", "competitive_programming", "direct"),
    ("projects portfolio ideas", "projects", "direct"),
    ("ikigai ki?", "life_coaching", "bangla"),
    ("biometric security ki?", "security", "bangla"),
    ("linkedin profile tips", "linkedin", "direct"),
    ("chef devops ki?", "chef_devops", "bangla"),
    ("kickstarter campaigns ki?", "kickstarter", "bangla"),
    ("factory farming ki?", "factory_farming", "bangla"),
    ("pilgrimage destinations ki?", "pilgrimage", "bangla"),
]


def run_test():
    """Run all 1000 questions and analyze results."""
    fast_mode = "--fast" in sys.argv
    sys.path.insert(0, ".")

    if fast_mode:
        import chatbot as cb
        def _skip_llm(self, *a, **kw):
            self.model = None
            self.available = False
        cb.TinyLlamaGenerator.__init__ = _skip_llm

    from chatbot import ChatBot

    print("=" * 70)
    print("  1000 QUESTIONS TEST")
    print("=" * 70)
    if fast_mode:
        print("  Mode: FAST (LLM skipped)")
    else:
        print("  Mode: FULL (LLM ON — real bot answers)")
        print("  Note: First load takes ~60-70s, then ~3-8s per question")
    print("\nLoading chatbot...", flush=True)

    load_start = time.time()
    bot = ChatBot()
    load_time = time.time() - load_start
    session_id = bot.db.create_session()
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  LLM: {bot.generator.model_name if bot.generator.available else 'OFF'}")
    print(f"  Questions: {len(bot.questions)}")
    print(f"  Categories: {len(bot.category_store_map)}")

    results = []
    correct = 0
    wrong = 0
    wrong_details = []
    total_time = 0
    output_file = "test_1000_results.json"

    def save_progress():
        """Save results after every question so nothing is lost."""
        avg = (total_time / len(results)) if results else 0
        tag_stats = {}
        for r in results:
            t = r["tag"]
            if t not in tag_stats:
                tag_stats[t] = {"total": 0, "correct": 0}
            tag_stats[t]["total"] += 1
            if r["correct"]:
                tag_stats[t]["correct"] += 1
        output = {
            "total_questions": len(TEST_QUESTIONS),
            "completed": len(results),
            "correct": correct,
            "wrong": wrong,
            "accuracy": round(correct / len(results), 4) if results else 0,
            "total_time_s": round(total_time, 1),
            "avg_time_ms": round(avg * 1000),
            "llm_enabled": not fast_mode and llm_status == "ON",
            "by_tag": {t: {"total": s["total"], "correct": s["correct"],
                            "accuracy": round(s["correct"]/s["total"], 4)}
                       for t, s in tag_stats.items()},
            "wrong_details": wrong_details,
            "all_results": results
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nRunning {len(TEST_QUESTIONS)} questions...\n")

    for i, (question, expected, tag) in enumerate(TEST_QUESTIONS, 1):
        start = time.time()
        result = bot.get_answer(question, session_id)
        elapsed = time.time() - start
        total_time += elapsed

        if result is None:
            got_cat = "UNKNOWN"
            confidence = 0
            reply = ""
            generated = False
        elif result.get("suggestions"):
            got_cat = "SUGGESTIONS"
            confidence = 0
            reply = ""
            generated = False
        else:
            got_cats = result.get("categories", [])
            got_cat = got_cats[0] if got_cats else result.get("intent", "???")
            confidence = result.get("confidence", 0)
            reply = result.get("reply", "")
            generated = result.get("generated", False)

        # Check if correct
        is_correct = (got_cat.lower().strip() == expected.lower().strip())
        if not is_correct and result and result.get("categories"):
            is_correct = expected.lower() in [c.lower() for c in result.get("categories", [])]

        if is_correct:
            correct += 1
            status_icon = "+"
        else:
            wrong += 1
            status_icon = "X"
            wrong_details.append({
                "num": i,
                "question": question,
                "expected": expected,
                "got": got_cat,
                "all_categories": result.get("categories", []) if result else [],
                "confidence": confidence,
                "reply": reply,
                "tag": tag,
                "time_ms": round(elapsed * 1000),
                "generated": generated
            })

        # Print like real chat
        conf_pct = f"{confidence:.0%}" if confidence else "---"
        gen_tag = " [LLM]" if generated else ""
        time_str = f"{elapsed:.1f}s"
        acc_pct = f"{correct/i:.0%}"

        print(f"  [{status_icon}] Q{i:>3}/{len(TEST_QUESTIONS)}: {question}")
        if reply:
            short_reply = reply.replace("\n", " ")
            if len(short_reply) > 120:
                short_reply = short_reply[:120] + "..."
            print(f"         Bot: {short_reply}")
        elif got_cat == "SUGGESTIONS":
            print(f"         Bot: (Did you mean? suggestions shown)")
        else:
            print(f"         Bot: (no answer)")
        print(f"         [{got_cat}] {conf_pct}{gen_tag} | {time_str} | Running: {acc_pct}")
        if not is_correct:
            print(f"         Expected: {expected}")
        print()

        results.append({
            "num": i,
            "question": question,
            "expected": expected,
            "got": got_cat,
            "correct": is_correct,
            "confidence": confidence,
            "reply": reply,
            "tag": tag,
            "time_ms": round(elapsed * 1000),
            "generated": generated
        })

        # Save after every question
        save_progress()

    # ============ SUMMARY ============
    avg_time = total_time / len(TEST_QUESTIONS)
    print("=" * 70)
    print(f"  RESULTS: {correct}/{len(TEST_QUESTIONS)} correct ({correct/len(TEST_QUESTIONS):.0%})")
    print(f"  Wrong: {wrong}")
    print(f"  Total time: {total_time:.1f}s | Avg: {avg_time:.2f}s/question")
    print("=" * 70)

    # Break down by tag
    tag_stats = {}
    for r in results:
        t = r["tag"]
        if t not in tag_stats:
            tag_stats[t] = {"total": 0, "correct": 0}
        tag_stats[t]["total"] += 1
        if r["correct"]:
            tag_stats[t]["correct"] += 1

    print("\n--- Accuracy by Question Type ---")
    print(f"{'Type':<20} {'Score':<15} {'Accuracy'}")
    print("-" * 50)
    for t, stats in sorted(tag_stats.items(), key=lambda x: x[1]["correct"]/x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        bar = "#" * round(acc * 10) + "." * (10 - round(acc * 10))
        print(f"{t:<20} {stats['correct']}/{stats['total']:<12} {bar} {acc:.0%}")

    # Wrong answers detail
    if wrong_details:
        print(f"\n{'=' * 70}")
        print(f"  WRONG ANSWERS ({len(wrong_details)})")
        print(f"{'=' * 70}\n")

        for w in wrong_details:
            print(f"  Q{w['num']}: \"{w['question']}\"")
            print(f"     Expected: {w['expected']}")
            print(f"     Got:      {w['got']} (all: {w['all_categories']})")
            print(f"     Conf:     {w['confidence']:.0%}")
            short = w['reply'].replace('\n', ' ')[:100] if w['reply'] else "(none)"
            print(f"     Reply:    {short}")
            gen = " [LLM generated]" if w.get('generated') else ""
            print(f"     Type:     {w['tag']}{gen}")
            print()

    # Final save (already saved incrementally, this is the final version)
    save_progress()
    print(f"\nResults saved to: {output_file}")
    return


if __name__ == "__main__":
    run_test()
