
# --- Cell 1: imports & setup ---
import streamlit as st
import re, string
from pathlib import Path
import pandas as pd
import numpy as np
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# Lightweight sentiment (VADER via NLTK)
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

pd.set_option('display.max_colwidth', 200)
random.seed(42)
np.random.seed(42)
print("Imports complete (lightweight + sentiment ready)")


# In[144]:


# --- Cell 2: text cleaning ---
def clean_text(text):
    """Lowercase, remove punctuation, collapse whitespace."""
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# quick sanity check
print(clean_text("Hello!! This is GREAT :-)  \n"))


# In[145]:


# --- Cell 3: Load datasets ---
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Intents dataset (must include: text, intent, emotion)
df_intents = pd.read_csv(DATA_DIR / "intents.txt", sep="\t")
for c in ["text", "intent", "emotion"]:
    if c not in df_intents.columns:
        raise ValueError("intents.txt must include columns: text, intent, emotion")
df_intents["text"] = df_intents["text"].astype(str)
df_intents["intent"] = df_intents["intent"].astype(str)
df_intents["emotion"] = df_intents["emotion"].astype(str)

# FAQ dataset
df_faq = pd.read_csv(DATA_DIR / "restaurant_faq.csv")
df_faq.columns = [c.lower() for c in df_faq.columns]
for c in ["question", "answer"]:
    if c not in df_faq.columns:
        raise ValueError("restaurant_faq.csv must include columns: Question, Answer (or lowercase)")

# Menu dataset
df_menu = pd.read_csv(DATA_DIR / "menu.csv")
expected_cols = ['name', 'ingredients', 'diet', 'cook_time', 'flavor_profile', 'course']
for col in expected_cols:
    if col not in df_menu.columns:
        raise ValueError(f"menu.csv missing column: {col}")

print(f"Intents: {len(df_intents)}, FAQ: {len(df_faq)}, Menu: {len(df_menu)}")


# In[155]:


# --- Cell 4: Preprocess & Train (different process: CountVectorizer + MultinomialNB) ---
texts = df_intents['text'].apply(clean_text).values
intent_labels = df_intents['intent'].values
emotion_labels = df_intents['emotion'].values

X_train, X_test, y_intent_train, y_intent_test, y_emotion_train, y_emotion_test = train_test_split(
    texts, intent_labels, emotion_labels, test_size=0.2, random_state=42, stratify=intent_labels
)

# Use CountVectorizer (different from TF-IDF + Logistic Regression used by your teammate)
vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# Multinomial Naive Bayes (different model choice)
clf_intent  = MultinomialNB().fit(X_train_vec, y_intent_train)
clf_emotion = MultinomialNB().fit(X_train_vec, y_emotion_train)

print("Intent Accuracy:",  round(accuracy_score(y_intent_test,  clf_intent.predict(X_test_vec)),  4))
print("Emotion Accuracy:", round(accuracy_score(y_emotion_test, clf_emotion.predict(X_test_vec)), 4))


# In[156]:


# --- Cell 5: Save models ---
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)

joblib.dump(clf_intent,  MODEL_PATH / "intent_classifier_nb.joblib")
joblib.dump(clf_emotion, MODEL_PATH / "emotion_classifier_nb.joblib")
joblib.dump(vectorizer,   MODEL_PATH / "count_vectorizer.joblib")
print("Models and vectorizer saved.")


# In[152]:


# --- Cell 6: FAQ retriever (TF-IDF + cosine similarity + threshold) ---
df_faq['clean_question'] = df_faq['question'].astype(str).apply(clean_text)
faq_vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
faq_matrix = faq_vectorizer.fit_transform(df_faq['clean_question'])

def search_faq(user_text, top_k=3, threshold=0.35):
    """Return top-k FAQ matches above similarity threshold."""
    user_clean = clean_text(user_text)
    user_vec = faq_vectorizer.transform([user_clean])
    sims = cosine_similarity(user_vec, faq_matrix)[0]
    idx_sorted = sims.argsort()[::-1]

    results = []
    for i in idx_sorted[:top_k]:
        if sims[i] >= threshold:
            results.append({
                "question": df_faq.iloc[i]["question"],
                "answer":   df_faq.iloc[i]["answer"],
                "score":    float(sims[i])
            })
    return results


# In[153]:


# --- Cell 7: menu search (robust + keyword-aware) ---
# Normalize important text columns
for col in ['name', 'ingredients', 'diet', 'flavor_profile', 'course']:
    if col in df_menu.columns:
        df_menu[col] = df_menu[col].astype(str).apply(clean_text)
    else:
        df_menu[col] = ""  # guard just in case

# Keyword pools
DIET_KEYWORDS = {
    "vegan": ["vegan"],
    "vegetarian": ["vegetarian", "veg"],
    "non_vegetarian": ["non vegetarian", "non-vegetarian", "chicken", "meat", "fish", "egg"],
    "gluten_free": ["gluten free", "gluten-free", "no gluten"]
}
FLAVOR_KEYWORDS = ["spicy", "sweet", "savory", "sour", "bitter", "mild", "hot"]

def _contains_any(text: str, words) -> bool:
    return any(w in text for w in words)

def search_menu(user_text: str, top_k: int = 5):
    """Heuristic menu search: diet → flavor → ingredients/name tokens."""
    t = clean_text(user_text)

    # 1) Diet filter
    if _contains_any(t, DIET_KEYWORDS["vegan"]):
        hits = df_menu[df_menu["diet"].str.contains("vegan")]
        if not hits.empty:
            return hits.head(top_k).to_dict("records")

    if _contains_any(t, DIET_KEYWORDS["vegetarian"]) and "non vegetarian" not in t:
        hits = df_menu[df_menu["diet"].str.contains("vegetarian")]
        if not hits.empty:
            return hits.head(top_k).to_dict("records")

    if _contains_any(t, DIET_KEYWORDS["non_vegetarian"]):
        hits = df_menu[df_menu["diet"].str.contains("non vegetarian")]
        if not hits.empty:
            return hits.head(top_k).to_dict("records")

    # 2) Flavor match
    for flavor in FLAVOR_KEYWORDS:
        if flavor in t:
            hits = df_menu[df_menu["flavor_profile"].str.contains(flavor)]
            if not hits.empty:
                return hits.head(top_k).to_dict("records")

    # 3) Token overlap on ingredients / name
    tokens = set(t.split())
    scored = []
    for _, row in df_menu.iterrows():
        text_blob = f"{row['name']} {row['ingredients']}"
        menu_tokens = set(text_blob.split())
        overlap = len(tokens & menu_tokens)
        if overlap > 0:
            scored.append((overlap, row.to_dict()))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    # 4) Fallback: name substring
    if len(t) >= 3:
        hits = df_menu[df_menu["name"].str.contains(t)]
        if not hits.empty:
            return hits.head(top_k).to_dict("records")

    return []


# In[154]:


# --- Cell 8: prediction & sentiment helpers (with "thanks" fast-path) ---
def predict_intent(user_text: str):
    vec = vectorizer.transform([clean_text(user_text)])
    probs = clf_intent.predict_proba(vec)[0]
    idx = int(np.argmax(probs))
    return clf_intent.classes_[idx], float(probs[idx])

def predict_emotion(user_text: str):
    vec = vectorizer.transform([clean_text(user_text)])
    probs = clf_emotion.predict_proba(vec)[0]
    idx = int(np.argmax(probs))
    return clf_emotion.classes_[idx], float(probs[idx])

def get_sentiment(user_text: str):
    s = sia.polarity_scores(user_text)
    comp = s["compound"]
    if comp >= 0.05:
        label = "positive"
    elif comp <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, comp

def format_menu_results(records):
    if not records:
        return "I couldn't find matching menu items."
    lines = []
    for r in records:
        name = r.get("name", "Unknown").title()
        diet = r.get("diet", "").strip()
        flavor = r.get("flavor_profile", "").strip()
        ings = r.get("ingredients", "").strip()
        parts = []
        if diet: parts.append(diet)
        if flavor: parts.append(flavor)
        meta = f" ({', '.join(parts)})" if parts else ""
        lines.append(f"- {name}{meta} — Ingredients: {ings}")
    return "Here are some suggestions:\n" + "\n".join(lines)

THANKS_PATTERNS = (
    "thanks", "thank you", "appreciate", "thx", "ty", "many thanks"
)
GREET_PATTERNS = (
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening"
)

def is_thanks(text: str) -> bool:
    t = clean_text(text)
    return any(p in t for p in THANKS_PATTERNS)

def is_greeting(text: str) -> bool:
    t = clean_text(text)
    return any(t.startswith(p) or f" {p} " in f" {t} " for p in GREET_PATTERNS)


# In[157]:


# --- Cell 9: get_bot_response (routing + FAQ + menu + sentiment-aware) ---
def get_bot_response(user_text: str, confidence_threshold: float = 0.35) -> str:
    # Fast-path: explicit thanks
    if is_thanks(user_text):
        return random.choice([
            "You're welcome! 😊",
            "Anytime! Happy to help.",
            "You're most welcome."
        ])

    # Sentiment & soft empathy
    sent_label, sent_score = get_sentiment(user_text)
    empathy = ""
    if sent_label == "negative":
        empathy = "I'm really sorry you're feeling this way. "
    elif sent_label == "positive":
        empathy = "Glad to hear that! "

    # Model predictions
    intent_label, intent_prob = predict_intent(user_text)

    # Low-confidence → try FAQ, then Menu, then fallback
    if intent_prob < confidence_threshold:
        hits = search_faq(user_text, top_k=1, threshold=0.3)
        if hits:
            return empathy + hits[0]["answer"]
        menu_hits = search_menu(user_text, top_k=5)
        if menu_hits:
            return empathy + format_menu_results(menu_hits)
        return empathy + "Sorry, I didn't quite get that. You can ask about our **menu** or any **FAQ** like hours, parking, or reservations."

    # High-confidence routing
    if intent_label.startswith("faq"):
        hits = search_faq(user_text, top_k=1, threshold=0.3)
        if hits:
            return empathy + hits[0]["answer"]
        return empathy + "I recognized this as an FAQ, but I couldn't find a close match. Could you rephrase it?"

    if intent_label.startswith("menu"):
        menu_hits = search_menu(user_text, top_k=5)
        if menu_hits:
            return empathy + format_menu_results(menu_hits)
        return empathy + "I understood you're asking about the menu, but I couldn't find matching items."

    if intent_label == "greeting":
        return random.choice([
            "Hi there 👋 How can I help you today?",
            "Hello! How can I assist you—menu suggestions or FAQs?",
            "Hey! What would you like to know?"
        ])

    if intent_label == "farewell":
        return random.choice([
            "Goodbye — have a wonderful day!",
            "See you next time!",
            "Take care! 👋"
        ])

    # For "other" or anything else, reflect emotion if classifier captured it
    emotion_label, _ = predict_emotion(user_text)
    emotion_word = emotion_label.replace("none", "neutral").replace("_", " ")
    return empathy + f"I got you. If you need help with our **menu** or **reservations**, just ask! (Detected emotion: {emotion_word})"


# In[158]:


# --- Cell 10: Streamlit chat UI (with first AI message) ---
st.set_page_config(page_title="Restaurant Chatbot", page_icon="🍽️", layout="centered")
st.title("🍽️ Restaurant Chatbot (Lightweight + Sentiment)")

# Keep a simple chat history
if "chat" not in st.session_state:
    st.session_state.chat = []
    # First AI message (AI-style greeting)
    st.session_state.chat.append(("bot", "Hi there 👋 How can I help you today?"))

# Render chat so far
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

# Input box
user_input = st.text_input("Type your message and press Enter:", "", key="user_input_box")

if user_input:
    st.session_state.chat.append(("user", user_input))
    bot_reply = get_bot_response(user_input)
    st.session_state.chat.append(("bot", bot_reply))
    st.experimental_rerun()  # refresh to show the new messages immediately


# In[159]:


# --- Cell 11: Quick local tests (stdout) ---
test_inputs = [
    "Hi!",
    "Thanks for the help",
    "Do you have outdoor seating?",
    "What time do you open?",
    "I want something spicy",
    "I'm really unhappy with my order",
    "Goodbye!"
]
print("\n--- Quick Tests ---")
for t in test_inputs:
    print(f"User: {t}")
    print(f"Bot : {get_bot_response(t)}")
    print("-" * 40)

