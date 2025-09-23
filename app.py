# app.py
# Gradio app for Movie Review Sentiment Analysis demo (bar chart)
import os, re
import joblib
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import gradio as gr

# Ensure NLTK resources
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

STOP_WORDS = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

ROOT = os.getcwd()
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "lr_clf.joblib")
VECT_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w and w not in STOP_WORDS]
    return " ".join(tokens)

# Safe wordcloud function (not used by the interface but kept)
def safe_wordcloud(text, outpath=None):
    if not text or not text.strip():
        return None
    wc = WordCloud(width=800, height=300, background_color="white").generate(text)
    if outpath:
        wc.to_file(outpath)
    return wc

# Try to load pre-saved models
lr_clf = None
vectorizer = None
if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
    try:
        lr_clf = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECT_FILE)
        print("Loaded models from models/ folder.")
    except Exception as e:
        print("Failed to load pre-saved models:", e)
        lr_clf = None
        vectorizer = None

# Fallback trainer (only used if no saved models)
def train_fallback(sample_size=8000):
    from datasets import load_dataset
    ds = load_dataset("imdb")
    df_train = pd.DataFrame({"review": ds["train"]["text"], "label": ds["train"]["label"]})
    df_train["sentiment"] = df_train["label"].map(lambda x: "positive" if x == 1 else "negative")
    df_small = df_train.sample(n=min(sample_size, len(df_train)), random_state=42).reset_index(drop=True)
    df_small["cleaned"] = df_small["review"].apply(clean_text)
    X = df_small["cleaned"].values
    y = df_small["sentiment"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    vect = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
    X_train_vec = vect.fit_transform(X_train)
    X_test_vec = vect.transform(X_test)
    lr = LogisticRegression(solver="liblinear", max_iter=1000)
    lr.fit(X_train_vec, y_train)
    y_pred = lr.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Fallback model accuracy: {acc:.4f}")
    try:
        joblib.dump(lr, MODEL_FILE)
        joblib.dump(vect, VECT_FILE)
        print("Saved fallback model to models/")
    except Exception:
        pass
    return lr, vect

if lr_clf is None or vectorizer is None:
    try:
        lr_clf, vectorizer = train_fallback(sample_size=8000)
    except Exception as e:
        raise RuntimeError("Model unavailable and auto-train failed. Ensure 'datasets' in requirements or upload models.") from e

feature_names = vectorizer.get_feature_names_out()
coefs = lr_clf.coef_[0]

def predict_and_explain(text: str):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    probs = lr_clf.predict_proba(vec)[0]
    classes = list(lr_clf.classes_)  # typically ['negative','positive']
    prob_dict = dict(zip(classes, probs))

    vader = sia.polarity_scores(text)
    compound = vader["compound"]

    max_prob = max(probs)
    uncertain = max_prob < 0.60
    is_neutral = (-0.05 <= compound <= 0.05) or uncertain

    if is_neutral:
        label = "neutral"
    else:
        label = classes[int(np.argmax(probs))]

    arr = vec.toarray()[0]
    present_idx = np.where(arr > 0)[0]
    contribs = []
    if present_idx.size > 0:
        for idx in present_idx:
            contribs.append((feature_names[idx], float(coefs[idx])))
        contribs = sorted(contribs, key=lambda x: x[1], reverse=True)[:8]
    else:
        contribs = []

    pos = prob_dict.get("positive", 0.0) * 100
    neg = prob_dict.get("negative", 0.0) * 100
    neutral_pct = max(0.0, 100.0 - max(pos, neg)) if is_neutral else 0.0

    percentages = {"positive": pos, "negative": neg, "neutral": neutral_pct}
    prob_str = f"P: {pos:.1f}%, N: {neg:.1f}%, Neutral(est): {neutral_pct:.1f}%"

    return {"label": label, "probabilities": percentages, "top_words": contribs, "vader": vader, "prob_str": prob_str}

def make_bar_fig(percentages):
    fig, ax = plt.subplots(figsize=(4, 2.6))
    labels = list(percentages.keys())
    values = [percentages[k] for k in labels]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage %")
    ax.set_title("Sentiment Distribution")
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    return fig

def gr_predict(text):
    info = predict_and_explain(text)
    label = info["label"].capitalize()
    words = ", ".join([f"{w}({round(c,3)})" for w, c in info["top_words"]]) if info["top_words"] else "N/A"
    probstr = info["prob_str"]
    fig = make_bar_fig(info["percentages"])
    return f"Predicted Sentiment: {label} | {probstr}", f"Top contributing words: {words}", fig

title = "Movie Review Sentiment Analysis"
desc = (
    "Type a movie review and see predicted sentiment (Positive/Negative/Neutral). "
    "Uses Logistic Regression (TF-IDF) + VADER for neutral detection. Top contributing words shown."
)

iface = gr.Interface(
    fn=gr_predict,
    inputs=gr.Textbox(lines=4, placeholder="Type a movie review here..."),
    outputs=[gr.Textbox(), gr.Textbox(), gr.Plot()],
    title=title,
    description=desc,
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch(share=False)
