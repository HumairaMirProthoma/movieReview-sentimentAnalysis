# app.py
# Movie Review Sentiment Analysis (Transformer prediction + TF-IDF explainability)
# - Uses a Hugging Face transformer model (3-class) for prediction (better negation/context)
# - Uses TF-IDF + LogisticRegression (if present) to compute top contributing words for explainability
# - Robust error handling so HF Space doesn't crash
# - Outputs: (text summary, top-words string, matplotlib bar chart)

import os, re, traceback
import numpy as np
import matplotlib.pyplot as plt
import joblib

# external libs (installed via requirements.txt)
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
except Exception:
    pipeline = None  # we'll handle missing transformer gracefully

import gradio as gr

# ----------------- config -----------------
ROOT = os.getcwd()
MODEL_DIR = os.path.join(ROOT, "models")
TFIDF_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
LR_FILE = os.path.join(MODEL_DIR, "lr_clf.joblib")

# transformer model id (3-class)
TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# ----------------- ensure NLTK -----------------
def ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")

ensure_nltk()
STOP_WORDS = set(stopwords.words("english"))
SIA = SentimentIntensityAnalyzer()

# ----------------- text cleaning -----------------
def clean_text(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w and w not in STOP_WORDS]
    return " ".join(tokens)

# ----------------- load optional TF-IDF + LR (for explainability) -----------------
tfidf = None
lr_clf = None
feature_names = None
coef_row = None

def load_tfidf_lr():
    global tfidf, lr_clf, feature_names, coef_row
    ok = False
    if os.path.exists(TFIDF_FILE) and os.path.exists(LR_FILE):
        try:
            tfidf = joblib.load(TFIDF_FILE)
            lr_clf = joblib.load(LR_FILE)
            # feature names safety
            if hasattr(tfidf, "get_feature_names_out"):
                feature_names = tfidf.get_feature_names_out()
            else:
                # fallback
                feature_names = np.array(tfidf.get_feature_names())
            # coefficient row: handle binary (shape (1,n)) or multiclass (choose positive-like)
            if hasattr(lr_clf, "coef_"):
                coefs = lr_clf.coef_
                # if binary, coefs.shape = (1, n) => interpret as positive-class weights
                if coefs.ndim == 2 and coefs.shape[0] == 1:
                    coef_row = coefs[0]
                else:
                    # try to pick the 'positive' row if class names exist
                    classes = list(lr_clf.classes_)
                    if "positive" in classes:
                        coef_row = coefs[classes.index("positive")]
                    else:
                        coef_row = coefs[0]
            ok = True
            print("[app] Loaded TF-IDF and LR models for explainability.")
        except Exception as e:
            print("[app] Failed to load TF-IDF/LR explainability models:", e)
            traceback.print_exc()
            tfidf = lr_clf = feature_names = coef_row = None
            ok = False
    else:
        print("[app] TF-IDF or LR model files not found in models/. Explainability disabled.")
    return ok

explain_ok = load_tfidf_lr()

# ----------------- Load transformer pipeline (primary predictor) -----------------
transformer_pipe = None
transformer_labels = None
transformer_available = False

def load_transformer():
    global transformer_pipe, transformer_labels, transformer_available
    try:
        # load tokenizer and model to guarantee access to id2label
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL)
        transformer_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)
        # map model.config.id2label (dictionary mapping ints to label strings)
        if hasattr(model.config, "id2label"):
            # build list by sorted keys
            id2label = model.config.id2label
            transformer_labels = [id2label[i].lower() for i in sorted(id2label.keys())]
        else:
            transformer_labels = None
        transformer_available = True
        print(f"[app] Transformer pipeline loaded ({TRANSFORMER_MODEL}).")
    except Exception as e:
        print("[app] Transformer pipeline not available / failed to load:", e)
        traceback.print_exc()
        transformer_pipe = None
        transformer_labels = None
        transformer_available = False

# try to load; this will download the model when you run locally or HF will download during build
load_transformer()

# ----------------- helper: bar figure -----------------
def make_bar_fig(percentages):
    labels = list(percentages.keys())
    values = [percentages[k] for k in labels]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage %")
    ax.set_title("Sentiment Distribution")
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    return fig

# ----------------- main predict function (uses transformer, plus TF-IDF explainability) -----------------
def predict_review(text: str):
    """
    Returns: (summary_text, top_words_text, matplotlib_fig)
    """
    try:
        if text is None or not isinstance(text, str) or text.strip() == "":
            fig = make_bar_fig({"positive":0.0, "negative":0.0, "neutral":100.0})
            return ("Please enter a non-empty review.", "N/A", fig)

        # 1) Primary prediction via transformer (if available)
        if transformer_available and transformer_pipe is not None:
            # transformer returns list of lists when return_all_scores=True, for single input -> list_of_scores
            scores_list = transformer_pipe(text)
            # pipeline with return_all_scores returns: [[{'label':..., 'score':...}, ...]]
            if isinstance(scores_list, list) and len(scores_list) > 0 and isinstance(scores_list[0], list):
                scores = scores_list[0]
            else:
                # unexpected shape
                scores = scores_list

            # Convert to dict label->score; attempt to map labels to readable names
            prob_dict = {}
            for s in scores:
                lbl = s.get("label", "")
                sc = float(s.get("score", 0.0))
                # if label looks like LABEL_0, we try to map using model config labels
                if lbl.startswith("LABEL_") and transformer_labels:
                    try:
                        # LABEL_0 -> index 0 -> transformer_labels[0]
                        idx = int(lbl.split("_")[1])
                        pretty = transformer_labels[idx]
                        prob_dict[pretty] = sc
                    except Exception:
                        prob_dict[lbl.lower()] = sc
                else:
                    prob_dict[lbl.lower()] = sc

            # ensure keys positive/negative/neutral exist (fill zeros if missing)
            pos = prob_dict.get("positive", 0.0) * 100
            neg = prob_dict.get("negative", 0.0) * 100
            neu = prob_dict.get("neutral", 0.0) * 100

            # heuristics: if transformer marks neutral as top or if top score low -> neutral
            top_label = max(prob_dict.items(), key=lambda x: x[1])[0]
            top_score = max(prob_dict.values())
            # use VADER as additional neutral clue
            vader = SIA.polarity_scores(text)
            compound = vader.get("compound", 0.0)
            uncertain = top_score < 0.60
            is_neutral = (top_label == "neutral") or (-0.05 <= compound <= 0.05) or uncertain

            if is_neutral:
                final_label = "neutral"
                # set neutral pct to 100 - max(pos,neg) so chart has something meaningful
                neutral_pct = max(0.0, 100.0 - max(pos, neg))
                percentages = {"positive": pos, "negative": neg, "neutral": neutral_pct}
            else:
                final_label = "positive" if pos >= neg else "negative"
                percentages = {"positive": pos, "negative": neg, "neutral": 0.0}

            prob_str = f"Predicted: {final_label.capitalize()} | P: {percentages['positive']:.1f}%, N: {percentages['negative']:.1f}%, Neutral(est): {percentages['neutral']:.1f}%"
        else:
            # transformer not available: try to fallback to local LR model (if available)
            if lr_clf is not None and tfidf is not None:
                cleaned = clean_text(text)
                vec = tfidf.transform([cleaned])
                prob = lr_clf.predict_proba(vec)[0]
                classes = list(lr_clf.classes_)
                prob_dict = dict(zip(classes, prob))
                pos = prob_dict.get("positive", 0.0) * 100
                neg = prob_dict.get("negative", 0.0) * 100
                top_label = classes[int(np.argmax(prob))]
                top_score = max(prob)
                vader = SIA.polarity_scores(text)
                compound = vader.get("compound", 0.0)
                uncertain = top_score < 0.60
                is_neutral = (-0.05 <= compound <= 0.05) or uncertain
                if is_neutral:
                    final_label = "neutral"
                    percentages = {"positive": pos, "negative": neg, "neutral": max(0.0, 100.0 - max(pos,neg))}
                else:
                    final_label = top_label
                    percentages = {"positive": pos, "negative": neg, "neutral": 0.0}
                prob_str = f"Predicted: {final_label.capitalize()} | P: {percentages['positive']:.1f}%, N: {percentages['negative']:.1f}%, Neutral(est): {percentages['neutral']:.1f}%"
            else:
                # nothing available
                fig = make_bar_fig({"positive":0.0, "negative":0.0, "neutral":100.0})
                return ("Model not available on the Space. Please upload transformer or tfidf+lr models into models/.", "N/A", fig)

        # 2) Explainability via TF-IDF + LR contributions (if available)
        top_words_str = "N/A"
        if explain_ok and tfidf is not None and lr_clf is not None and feature_names is not None and coef_row is not None:
            cleaned = clean_text(text)
            vec = tfidf.transform([cleaned])
            arr = vec.toarray()[0]
            present_idx = np.where(arr > 0)[0]
            contribs = []
            for idx in present_idx:
                if idx < len(feature_names) and idx < len(coef_row):
                    # contribution = tfidf_value * coef (signed)
                    contrib = float(arr[idx] * coef_row[idx])
                    contribs.append((feature_names[idx], contrib))
            # sort by absolute contribution descending
            contribs = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)[:8]
            # format: word(+0.23) or word(-0.12)
            if contribs:
                top_words_str = ", ".join([f"{w}({round(c,3)})" for w,c in contribs])
            else:
                top_words_str = "N/A"

        fig = make_bar_fig(percentages)
        return (prob_str, top_words_str, fig)

    except Exception as e:
        traceback.print_exc()
        fig = make_bar_fig({"positive":0.0, "negative":0.0, "neutral":100.0})
        return (f"Error during prediction: {str(e)}", "N/A", fig)

# ----------------- Gradio UI -----------------
title = "Movie Review Sentiment Analysis"
description = "Transformer-based sentiment prediction (positive/negative/neutral)."

iface = gr.Interface(
    fn=predict_review,
    inputs=gr.Textbox(lines=4, placeholder="Type a movie review here..."),
    outputs=[gr.Textbox(), gr.Textbox(), gr.Plot()],
    title=title,
    description=description,
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch(share=False)
