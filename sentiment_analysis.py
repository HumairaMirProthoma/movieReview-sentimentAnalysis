# sentiment_analysis.py
# IMDB Sentiment Analysis: Positive, Negative, Neutral
# Features:
# - TF-IDF vectorization (unigrams + bigrams)
# - Logistic Regression
# - Top contributing words
# - WordCloud visualizations
# - Gradio demo with sentiment probabilities and bar chart
# - Neutral detection using VADER

import os, sys, argparse
print("Starting script... Python:", sys.version.split()[0])

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    import nltk
    import joblib
    from wordcloud import WordCloud
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
except Exception as e:
    print("Missing packages:", e)
    print("Install with: python -m pip install pandas numpy matplotlib seaborn nltk scikit-learn joblib wordcloud")
    raise

# Gradio
GRADIO_AVAILABLE = True
try:
    import gradio as gr
except Exception:
    GRADIO_AVAILABLE = False

# --- Paths ---
PROJECT_DIR = os.getcwd()
DATA_CANDIDATES = [
    os.path.join(PROJECT_DIR, "Data", "IMDB_Dataset.csv"),
    os.path.join(PROJECT_DIR, "IMDB_Dataset.csv"),
]
dataset_path = None
for p in DATA_CANDIDATES:
    if os.path.exists(p):
        dataset_path = p
        break
if dataset_path is None:
    print("Dataset not found. Please place IMDB_Dataset.csv in Data folder.")
    sys.exit(1)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Dataset ---
try:
    df = pd.read_csv(dataset_path)
except UnicodeDecodeError:
    df = pd.read_csv(dataset_path, encoding='latin-1')
if 'review' not in df.columns or 'sentiment' not in df.columns:
    print("Expected columns: 'review', 'sentiment'")
    sys.exit(1)
print("Dataset shape:", df.shape)

# --- NLTK stopwords and VADER ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# --- Clean text ---
def clean_text(text):
    if not isinstance(text,str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [w for w in text.split() if w and w not in stop_words]
    return ' '.join(tokens)
df['cleaned'] = df['review'].apply(clean_text)

# --- WordCloud ---
def create_wordcloud(text, title, path):
    wc = WordCloud(width=800,height=400,background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off'); plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

create_wordcloud(" ".join(df[df['sentiment']=='positive']['cleaned']),
                 "WordCloud Positive Reviews", os.path.join(OUTPUT_DIR,"wordcloud_positive.png"))
create_wordcloud(" ".join(df[df['sentiment']=='negative']['cleaned']),
                 "WordCloud Negative Reviews", os.path.join(OUTPUT_DIR,"wordcloud_negative.png"))

# --- Train/test split ---
X = df['cleaned']; y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# --- Vectorization ---
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Logistic Regression ---
lr_clf = LogisticRegression(solver='liblinear', max_iter=1000)
lr_clf.fit(X_train_vec, y_train)

# --- Evaluate ---
y_pred = lr_clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# --- Feature explainability ---
feature_names = vectorizer.get_feature_names_out()
coefs = lr_clf.coef_[0]

# --- Save model/vectorizer ---
joblib.dump(lr_clf, os.path.join(MODEL_DIR,"lr_clf.joblib"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR,"tfidf_vectorizer.joblib"))

# --- Prediction function with neutral detection ---
def predict_with_neutral(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    # Logistic Regression probabilities
    prob = lr_clf.predict_proba(vec)[0]
    prob_dict = dict(zip(lr_clf.classes_, prob))
    
    # VADER compound score for neutral detection
    vader_scores = sia.polarity_scores(text)
    compound = vader_scores['compound']
    
    # Determine label
    if -0.05 <= compound <= 0.05:
        label = 'neutral'
    else:
        label = lr_clf.classes_[np.argmax(prob)]
    
    # Top word contributions
    arr = vec.toarray()[0]
    present_idx = np.where(arr>0)[0]
    contribs = [(feature_names[i], float(coefs[i])) for i in present_idx]
    contribs = sorted(contribs, key=lambda x:x[1], reverse=True)[:6]
    
    # Percentages for bar chart
    if label == 'neutral':
        percentages = {
            'positive': prob_dict.get('positive',0)*100,
            'negative': prob_dict.get('negative',0)*100,
            'neutral': 100 - max(prob_dict.get('positive',0)*100, prob_dict.get('negative',0)*100)
        }
    else:
        percentages = {
            'positive': prob_dict.get('positive',0)*100,
            'negative': prob_dict.get('negative',0)*100,
            'neutral': 0
        }
    
    return {
        'label': label,
        'percentages': percentages,
        'top_words': contribs
    }

# --- Gradio demo ---
def launch_gradio():
    if not GRADIO_AVAILABLE:
        print("Install gradio: python -m pip install gradio")
        return

    def gr_predict(text):
        info = predict_with_neutral(text)
        label = info['label'].capitalize()
        words = ', '.join([f"{w}({round(c,3)})" for w,c in info['top_words']])
        # Plot bar chart
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(info['percentages'].keys(), info['percentages'].values(), color=['green','red','gray'])
        ax.set_ylim(0,100)
        ax.set_ylabel("Percentage %")
        ax.set_title("Sentiment Distribution")
        fig.tight_layout()
        return f"Predicted Sentiment: {label}", f"Top contributing words: {words}", fig

    iface = gr.Interface(
        fn=gr_predict,
        inputs="text",
        outputs=["text","text","plot"],
        title="IMDB Sentiment Classifier (Positive/Negative/Neutral)",
        description="Type a review and see predicted sentiment, top contributing words, and visual sentiment percentages.",
        allow_flagging='never'
    )
    print("Launching Gradio demo...")
    iface.launch(share=True)

# --- CLI arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--demo', action='store_true', help='Launch Gradio demo')
args = parser.parse_args()
if args.demo:
    launch_gradio()
else:
    print("Script ready. Run: python sentiment_analysis.py --demo to launch the interactive demo.")
