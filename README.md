# Movie Review Sentiment Analysis

**An interactive Gradio demo that predicts the sentiment of movie reviews (Positive, Negative, Neutral) using Logistic Regression (TF-IDF) and VADER heuristic for neutral detection.**

---

## Project Context

The goal of this project is to explore **Natural Language Processing (NLP)** and **machine learning techniques** for sentiment analysis, while creating an **explainable and interactive demo**.

Most sentiment analysis tools are **binary (positive/negative)** and do not provide uncertainty estimation or reasoning. This project shows how combining **machine learning** with **heuristics** can produce a more nuanced and interpretable system.

---

## Problem Statement

- Sentiment classifiers usually output only **positive or negative**.
- Users rarely see **why** a model predicts a certain class.
- Neutral or vague reviews (e.g., _“The movie was okay”_, _“Not bad”_) are often ignored or misclassified.

---

## Solution Overview

- **TF-IDF Vectorization + Logistic Regression:** Trained on the IMDB dataset (positive/negative labels).
- **Neutral/Uncertain Detection:** Integrated **VADER sentiment analyzer** and a **low-confidence heuristic**.
- **Explainability:** Shows **top contributing words** that influenced the decision.
- **Interactive Demo:** A **Gradio web app** visualizes probabilities in a bar chart and highlights explanations.

---

## Features

1. Preprocess reviews with tokenization, stopword removal, normalization.
2. Train Logistic Regression with **TF-IDF (unigrams + bigrams)**.
3. Save and load models with `joblib` for deployment.
4. Predict sentiment with **probability distribution** (positive, negative, neutral).
5. Show **top contributing words** for interpretability.
6. Visualize predictions in **Gradio interface**.
7. Run locally or deploy to **Hugging Face Spaces** with ease.

---

## My Contributions

- Cleaned and preprocessed the IMDB dataset.
- Trained Logistic Regression classifier with TF-IDF features.
- Integrated **neutral handling** using VADER and confidence thresholds.
- Implemented **explainability** with word-level feature contributions.
- Built the **Gradio interface** for interactive predictions.
- Deployed the model successfully to **Hugging Face Spaces**.

---

## Limitations

- **Neutral Handling:**

  - The IMDB dataset only has **positive/negative labels**.
  - Neutral detection is **rule-based** (VADER + confidence heuristic).
  - Sometimes vague reviews (e.g., _“The movie is not bad”_) may still be misclassified.

- **Binary Training:**

  - The Logistic Regression model is fundamentally binary.
  - A proper **3-class model** would require neutral-labeled training data.

- **Domain Specificity:**

  - Model was trained on **movie reviews**.
  - May not generalize well to product reviews, tweets, or other domains.

---

## Future Improvements

- Collect or use datasets with **explicit neutral labels** for **3-class training**.
- Explore **transformer models** (BERT, DistilBERT) for better accuracy.
- Use **SHAP/LIME** for deeper explainability.
- Expand to other domains beyond movie reviews.

---

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/HumairaMirProthoma/movieReview-sentimentAnalysis.git
cd movieReview-sentimentAnalysis
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open the demo in your browser:

```
http://127.0.0.1:7860
```

---

## Demo

Try it live on **Hugging Face Spaces**:
👉 [Movie Review Sentiment Analysis](https://huggingface.co/spaces/HumairaProthoma/movieReview-sentimentAnalysis)

---
