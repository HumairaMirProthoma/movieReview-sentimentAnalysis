# IMDB Sentiment Analysis (Personal Project)

Context:
I built this project to practice NLP and machine learning and to create an explainable demo.

Problem:
Most sentiment classifiers are binary and do not show uncertainty or reasoning for predictions.

Solution:

- TF-IDF + Logistic Regression trained on IMDB dataset.
- VADER compound score used to heuristically detect neutral / uncertain reviews.
- Interactive Gradio demo showing predicted class, probability breakdown, and top contributing words.

My Contribution:

- Data cleaning, model training and evaluation.
- Explainability (top contributing words).
- Gradio demo and deployment to Hugging Face Spaces.

Limitations:

- IMDB dataset contains only positive/negative labels. Neutral detection is rule-based (VADER + low-confidence heuristic).
- For true neutral training, one must collect or label neutral examples and retrain as a 3-class classifier.

How to run locally:

```bash
python app.py
# then open http://127.0.0.1:7860
```
