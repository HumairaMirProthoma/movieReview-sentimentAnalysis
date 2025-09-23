1. Suggested structure

Title: Movie Review Sentiment Analysis (Positive / Negative / Neutral) – Personal Project

Context:
I built this project to practice NLP, machine learning, and explainable AI, with the goal of creating an interactive demo that shows not only predictions but also reasoning behind them.

Problem:
Most sentiment classifiers are binary and do not show uncertainty or explain why a prediction is made. This makes it hard to trust the model, especially for borderline reviews.

Solution:

Trained a Logistic Regression model with TF-IDF (unigrams + bigrams) features on the IMDB dataset.

Used VADER compound score to heuristically detect neutral or uncertain reviews.

Developed an interactive Gradio demo showing:

Predicted sentiment (Positive / Negative / Neutral)

Probability breakdown (bar chart)

Top contributing words for the prediction

My Contribution:

Data cleaning and preprocessing (removing HTML, punctuation, stopwords).

Model training, evaluation, and saving (joblib).

Feature explainability (top contributing words).

Designed and implemented the Gradio demo.

Deployed the project on Hugging Face Spaces, making it publicly accessible.

Limitations:

IMDB dataset contains only positive/negative labels; neutral detection is rule-based using VADER + low-confidence heuristic.

True neutral training would require collecting/labelling neutral examples and retraining as a 3-class classifier.

Logistic Regression may not handle very complex language patterns compared to deep learning models.

How to run locally:

Clone the repository.

Install dependencies: pip install -r requirements.txt.

Run: python app.py (Gradio demo will launch on localhost).

Hugging Face Space:
The project is deployed at: [your-space-li – accessible on any device via web browser.
