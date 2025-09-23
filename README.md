---
title: Movie Review Sentiment Analysis
emoji: 🎬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "3.41.0"
app_file: app.py
pinned: false
---

# Movie Review Sentiment Analysis

**An interactive Gradio demo that predicts the sentiment of movie reviews (Positive, Negative, Neutral) using Logistic Regression (TF-IDF) and VADER heuristic for neutral detection.**

## Project Context

The goal of this project is to explore Natural Language Processing (NLP) and machine learning techniques for sentiment analysis, while creating an **explainable and interactive demo**. Sentiment analysis is widely used in analyzing customer reviews, social media data, and feedback systems, but most off-the-shelf models are **binary (positive/negative)** and do not provide reasoning or uncertainty estimates.

This project demonstrates how to combine **machine learning models** with **heuristics** to provide a more nuanced understanding of sentiment, including neutral or uncertain opinions.

## Problem Statement

- Most sentiment classifiers output only binary labels (positive or negative).
- Users cannot see the reasoning behind predictions or understand confidence levels.
- Neutral reviews are often ignored or misclassified, limiting real-world usability.

## Solution Overview

- **TF-IDF Vectorization + Logistic Regression:** Trains a model on the IMDB movie reviews dataset for positive and negative classification.
- **Neutral/Uncertain Detection:** Uses **VADER sentiment analyzer** compound score and low-confidence heuristic to detect potentially neutral reviews.
- **Explainability:** Displays **top contributing words** from the input review that influenced the model's prediction.
- **Interactive Demo:** A **Gradio web interface** visualizes sentiment probabilities in a bar chart and shows top contributing words.

## Features

1. Clean and preprocess movie reviews for model input.
2. Train logistic regression on TF-IDF vectors with unigrams and bigrams.
3. Save and load trained models for quick deployment.
4. Predict sentiment with probability scores for positive, negative, and neutral.
5. Show top contributing words to aid explainability.
6. Visualize sentiment distribution as a bar chart in the Gradio interface.
7. Easy-to-run local demo and Hugging Face deploy
