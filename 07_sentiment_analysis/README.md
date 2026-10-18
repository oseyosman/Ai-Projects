# Project 07: Sentiment Analysis (NLP Text Classification)

## 🎯 Goal
Classify the sentiment of customer product reviews into **Positive** or **Negative** using Natural Language Processing (NLP) and supervised machine learning on the **UCI Amazon Product Reviews** benchmark dataset.

The system tokenizes raw natural language, extracts **TF-IDF n-gram features**, and learns the subtle linguistic cues that signify consumer satisfaction or dissatisfaction.

---

## 🔤 NLP Pipeline Architecture

```
Raw Review Text ("The sound quality is crisp, but battery life is awful!")
    │
    ▼
Text Normalization (Lowercasing, punctuation handling, tokenization)
    │
    ▼
TF-IDF Vectorizer (Sublinear term frequency, unigrams & bigrams: 1-to-2 words)
    │
    ▼
Classifiers Benchmark (LinearSVC, Logistic Regression, Naive Bayes, Random Forest)
    │
    ▼
Predicted Sentiment (👍 POSITIVE / 👎 NEGATIVE) + Confidence Gauge + Signal Highlight
```

---

## 📁 Project Structure

```
07_sentiment_analysis/
├── README.md               # Project documentation & NLP guide
├── data/
│   └── amazon_reviews.csv  # 1,000 labelled Amazon customer reviews (500 positive, 500 negative)
├── eda.py                  # Exploratory Data Analysis & vocabulary frequency analysis
├── train.py                # Model training, cross-validation & feature coefficient export
├── predict.py              # Interactive CLI & real-time review sentiment analyzer
└── models/
    └── sentiment_classifier.joblib  # Trained TF-IDF + Classifier pipeline
```

---

## 🚀 Quick Start

### 1. Run Exploratory Data Analysis
Inspect review length distributions, class balance, and the most predictive positive vs. negative vocabulary:
```bash
python3 07_sentiment_analysis/eda.py
```

### 2. Train and Benchmark Classifiers
Benchmarks Naive Bayes, Logistic Regression, LinearSVC, and Random Forest:
```bash
python3 07_sentiment_analysis/train.py
```

### 3. Analyze Review Sentiment
Test any custom review string via command line:
```bash
python3 07_sentiment_analysis/predict.py --text "This headset has incredible noise cancellation and comfortable earcups!"
```

Or run the interactive console menu:
```bash
python3 07_sentiment_analysis/predict.py
```

---

## 📊 Key Machine Learning & NLP Concepts Explored

- **Bag-of-Words & TF-IDF**: Transforming unstructured variable-length text into high-dimensional numerical vectors.
- **N-Grams (Bigrams)**: Crucial for capturing negation and context (e.g., distinguishing between *"good"* vs *"not good"* or *"waste of money"*).
- **Linear Classifiers in High Dimensions**: Understanding why Linear Support Vector Machines and Logistic Regression perform well on sparse text spaces.
- **Model Explainability**: Inspecting model weights ($\beta$ coefficients) to discover which words drive positive or negative classification.
