"""
Interactive Sentiment Analysis Predictor for Customer Reviews.
Accepts custom product review text via CLI arguments or interactive menu,
computes positive/negative sentiment probabilities, and extracts key polar words.
"""

import os
import sys
import argparse
import joblib
import numpy as np


SAMPLE_REVIEWS = [
    {
        "label": "Glowing 5-Star Review",
        "text": "This headset is phenomenal! Crystal clear sound, strong battery life, and super comfortable.",
        "expected": "Positive 👍"
    },
    {
        "label": "Disastrous 1-Star Review",
        "text": "Absolute junk. Stopped working after three days, completely defective and a total waste of money.",
        "expected": "Negative 👎"
    },
    {
        "label": "Subtle Negation Review",
        "text": "Looks sleek on the outside, but it is not good and has terrible mic distortion.",
        "expected": "Negative 👎"
    },
    {
        "label": "Moderate Satisfied Review",
        "text": "Great value for the price. Works fine and arrived fast.",
        "expected": "Positive 👍"
    }
]


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "sentiment_classifier.joblib")
    if not os.path.exists(model_path):
        print(f"Error: Model artifact not found at '{model_path}'.")
        print("Please train the model first by running:\n  python3 07_sentiment_analysis/train.py")
        sys.exit(1)
    return joblib.load(model_path)


def format_probability_bar(prob: float, width: int = 24) -> str:
    filled = int(round(prob * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {prob * 100:>5.1f}%"


def explain_prediction(text: str, pipeline):
    """Identifies influential n-grams present in the text."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]

    # Transform text to see active features
    vec = tfidf.transform([text])
    active_indices = vec.indices

    signals = []
    for idx in active_indices:
        token = feature_names[idx]
        weight = coefs[idx] * vec[0, idx]
        signals.append((token, weight))

    signals.sort(key=lambda x: abs(x[1]), reverse=True)
    return signals[:6]


def analyze_sentiment(text: str, model_artifact: dict):
    pipeline = model_artifact["pipeline"]
    probs = pipeline.predict_proba([text])[0]
    pred = pipeline.predict([text])[0]

    neg_prob = probs[0]
    pos_prob = probs[1]

    if pred == 1:
        sentiment_label = "POSITIVE 👍"
        confidence = pos_prob
    else:
        sentiment_label = "NEGATIVE 👎"
        confidence = neg_prob

    print("\n" + "=" * 68)
    print(f" 💬 SENTIMENT VERDICT: {sentiment_label}")
    print("=" * 68)
    print(f"  • Input Text   : \"{text}\"")
    print(f"  • Confidence   : {confidence * 100:.1f}%")
    print("\n  Class Probabilities:")
    print(f"    Positive 👍 : {format_probability_bar(pos_prob)}")
    print(f"    Negative 👎 : {format_probability_bar(neg_prob)}")

    # Explanatory signals
    signals = explain_prediction(text, pipeline)
    if signals:
        print("\n  Top Detected Linguistic Signals:")
        for token, w in signals:
            polarity = "Positive Signal (+)" if w > 0 else "Negative Signal (-)"
            indicator = "🟢" if w > 0 else "🔴"
            print(f"    {indicator} '{token:<16}' : {polarity} (impact: {w:+.2f})")

    print("=" * 68 + "\n")
    return pred


def run_interactive(model_artifact):
    while True:
        print("\n" + "=" * 60)
        print(" 💬 CUSTOMER REVIEW SENTIMENT ANALYZER - MENU 💬")
        print("=" * 60)
        print("  1) Analyze Custom Customer Review")
        print("  2) Test Pre-loaded Benchmark Reviews")
        print("  3) View Model Info & Architecture")
        print("  4) Exit")
        print("-" * 60)

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            review_input = input("\nEnter product review text to analyze:\n> ").strip()
            if review_input:
                analyze_sentiment(review_input, model_artifact)
            else:
                print("Empty review text entered.")

        elif choice == "2":
            print("\nSelect sample review:")
            for idx, item in enumerate(SAMPLE_REVIEWS, 1):
                print(f"  {idx}) {item['label']} (Expected: {item['expected']})")
                print(f"     \"{item['text']}\"")

            sel = input(f"Select (1-{len(SAMPLE_REVIEWS)}): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(SAMPLE_REVIEWS):
                chosen = SAMPLE_REVIEWS[int(sel) - 1]
                analyze_sentiment(chosen["text"], model_artifact)
            else:
                print("Invalid selection.")

        elif choice == "3":
            print("\nModel Artifact Info:")
            print(f"  • Model Architecture : {model_artifact['model_name']}")
            print(f"  • Test Accuracy      : {model_artifact['test_accuracy'] * 100:.2f}%")
            print("  • Feature Extractor  : TF-IDF with Unigrams and Bigrams")

        elif choice == "4":
            print("Exiting Sentiment Analyzer. Happy NLP analyzing!\n")
            break
        else:
            print("Invalid option. Please choose 1 to 4.")


def main():
    parser = argparse.ArgumentParser(description="Classify product review sentiment as positive or negative.")
    parser.add_argument("--text", type=str, help="Product review text to classify")

    args = parser.parse_args()
    model_artifact = load_model()

    if args.text:
        analyze_sentiment(args.text, model_artifact)
    else:
        run_interactive(model_artifact)


if __name__ == "__main__":
    main()
