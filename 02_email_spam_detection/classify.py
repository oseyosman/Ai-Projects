"""
Interactive Email & SMS Spam Classifier CLI.
Predicts whether a given text is HAM (legitimate) or SPAM (scam/phishing) with confidence scores.
"""

import os
import argparse
import joblib
import numpy as np


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_spam_detector.joblib")

PRESET_EMAILS = {
    "1": {
        "title": "Bank Account Suspension Phishing",
        "expected": "SPAM",
        "text": "URGENT! Your Bank of America account access has been locked due to suspicious activity. Click http://verify-bank-security.xyz immediately to confirm your SSN and restore access."
    },
    "2": {
        "title": "Lottery / Prize Giveaway Scam",
        "expected": "SPAM",
        "text": "CONGRATULATIONS!! You were selected as the lucky winner of a $1,000 Walmart gift card and cash prize. Call 08712400603 or reply WIN to claim your free reward now!"
    },
    "3": {
        "title": "Cryptocurrency 10x ROI Scam",
        "expected": "SPAM",
        "text": "Guaranteed 500% passive income! Double your Bitcoin in 24 hours with our automated crypto trading algorithm. Act now, limited slots remaining."
    },
    "4": {
        "title": "Work Sprint Retrospective Invite",
        "expected": "HAM",
        "text": "Hi team, please remember our sprint retrospective is scheduled for tomorrow at 2:00 PM. Please add your discussion points to the shared document before noon."
    },
    "5": {
        "title": "Amazon Order Delivery Notification",
        "expected": "HAM",
        "text": "Your package containing 'Python Data Science Handbook' has been delivered to your front door. If you have any questions, you can view your order details in your account."
    },
    "6": {
        "title": "Casual Personal Message",
        "expected": "HAM",
        "text": "Hey, are we still meeting up for coffee this Saturday around 11? Let me know if that time works for you!"
    }
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please run: python3 train.py first to benchmark and save the detector."
        )
    return joblib.load(MODEL_PATH)


def compute_confidence(pipeline, text: str):
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = pipeline.predict_proba([text])[0]
        spam_prob = float(probs[1])
    elif hasattr(clf, "decision_function"):
        score = pipeline.decision_function([text])[0]
        # Sigmoid conversion
        spam_prob = float(1.0 / (1.0 + np.exp(-score)))
    else:
        pred = pipeline.predict([text])[0]
        spam_prob = 0.95 if pred == 1 else 0.05
    return spam_prob


def explain_prediction(pipeline, text: str):
    """Finds words in the input message that influenced the spam classification."""
    try:
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]

        if not hasattr(clf, "coef_"):
            return []

        vocab = tfidf.vocabulary_
        coefs = clf.coef_[0]

        words = text.lower().split()
        triggers = []

        for word in set(words):
            clean_w = "".join(c for c in word if c.isalnum())
            if clean_w in vocab:
                idx = vocab[clean_w]
                weight = coefs[idx]
                if weight > 0.4:
                    triggers.append((clean_w, weight))

        triggers.sort(key=lambda x: x[1], reverse=True)
        return triggers[:5]
    except Exception:
        return []


def classify_message(artifact, text: str):
    pipeline = artifact["pipeline"]
    spam_prob = compute_confidence(pipeline, text)
    is_spam = spam_prob >= 0.50
    triggers = explain_prediction(pipeline, text)

    return {
        "is_spam": is_spam,
        "label": "SPAM" if is_spam else "HAM (LEGITIMATE)",
        "confidence": spam_prob if is_spam else (1.0 - spam_prob),
        "spam_probability": spam_prob,
        "triggers": triggers
    }


def print_result(res: dict, text: str):
    icon = "🚨" if res["is_spam"] else "✅"
    color_label = "SPAM / SCAM" if res["is_spam"] else "HAM / LEGITIMATE"

    print("\n" + "=" * 68)
    print(f" Message: \"{text[:120]}{'...' if len(text) > 120 else ''}\"")
    print("-" * 68)
    print(f" {icon} CLASSIFICATION: {color_label}")
    print(f" 📊 Confidence:     {res['confidence'] * 100:.1f}% (Spam Probability: {res['spam_probability'] * 100:.1f}%)")

    if res["triggers"]:
        print(f" 🔍 Detected Spam Triggers: {', '.join([f'{w} (+{score:.1f})' for w, score in res['triggers']])}")
    print("=" * 68)


def interactive_menu(artifact):
    print("\n" + "=" * 68)
    print(" 📧 Email & SMS Spam Detector CLI")
    print(f" Loaded Model: {artifact['model_name']} | Accuracy: {artifact['metrics']['accuracy']:.2%} | Precision: {artifact['metrics']['precision']:.2%}")
    print("=" * 68)

    print("\nSelect a sample to classify:")
    for key, p in PRESET_EMAILS.items():
        print(f" [{key}] {p['title']} (Expected: {p['expected']})")
    print(" [7] Enter custom email / message text")
    print(" [q] Quit")

    choice = input("\nYour choice [1-7, q]: ").strip()

    if choice in PRESET_EMAILS:
        sample = PRESET_EMAILS[choice]
        print(f"\nAnalyzing: {sample['title']}")
        print(f"Text: \"{sample['text']}\"")
        res = classify_message(artifact, sample["text"])
        print_result(res, sample["text"])

    elif choice == "7":
        custom_text = input("\nEnter or paste email text: ").strip()
        if custom_text:
            res = classify_message(artifact, custom_text)
            print_result(res, custom_text)
        else:
            print("No text entered.")
    elif choice.lower() == "q":
        print("Goodbye!")
    else:
        print("Invalid choice.")


def main():
    parser = argparse.ArgumentParser(description="Classify Email / SMS as HAM or SPAM")
    parser.add_argument("--text", type=str, help="Email or message text to classify")
    args = parser.parse_args()

    artifact = load_model()

    if args.text:
        res = classify_message(artifact, args.text)
        print_result(res, args.text)
    else:
        interactive_menu(artifact)


if __name__ == "__main__":
    main()
