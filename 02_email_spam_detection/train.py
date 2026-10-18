"""
Training and Evaluation Pipeline for Email & SMS Spam Detection.
Benchmarks MultinomialNB, ComplementNB, Logistic Regression, LinearSVC, and Random Forest.
Evaluates Precision, Recall, F1, and Confusion Matrices.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "spam_dataset.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_data():
    if not os.path.exists(DATA_FILE):
        import download_data
        download_data.download_or_get_dataset()
    df = pd.read_csv(DATA_FILE)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    df.dropna(subset=["text", "label_num"], inplace=True)
    return df


def train_and_benchmark():
    print("=" * 72)
    print(" 📧 Email & SMS Spam Detection - Model Benchmarking Pipeline")
    print("=" * 72)

    df = load_data()
    print(f"\n[Step 1/4] Loaded {len(df):,} messages ({df['label_num'].sum():,} spam, {len(df) - df['label_num'].sum():,} ham)")

    X = df["text"].astype(str)
    y = df["label_num"]

    # Stratified Split
    print("\n[Step 2/4] Stratified 80/20 Train/Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # Shared TF-IDF Vectorizer configuration
    def make_vectorizer():
        return TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=6000,
            sublinear_tf=True
        )

    # Models to compare
    models = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", MultinomialNB(alpha=0.1))
        ]),
        "Complement Naive Bayes": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", ComplementNB(alpha=0.2))
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LogisticRegression(C=5.0, max_iter=1000, random_state=42))
        ]),
        "Linear Support Vector (SVC)": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LinearSVC(C=1.0, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
    }

    results = []
    trained_pipelines = {}

    print("\n[Step 3/4] Training and evaluating NLP classification models...")
    print("-" * 72)

    for name, pipeline in models.items():
        print(f"Training {name:<28}...", end="", flush=True)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Spam)": prec,
            "Recall (Spam)": rec,
            "F1 Score": f1,
            "False Positives (Ham->Spam)": fp,
            "False Negatives (Missed)": fn
        })
        trained_pipelines[name] = pipeline
        print(f" Done! (F1: {f1:.4f} | Precision: {prec:.4f} | FP: {fp})")

    # Present Leaderboard (Ranked by F1 Score and Precision)
    df_results = pd.DataFrame(results).sort_values(by=["F1 Score", "Precision (Spam)"], ascending=False)
    print("\n" + "=" * 72)
    print(" 🏆 Model Performance Leaderboard")
    print("=" * 72)
    for idx, row in df_results.reset_index(drop=True).iterrows():
        prefix = " ⭐ BEST -> " if idx == 0 else "           "
        print(f"{prefix}{row['Model']:<27} | Acc: {row['Accuracy']:.4f} | Prec: {row['Precision (Spam)']:.4f} | Rec: {row['Recall (Spam)']:.4f} | F1: {row['F1 Score']:.4f} | FP: {row['False Positives (Ham->Spam)']}")
    print("=" * 72)

    # Pick the best model: high F1 with lowest False Positives
    best_row = df_results.iloc[0]
    best_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_name]

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_DIR, "best_spam_detector.joblib")

    artifact = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "metrics": {
            "accuracy": best_row["Accuracy"],
            "precision": best_row["Precision (Spam)"],
            "recall": best_row["Recall (Spam)"],
            "f1": best_row["F1 Score"],
            "false_positives": int(best_row["False Positives (Ham->Spam)"]),
            "false_negatives": int(best_row["False Negatives (Missed)"])
        }
    }
    joblib.dump(artifact, model_save_path)
    print(f"\n[Step 4/4] Successfully saved winning model ({best_name}) to:")
    print(f" -> {model_save_path}")

    # Inspect Top Spam Signal Words
    try:
        tfidf = best_pipeline.named_steps["tfidf"]
        feature_names = np.array(tfidf.get_feature_names_out())
        clf = best_pipeline.named_steps["clf"]

        if hasattr(clf, "coef_"):
            coefs = clf.coef_[0]
            top_spam_idx = np.argsort(coefs)[::-1][:10]
            top_ham_idx = np.argsort(coefs)[:10]

            print("\nTop 10 Words Indicating SPAM (Highest Coefficients):")
            for rank, i in enumerate(top_spam_idx, 1):
                print(f"  {rank:>2}. 🚨 {feature_names[i]:<16} (+{coefs[i]:.3f})")

            print("\nTop 10 Words Indicating HAM / Legitimate (Lowest Coefficients):")
            for rank, i in enumerate(top_ham_idx, 1):
                print(f"  {rank:>2}. ✅ {feature_names[i]:<16} ({coefs[i]:.3f})")
    except Exception as e:
        pass

    print("\nTraining completed! Run `python3 classify.py` to test the detector on any message.")


if __name__ == "__main__":
    train_and_benchmark()
