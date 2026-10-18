"""
Training and Evaluation Pipeline for Amazon Review Sentiment Analysis.
Benchmarks Multinomial Naive Bayes, Logistic Regression, Linear SVC, and Random Forest.
Extracts top positive and negative sentiment token coefficients and exports champion model.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def print_banner(text: str, char: str = "=", width: int = 75):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def render_confusion_matrix(cm):
    print("\n  Confusion Matrix:")
    print("                 Pred NEGATIVE 👎   Pred POSITIVE 👍")
    print("  Actual NEGATIVE 👎      " + f"{cm[0][0]:>5}              {cm[0][1]:>5}")
    print("  Actual POSITIVE 👍      " + f"{cm[1][0]:>5}              {cm[1][1]:>5}")


def main():
    print_banner("💬 TRAINING AMAZON REVIEWS SENTIMENT CLASSIFIER 💬")

    csv_path = os.path.join(os.path.dirname(__file__), "data", "amazon_reviews.csv")
    df = pd.read_csv(csv_path)

    X = df["review_text"]
    y = df["sentiment"]

    # 1. Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Dataset Split: {len(X_train)} training reviews, {len(X_test)} test reviews (Stratified)")

    # 2. Pipeline Candidates
    models = {
        "Logistic Regression (L2)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=2.5, max_iter=1000, random_state=42))
        ]),
        "Linear Support Vector (SVC)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LinearSVC(C=1.0, random_state=42))
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.5))
        ]),
        "Random Forest (100 Trees)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }

    print_banner("1. 5-FOLD STRATIFIED CROSS-VALIDATION & TEST BENCHMARK")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    trained_pipelines = {}

    for name, pipeline in models.items():
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        trained_pipelines[name] = pipeline
        results.append({
            "Model": name,
            "CV Mean Acc": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Test Acc": test_acc
        })

    print(f"{'Model Name':<32} | {'CV Accuracy (Train)':<20} | {'Test Accuracy':<14}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: (x["Test Acc"], x["CV Mean Acc"]), reverse=True):
        cv_str = f"{r['CV Mean Acc'] * 100:.2f}% (±{r['CV Std'] * 100:.2f}%)"
        test_str = f"{r['Test Acc'] * 100:.2f}%"
        print(f"{r['Model']:<32} | {cv_str:<20} | {test_str:<14}")

    # 3. Champion Evaluation
    champion_name = "Logistic Regression (L2)"
    champion_pipeline = trained_pipelines[champion_name]

    print_banner(f"2. CHAMPION EVALUATION: {champion_name.upper()}")
    y_pred_best = champion_pipeline.predict(X_test)
    best_acc = accuracy_score(y_test, y_pred_best)

    print(f"Final Test Accuracy: {best_acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=["Negative (0)", "Positive (1)"]))

    cm = confusion_matrix(y_test, y_pred_best)
    render_confusion_matrix(cm)

    # 4. Feature Linguistic Analysis (Top Sentiment Drivers)
    print_banner("3. TOP 12 MOST INFLUENTIAL SENTIMENT TOKENS")
    tfidf = champion_pipeline.named_steps["tfidf"]
    clf = champion_pipeline.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]

    top_pos_idx = np.argsort(coefs)[-12:][::-1]
    top_neg_idx = np.argsort(coefs)[:12]

    print(f"{'Top Positive Indicators (👍)':<36} | {'Top Negative Indicators (👎)':<36}")
    print("-" * 75)
    for i in range(12):
        pos_token = feature_names[top_pos_idx[i]]
        pos_weight = coefs[top_pos_idx[i]]
        neg_token = feature_names[top_neg_idx[i]]
        neg_weight = coefs[top_neg_idx[i]]

        p_str = f"  +{pos_weight:>4.2f} : '{pos_token}'"
        n_str = f"  {neg_weight:>5.2f} : '{neg_token}'"
        print(f"{p_str:<36} | {n_str:<36}")

    # 5. Persist Model Artifact
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "sentiment_classifier.joblib")

    artifact = {
        "pipeline": champion_pipeline,
        "model_name": champion_name,
        "test_accuracy": best_acc,
        "classes": ["Negative", "Positive"]
    }
    joblib.dump(artifact, model_path)

    print_banner("4. MODEL PERSISTENCE & READINESS")
    print(f"  Artifact successfully saved to: {model_path}")
    print(f"  Accuracy: {best_acc * 100:.2f}%")
    print("  Ready for interactive and CLI sentiment analysis!\n")


if __name__ == "__main__":
    main()
