"""
Training and Evaluation Pipeline for Handwritten Digit Recognition.
Benchmarks KNN, SVM (RBF), Logistic Regression, Random Forest, and Decision Tree.
Prints per-class accuracy and a visual confusion matrix.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def print_confusion_matrix(cm, class_names):
    """Render a readable, compact confusion matrix in the terminal."""
    n = len(class_names)
    col_w = 5
    header = " " * 8 + "".join(f"{'pred ' + str(c):>{col_w}}" for c in class_names)
    print(header)
    print(" " * 8 + "-" * (col_w * n))
    for i, row in enumerate(cm):
        row_str = f"  true {class_names[i]} |"
        for j, val in enumerate(row):
            cell = str(val)
            if i == j:
                row_str += f"{'[' + cell + ']':>{col_w}}"  # diagonal (correct)
            elif val > 0:
                row_str += f"{cell:>{col_w}}"
            else:
                row_str += " " * col_w
        print(row_str)


def train_and_benchmark():
    print("=" * 72)
    print(" ✍️  Handwritten Digit Recognition - Model Benchmark & Training")
    print("=" * 72)

    # 1. Load dataset
    print("\n[Step 1/5] Loading Handwritten Digits Dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"Total samples: {len(X):,} | Features: {X.shape[1]} pixels | Classes: {len(np.unique(y))}")

    # 2. Train / Test Split (stratified)
    print("\n[Step 2/5] Stratified 80/20 Train/Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # 3. Model Candidates
    models = {
        "K-Nearest Neighbors (K=3)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3, metric="euclidean"))
        ]),
        "SVM (RBF Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=10.0, max_iter=2000, random_state=42, multi_class="auto"))
        ]),
        "Random Forest (200 trees)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=20, random_state=42))
        ])
    }

    results = []
    trained_pipelines = {}

    print("\n[Step 3/5] Training and evaluating classifiers...")
    print("-" * 72)

    for name, pipeline in models.items():
        print(f"Training {name:<26}...", end="", flush=True)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc})
        trained_pipelines[name] = pipeline
        print(f" Done! (Accuracy: {acc:.4f} → {acc * 100:.2f}%)")

    # Leaderboard
    df_results = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print("\n" + "=" * 72)
    print(" 🏆 Model Performance Leaderboard")
    print("=" * 72)
    for idx, row in df_results.reset_index(drop=True).iterrows():
        prefix = " ⭐ BEST -> " if idx == 0 else "           "
        bar = "█" * int(row["Accuracy"] * 30)
        print(f"{prefix}{row['Model']:<26} | {row['Accuracy']:.4f} ({row['Accuracy'] * 100:.2f}%)  {bar}")
    print("=" * 72)

    # 4. Best model — Confusion Matrix & Per-class Report
    best_name = df_results.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_name]
    y_pred_best = best_pipeline.predict(X_test)

    print(f"\n[Step 4/5] Confusion Matrix for Best Model: {best_name}")
    cm = confusion_matrix(y_test, y_pred_best)
    class_names = list(range(10))
    print_confusion_matrix(cm, class_names)

    # Per-class classification report
    print(f"\nPer-Digit Classification Report ({best_name}):")
    report = classification_report(y_test, y_pred_best, target_names=[str(d) for d in range(10)], digits=3)
    print(report)

    # Identify hardest-to-classify digit pairs
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    if cm_copy.max() > 0:
        row_err, col_err = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
        print(f"  💡 Most confused pair: Digit '{row_err}' mistaken as '{col_err}' "
              f"({cm_copy[row_err, col_err]} times out of {cm[row_err].sum()} '{row_err}' samples)")

    # 5. Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "best_digit_classifier.joblib")

    artifact = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "class_names": list(range(10)),
        "dataset_images": digits.images,  # store for sample visualization
        "dataset_labels": digits.target,
        "metrics": {
            "accuracy": df_results.iloc[0]["Accuracy"],
        }
    }
    joblib.dump(artifact, save_path)

    print(f"\n[Step 5/5] Successfully saved winning model ({best_name}) to:")
    print(f" -> {save_path}")
    print("\nRun `python3 predict.py` to classify individual digit samples interactively!")


if __name__ == "__main__":
    train_and_benchmark()
