"""
Training and Evaluation Pipeline for Iris Flower Classification.
Benchmarks K-Nearest Neighbors, SVM, Logistic Regression, Decision Tree, and Random Forest.
Exports the highest-performing pipeline for inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def print_banner(text: str, char: str = "=", width: int = 70):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def render_confusion_matrix(cm, labels):
    print("\n  Confusion Matrix (Rows: Actual, Columns: Predicted):")
    header = "             " + "  ".join([f"{l[:7]:>7}" for l in labels])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx, row in enumerate(cm):
        row_str = f"  {labels[idx]:<10} | " + "  ".join([f"{val:>7d}" for val in row])
        print(row_str)


def main():
    print_banner("🌸 TRAINING IRIS FLOWER CLASSIFIERS 🌸")

    # 1. Load Data
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = list(iris.target_names)

    # 2. Stratified Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples (Stratified)")

    # 3. Model Candidates Pipeline definitions
    models = {
        "K-Nearest Neighbors (k=3)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3))
        ]),
        "K-Nearest Neighbors (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Support Vector Machine (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
        ]),
        "Logistic Regression (Multinomial)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=4, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
        ]),
    }

    print_banner("1. 5-FOLD STRATIFIED CROSS-VALIDATION & TEST BENCHMARK")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    trained_pipelines = {}

    for name, pipeline in models.items():
        # CV Score on train set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        
        # Fit on entire training set & evaluate on held-out test set
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

    # Summary table
    print(f"{'Model Name':<34} | {'CV Accuracy':<16} | {'Test Accuracy':<14}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: (x["Test Acc"], x["CV Mean Acc"]), reverse=True):
        cv_str = f"{r['CV Mean Acc'] * 100:.2f}% (±{r['CV Std'] * 100:.2f}%)"
        test_str = f"{r['Test Acc'] * 100:.2f}%"
        print(f"{r['Model']:<34} | {cv_str:<16} | {test_str:<14}")

    # 4. Select Champion Model
    best_candidate = max(results, key=lambda x: (x["Test Acc"], x["CV Mean Acc"]))
    champion_name = best_candidate["Model"]
    champion_pipeline = trained_pipelines[champion_name]

    print_banner(f"2. CHAMPION MODEL EVALUATION: {champion_name.upper()}")
    y_pred_best = champion_pipeline.predict(X_test)

    print(f"Final Test Accuracy: {best_candidate['Test Acc'] * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred_best)
    render_confusion_matrix(cm, target_names)

    # 5. Feature Importances (from Random Forest)
    rf_pipeline = trained_pipelines["Random Forest"]
    rf_model = rf_pipeline.named_steps["clf"]
    importances = rf_model.feature_importances_

    print_banner("3. FEATURE IMPORTANCE RANKING (Random Forest)")
    feat_ranks = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for rank, (feat, imp) in enumerate(feat_ranks, 1):
        bar = "█" * int(imp * 40)
        print(f"  {rank}. {feat:<22} : {imp * 100:>5.2f}% |{bar}")

    # 6. Save Model Artifact
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "iris_classifier.joblib")

    artifact = {
        "pipeline": champion_pipeline,
        "model_name": champion_name,
        "feature_names": feature_names,
        "target_names": target_names,
        "accuracy": best_candidate["Test Acc"]
    }
    joblib.dump(artifact, model_path)
    print_banner("4. MODEL PERSISTENCE")
    print(f"  Saved trained pipeline to: {model_path}")
    print(f"  Accuracy: {best_candidate['Test Acc'] * 100:.2f}%")
    print("  Ready for interactive and CLI prediction!\n")


if __name__ == "__main__":
    main()
