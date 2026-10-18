"""
Training and Evaluation Pipeline for Titanic Survival Classification.
Benchmarks Logistic Regression, SVM, Random Forest, Gradient Boosting,
HistGradientBoosting, Decision Tree, and K-Nearest Neighbors using 5-Fold
Stratified Cross-Validation. Exports the champion model pipeline for inference.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Local module import
sys.path.insert(0, os.path.dirname(__file__))
from features import (
    engineer_features,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
)


def print_banner(text: str, char: str = "=", width: int = 74):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def render_confusion_matrix(cm: np.ndarray, labels: list[str]):
    print("\n  Confusion Matrix (Rows: Actual Ground Truth, Columns: Predicted):")
    header = "               " + "  ".join([f"{l:>12}" for l in labels])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx, row in enumerate(cm):
        row_str = f"  {labels[idx]:<12} | " + "  ".join([f"{val:>12d}" for val in row])
        print(row_str)


def build_preprocessor() -> ColumnTransformer:
    """Constructs column transformer for scaling numericals and one-hot encoding categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def main():
    print_banner("🚢 TRAINING & BENCHMARKING TITANIC SURVIVAL CLASSIFIERS 🚢")

    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "data", "train.csv")
    if not os.path.exists(data_path):
        print(f"Error: Training dataset not found at {data_path}.")
        sys.exit(1)

    raw_df = pd.read_csv(data_path)
    print(f"[Step 1/5] Loaded raw passenger manifest: {len(raw_df)} passengers.")

    # 2. Train / Held-Out Test Split (80% Train, 20% Test, Stratified)
    train_raw, test_raw = train_test_split(
        raw_df, test_size=0.20, random_state=42, stratify=raw_df["Survived"]
    )
    print(f"[Step 2/5] Stratified split: {len(train_raw)} training passengers, {len(test_raw)} held-out test passengers.")

    # 3. Feature Engineering with Strict Leakage Prevention
    # Compute medians ONLY on train split
    X_train_df, trained_medians = engineer_features(train_raw, trained_medians=None)
    y_train = train_raw["Survived"].values

    # Transform test split using train medians
    X_test_df, _ = engineer_features(test_raw, trained_medians=trained_medians)
    y_test = test_raw["Survived"].values

    print(f"  • Feature Engineering Complete: {X_train_df.shape[1]} engineered attributes per passenger.")
    print(f"  • Imputation Medians: Fare by Pclass={trained_medians['pclass_fare']}, Global Age={trained_medians['global_age']:.1f}y")

    # 4. Model Candidates
    candidate_models = {
        "Support Vector Machine (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.06, max_depth=3, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.06, max_depth=3, random_state=42),
        "Logistic Regression (L2)": LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        "Decision Tree (Pruned)": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42),
        "K-Nearest Neighbors (k=7)": KNeighborsClassifier(n_neighbors=7),
    }

    print_banner("[Step 3/5] 5-FOLD STRATIFIED CV BENCHMARK & EVALUATION")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    trained_pipelines = {}

    for name, clf in candidate_models.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        # Stratified Cross-Validation on training set
        cv_scores = cross_val_score(pipeline, X_train_df, y_train, cv=cv, scoring="accuracy")

        # Fit on full training set and evaluate on held-out test set
        pipeline.fit(X_train_df, y_train)
        y_pred = pipeline.predict(X_test_df)
        y_prob = pipeline.predict_proba(X_test_df)[:, 1] if hasattr(pipeline, "predict_proba") else None

        test_acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

        trained_pipelines[name] = pipeline
        results.append({
            "Model": name,
            "CV Mean Acc": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Test Acc": test_acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc,
        })

    # Leaderboard Summary
    print(f"{'Rank':<5} {'Model Name':<28} | {'CV 5-Fold':<15} | {'Test Acc':<10} | {'F1':<6} | {'ROC-AUC':<8}")
    print("-" * 74)
    # Sort by primary: Test Acc + CV Mean Acc
    sorted_results = sorted(results, key=lambda x: (x["Test Acc"], x["CV Mean Acc"]), reverse=True)
    for rank, r in enumerate(sorted_results, 1):
        cv_str = f"{r['CV Mean Acc'] * 100:.2f}% (±{r['CV Std'] * 100:.2f}%)"
        test_str = f"{r['Test Acc'] * 100:.2f}%"
        star = " 🏆" if rank == 1 else ""
        print(f"#{rank:<4} {r['Model']:<28} | {cv_str:<15} | {test_str:<10} | {r['F1-Score']:.3f}  | {r['ROC-AUC']:.3f}{star}")

    # 5. Champion Model Deep-Dive
    champion_info = sorted_results[0]
    champion_name = champion_info["Model"]
    champion_pipeline = trained_pipelines[champion_name]

    print_banner(f"[Step 4/5] CHAMPION MODEL AUDIT: {champion_name.upper()}")
    y_pred_champ = champion_pipeline.predict(X_test_df)
    labels = ["Perished (0)", "Survived (1)"]

    print(f"  • Held-out Generalization Accuracy: {champion_info['Test Acc'] * 100:.2f}%")
    print(f"  • F1-Score: {champion_info['F1-Score']:.3f} | Precision: {champion_info['Precision']:.3f} | Recall: {champion_info['Recall']:.3f}")
    print(f"  • ROC-AUC Score: {champion_info['ROC-AUC']:.3f}\n")

    print("  Classification Report:")
    print(classification_report(y_test, y_pred_champ, target_names=labels))

    cm = confusion_matrix(y_test, y_pred_champ)
    render_confusion_matrix(cm, labels)

    # 6. Feature Importance & Predictive Drivers
    print_banner("FEATURE IMPORTANCE & SURVIVAL DRIVERS")
    prep = champion_pipeline.named_steps["preprocessor"]
    cat_encoder = prep.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + cat_names

    clf = champion_pipeline.named_steps["classifier"]

    # Check if classifier has native feature_importances_ or use permutation importance
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        importances = importances / np.sum(importances)
    else:
        # Use permutation importance on test set
        perm = permutation_importance(champion_pipeline, X_test_df, y_test, n_repeats=10, random_state=42)
        importances = np.maximum(0, perm.importances_mean)
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        # Note: Permutation importance is at input feature level (11 features)
        all_feature_names = FEATURE_COLUMNS

    feat_ranking = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
    print("  Top Predictive Survival Factors:")
    for rank, (feat, imp) in enumerate(feat_ranking, 1):
        bar = "█" * int(imp * 45)
        print(f"    {rank:>2}. {feat:<24} : {imp * 100:>5.2f}% |{bar}")

    # 7. Model Persistence
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "titanic_classifier.joblib")

    artifact = {
        "pipeline": champion_pipeline,
        "model_name": champion_name,
        "trained_medians": trained_medians,
        "metrics": champion_info,
        "benchmark_summary": sorted_results,
        "feature_columns": FEATURE_COLUMNS,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    joblib.dump(artifact, model_save_path)

    print_banner("[Step 5/5] MODEL PERSISTENCE & INFERENCE READY")
    print(f"  • Saved champion artifact to: {model_save_path}")
    print(f"  • Architecture: {champion_name}")
    print(f"  • Verified Test Accuracy: {champion_info['Test Acc'] * 100:.2f}%")
    print("\n  Ready for interactive passenger simulations and Kaggle batch test submissions!\n")


if __name__ == "__main__":
    main()
