"""
Training and Evaluation Pipeline for Michigan House Price Prediction.
Handles categorical features (city) and numerical house attributes.
Compares Linear Regression, Ridge, Decision Tree, Random Forest, and Gradient Boosting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "michigan_housing.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_dataset():
    if not os.path.exists(DATA_PATH):
        print("Data not found, generating Michigan dataset...")
        import generate_michigan_data
        generate_michigan_data.main()
    return pd.read_csv(DATA_PATH)


def train_models():
    print("=" * 70)
    print(" 🌲 Michigan House Price Prediction - Model Benchmark & Training")
    print("=" * 70)

    # 1. Load Data
    df = load_dataset()
    print(f"\n[Step 1/4] Loaded {len(df):,} Michigan residential properties.")

    X = df.drop(columns=["price"])
    y = df["price"]

    categorical_features = ["city"]
    numerical_features = [
        "square_feet", "bedrooms", "bathrooms", "lot_size_sqft",
        "year_built", "garage_spaces", "has_basement",
        "has_lake_access", "school_rating"
    ]

    # 2. Train / Test Split
    print("\n[Step 2/4] Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # 3. Model Candidates
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=10.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=16, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    }

    results = []
    trained_pipelines = {}

    print("\n[Step 3/4] Training candidate regression models...")
    print("-" * 70)

    for name, regressor in candidate_models.items():
        print(f"Training {name:<24}...", end="", flush=True)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            "Model": name,
            "R2 Score": r2,
            "MAE ($)": mae,
            "RMSE ($)": rmse
        })
        trained_pipelines[name] = pipe
        print(f" Done! (R²: {r2:.4f} | MAE: ${mae:,.0f})")

    # Present Leaderboard
    df_results = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    print("\n" + "=" * 70)
    print(" 🏆 Model Performance Leaderboard")
    print("=" * 70)
    for idx, row in df_results.reset_index(drop=True).iterrows():
        prefix = " ⭐ BEST -> " if idx == 0 else "           "
        print(f"{prefix}{row['Model']:<22} | R²: {row['R2 Score']:.4f} | MAE: ${row['MAE ($)']:>8,.0f} | RMSE: ${row['RMSE ($)']:>8,.0f}")
    print("=" * 70)

    # 4. Save the winning model
    best_row = df_results.iloc[0]
    best_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_name]

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "best_michigan_house_model.joblib")

    artifact = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "known_cities": sorted(df["city"].unique().tolist()),
        "metrics": {
            "r2": best_row["R2 Score"],
            "mae": best_row["MAE ($)"],
            "rmse": best_row["RMSE ($)"]
        }
    }
    joblib.dump(artifact, save_path)
    print(f"\n[Step 4/4] Successfully saved winning model ({best_name}) to:")
    print(f" -> {save_path}")

    # Top feature importances if tree-based
    if best_name in ["Gradient Boosting", "Random Forest"]:
        reg = best_pipeline.named_steps["regressor"]
        prep = best_pipeline.named_steps["preprocessor"]
        ohe = prep.named_transformers_["cat"]
        cat_encoded_names = list(ohe.get_feature_names_out(categorical_features))
        all_features = numerical_features + cat_encoded_names

        importances = reg.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:8]

        print("\nTop 8 Most Important Value Drivers in Michigan:")
        for rank, idx in enumerate(sorted_indices, 1):
            bar = "■" * int(importances[idx] * 35)
            print(f"  {rank}. {all_features[idx]:<20}: {importances[idx]:.3f} {bar}")

    print("\nReady! Run `python3 predict.py` to estimate prices for any Michigan house.")


if __name__ == "__main__":
    train_models()
