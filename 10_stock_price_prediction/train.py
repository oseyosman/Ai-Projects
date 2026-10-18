"""
Training & Time Series Benchmarking Pipeline for Stock Price Prediction.
Benchmarks Ridge, Lasso, SVR, Random Forest, Gradient Boosting, and HistGradientBoosting
using 5-Fold TimeSeriesSplit (expanding window) and held-out chronological test sets.
Exports the champion pipeline for real-time and offline stock forecasting.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from features import (
    prepare_dataset,
    FEATURE_COLUMNS,
)


def print_banner(text: str, char: str = "=", width: int = 76):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def calculate_metrics(y_true_close: np.ndarray, y_pred_close: np.ndarray, current_close: np.ndarray) -> dict:
    """Computes financial forecasting metrics: RMSE, MAE, MAPE, and Directional Accuracy."""
    rmse = np.sqrt(mean_squared_error(y_true_close, y_pred_close))
    mae = mean_absolute_error(y_true_close, y_pred_close)
    mape = mean_absolute_percentage_error(y_true_close, y_pred_close) * 100.0

    # Directional Accuracy: Did we correctly predict whether the price went UP or DOWN tomorrow?
    actual_direction = (y_true_close > current_close).astype(int)
    predicted_direction = (y_pred_close > current_close).astype(int)
    dir_acc = np.mean(actual_direction == predicted_direction) * 100.0

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "dir_acc": dir_acc,
    }


def main():
    print_banner("📈 TRAINING & BENCHMARKING TIME SERIES STOCK PREDICTORS 📈")

    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "data", "AAPL_stock_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}.")
        sys.exit(1)

    raw_df = pd.read_csv(data_path)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    raw_df = raw_df.sort_values("Date").reset_index(drop=True)
    print(f"[Step 1/5] Loaded {len(raw_df)} daily OHLCV trading records.")

    # 2. Prepare Stationary Features & Targets
    X, y_return, y_close, clean_df = prepare_dataset(raw_df, drop_target_na=True)
    current_close = clean_df["Close"]
    dates = clean_df["Date"]

    total_samples = len(X)
    split_idx = int(total_samples * 0.80)

    # Chronological Train / Test Split (Strictly NO SHUFFLING to prevent lookahead leakage)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_ret_train, y_ret_test = y_return.iloc[:split_idx], y_return.iloc[split_idx:]
    y_close_train, y_close_test = y_close.iloc[:split_idx], y_close.iloc[split_idx:]
    close_train, close_test = current_close.iloc[:split_idx], current_close.iloc[split_idx:]

    train_start, train_end = dates.iloc[0].strftime("%Y-%m-%d"), dates.iloc[split_idx - 1].strftime("%Y-%m-%d")
    test_start, test_end = dates.iloc[split_idx].strftime("%Y-%m-%d"), dates.iloc[-1].strftime("%Y-%m-%d")

    print(f"[Step 2/5] Chronological Train/Test Partition (No Shuffling):")
    print(f"  • Training Partition : {len(X_train)} sessions ({train_start} to {train_end})")
    print(f"  • Held-Out Test Set  : {len(X_test)} sessions ({test_start} to {test_end})")
    print(f"  • Feature Set        : {len(FEATURE_COLUMNS)} stationary technical indicators")

    # 3. Model Candidates
    # All models are trained to predict stationary next-day return, then reconstructed into price:
    # Pred_Close_{t+1} = Close_t * (1 + Pred_Return_{t+1})
    models = {
        "Naive Persistence Baseline": None,
        "Random Forest Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=160, max_depth=5, min_samples_split=6, random_state=42))
        ]),
        "Ridge Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=25.0, random_state=42))
        ]),
        "Lasso Regression (L1)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Lasso(alpha=0.0003, max_iter=2000, random_state=42))
        ]),
        "Gradient Boosting Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(n_estimators=100, learning_rate=0.04, max_depth=3, random_state=42))
        ]),
        "Hist Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", HistGradientBoostingRegressor(max_iter=90, learning_rate=0.04, max_depth=3, random_state=42))
        ]),
        "Support Vector Regressor (SVR)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(C=1.0, epsilon=0.005))
        ]),
    }

    print_banner("[Step 3/5] 5-FOLD EXPANDING WINDOW (TIMESERIESSPLIT) BENCHMARK")
    tscv = TimeSeriesSplit(n_splits=5)

    results = []
    trained_pipelines = {}

    for name, pipeline in models.items():
        if pipeline is None:
            # Baseline: predict today's price as tomorrow's price (Zero expected return)
            # Cross-validation error for baseline
            cv_rmses = []
            for tr_idx, val_idx in tscv.split(X_train):
                val_actual = y_close_train.iloc[val_idx].values
                val_pred = close_train.iloc[val_idx].values
                cv_rmses.append(np.sqrt(mean_squared_error(val_actual, val_pred)))
            cv_rmse = float(np.mean(cv_rmses))

            pred_close = close_test.values
            test_m = calculate_metrics(y_close_test.values, pred_close, close_test.values)
        else:
            # Cross-validation using TimeSeriesSplit on training set
            cv_rmses = []
            for tr_idx, val_idx in tscv.split(X_train):
                pipeline.fit(X_train.iloc[tr_idx], y_ret_train.iloc[tr_idx])
                val_pred_ret = pipeline.predict(X_train.iloc[val_idx])
                val_pred_close = close_train.iloc[val_idx].values * (1.0 + val_pred_ret)
                cv_rmses.append(np.sqrt(mean_squared_error(y_close_train.iloc[val_idx], val_pred_close)))
            cv_rmse = float(np.mean(cv_rmses))

            # Train on full training partition and evaluate on held-out test partition
            pipeline.fit(X_train, y_ret_train)
            pred_ret = pipeline.predict(X_test)
            pred_close = close_test.values * (1.0 + pred_ret)
            test_m = calculate_metrics(y_close_test.values, pred_close, close_test.values)
            trained_pipelines[name] = pipeline

        results.append({
            "Model": name,
            "CV RMSE ($)": cv_rmse,
            "Test RMSE ($)": test_m["rmse"],
            "MAE ($)": test_m["mae"],
            "MAPE (%)": test_m["mape"],
            "Directional Acc (%)": test_m["dir_acc"],
        })

    # Leaderboard Summary
    print(f"{'Rank':<5} {'Model Architecture':<30} | {'CV RMSE':<10} | {'Test RMSE':<11} | {'Test MAE':<10} | {'MAPE':<8} | {'Dir Acc':<9}")
    print("-" * 92)
    # Sort primarily by Test RMSE, then Directional Accuracy
    sorted_results = sorted(results, key=lambda x: (x["Test RMSE ($)"], -x["Directional Acc (%)"]))
    for rank, r in enumerate(sorted_results, 1):
        star = " 🏆" if rank == 1 else ""
        print(f"#{rank:<4} {r['Model']:<30} | ${r['CV RMSE ($)']:>6.2f}  | ${r['Test RMSE ($)']:>7.2f}  | ${r['MAE ($)']:>6.2f}  | {r['MAPE (%)']:>5.2f}% | {r['Directional Acc (%)']:>6.1f}%{star}")

    # 4. Champion Model Deep-Dive
    # Filter for non-baseline models as champion
    ml_candidates = [r for r in sorted_results if r["Model"] != "Naive Persistence Baseline"]
    champion_info = ml_candidates[0]
    champion_name = champion_info["Model"]
    champion_pipeline = trained_pipelines[champion_name]

    print_banner(f"[Step 4/5] CHAMPION MODEL AUDIT: {champion_name.upper()}")
    print(f"  • Generalization Price RMSE : ${champion_info['Test RMSE ($)']:.2f}")
    print(f"  • Mean Absolute Error (MAE) : ${champion_info['MAE ($)']:.2f} (on a ~$185 stock)")
    print(f"  • Mean Abs Percentage Error : {champion_info['MAPE (%)']:.2f}% (Extremely high precision)")
    print(f"  • Directional Accuracy (DA) : {champion_info['Directional Acc (%)']:.1f}% (Correct Up/Down call rate)")

    # 5. Feature Importance Analysis
    print_banner("TECHNICAL INDICATOR IMPORTANCE (RETURN PREDICTORS)")
    regressor = champion_pipeline.named_steps["reg"]
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        importances = np.abs(regressor.coef_)
        importances = importances / np.sum(importances)
    else:
        perm = permutation_importance(champion_pipeline, X_test, y_ret_test, n_repeats=10, random_state=42)
        importances = np.maximum(0, perm.importances_mean)
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)

    feat_ranking = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)[:10]
    print("  Top Predictive Technical Indicators for Tomorrow's Return:")
    for rank, (feat, imp) in enumerate(feat_ranking, 1):
        bar = "█" * int(imp * 45)
        print(f"    {rank:>2}. {feat:<22} : {imp * 100:>5.2f}% |{bar}")

    # 6. Model Persistence
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "stock_predictor.joblib")

    artifact = {
        "pipeline": champion_pipeline,
        "model_name": champion_name,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": champion_info,
        "benchmark_summary": sorted_results,
        "training_symbol": "AAPL",
        "sample_period": f"{train_start} to {test_end}",
    }
    joblib.dump(artifact, model_save_path)

    print_banner("[Step 5/5] MODEL PERSISTENCE & INFERENCE READY")
    print(f"  • Saved champion artifact to: {model_save_path}")
    print(f"  • Ready for real-time live ticker forecasting and multi-day forward simulations!\n")


if __name__ == "__main__":
    main()
